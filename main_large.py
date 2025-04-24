import os
import random

import numpy as np
import torch
import tqdm
from tsl.nn.metrics import MaskedMetric, MaskedMAE
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import opcr
from utils import *


def get_model_classes(args, device):

    if args.model_name == 'opcr':
        base_model = opcr.Model
    else:
        raise ValueError(f'Model {args.model_name} not available.')
    
    model = base_model(in_dim=args.in_dim,
                  out_dim=args.in_dim,
                  in_len=args.in_len,
                  hidden_dim=args.hidden_dim,
                  time_dim=args.time_dim,
                  num_layers=args.num_layers,
                  s_layers=args.s_layers,
                  dropout=args.dropout,
                  num_nodes=args.num_nodes,
                  device=device)

    model = model.to(device)

    return model


def parse_args():
    # Argument parser
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    parser.add_argument('--model-name', type=str, default="opcr")
    parser.add_argument('--scaler', type=str, default='zscore')

    parser.add_argument('--in-len', type=int, default=24)
    parser.add_argument('--out-len', type=int, default=24)
    parser.add_argument('--step', type=int, default=1)

    # dataset
    parser.add_argument('--dataset-name', type=str, default="pvus", help="pvus, cer") 

    parser.add_argument('--missing-type', type=str, default="point", help="spatial, point")  
    parser.add_argument('--missing-rate', type=float, default=0.95)  

    parser.add_argument('--in-dim', type=int, default=1)
    parser.add_argument('--out-dim', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=128)

    parser.add_argument('--s-hidden-dim', type=int, default=128)

    parser.add_argument('--s-layers', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=3)

    parser.add_argument('--batch-size', type=int, default=4, help="batch_size")
    parser.add_argument('--epochs', type=int, default=200, help="epochs")
    parser.add_argument('--loss-fn', type=str, default='l1_loss')

    parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
    parser.add_argument('--model-state', type=str, default='train', help="train or test")

    parser.add_argument('--save-dir', type=str, default="./save/rec")
    parser.add_argument('--data-dir', type=str, default="./data")

    args = parser.parse_args()

    return args


def run_experiment(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)

    # load dataset  (N, T, F)
    if args.dataset_name == 'pvus':
        threshold = 30
        knn = 10
    elif args.dataset_name == 'cer':
        threshold = 0.8
        knn = 5
    else:
        ValueError(f"Invalid dataset name: {args.dataset_name}.")
    X, Ex, edge_index, edge_attr = get_dataset(data_dir=args.data_dir, dataset_name=args.dataset_name, threshold=threshold, knn=knn)
    mask_X, Mask = add_missing(X=X, missing_type=args.missing_type, missing_rate=args.missing_rate, seed=args.seed)

    # split train and test
    # pvus: train(2006-01-01 - 2006-04-30) val(2006-05-01 - 2006-05-31) test(2006-06-01 - 2006-06-30)
    # cer: train(2010-01-01 - 2010-04-30) val(2010-05-01 - 2010-05-31) test(2010-06-01 - 2010-06-30)
    args.train_len = 120 * 24 * 2    # 4 months
    args.val_len = 31 * 24 * 2    # 1 months
    args.test_len = 30 * 24 * 2    # 1 months

    if args.scaler == 'minmax':
        max_val = np.nanmax(mask_X[:, :args.train_len, ])
        min_val = np.nanmin(mask_X[:, :args.train_len, ])
        mask_X = (mask_X - min_val) / (max_val - min_val)
    elif args.scaler == 'zscore':
        mean_val = np.nanmean(mask_X[:, :args.train_len, ])
        std_val = np.nanstd(mask_X[:, :args.train_len, ])
        mask_X = (mask_X - mean_val) / std_val

    else:
        raise ValueError(f"Invalid scaler type: {args.scaler}.")
    
    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # numpy -> torch 
    # N, T, F -> T, N, F
    X = torch.from_numpy(X).transpose(0, 1)
    mask_X = torch.from_numpy(mask_X).transpose(0, 1)
    Mask = torch.from_numpy(Mask).transpose(0, 1)
    Ex = torch.from_numpy(Ex)

    args.num_nodes = X.shape[1]
    args.num_edges = edge_index.shape[-1]
    args.time_dim = Ex.shape[-1]

    node_embed = torch.arange(args.num_nodes).to(device)
    args.node_embed_dim = None

    edge_index = torch.from_numpy(edge_index).to(device)
    edge_attr = torch.from_numpy(edge_attr).to(device) 

    # get dataloader
    train_dataset, val_dataset, test_dataset = get_dataloader(args, X=X, mask_X=mask_X, Ex=Ex, Mask=Mask, TimeFirst=False)

    # save
    save_path = f"{args.save_dir}/{args.dataset_name}/{args.model_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # print parameters
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
        
    # model
    model = get_model_classes(args, device)

    # loss function
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn), compute_on_step=True, metric_kwargs={'reduction': 'none'}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{save_path}rec_{args.dataset_name}_{args.missing_type}_{args.missing_rate}_{args.seed}_{args.s_hidden_dim}_{args.num_layers}.pt")

    if args.model_state == 'train':
        # Training
        for epoch in tqdm.tqdm(range(args.epochs), "epochs", total=args.epochs):

            # train
            model.train()
            pbar = tqdm.tqdm(DataLoader(train_dataset, batch_size=args.batch_size), "train", total=len(train_dataset) // args.batch_size)

            losses = []
            for _, x, y, u, mask in pbar:

                x, y, u, mask = x.to(device), y.to(device), u.to(device), mask.to(device)   # B, N, T, F

                # impute
                x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=mask)
                
                x_rec = x_rec * std_val + mean_val
                loss = loss_fn(x_rec, y, mask)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.cpu().item())
                pbar.set_postfix(loss=loss.cpu().item())

            print(f"train_loss={np.mean(losses)} after epoch {epoch}")

            # validation
            model.eval()
            losses = []
            with torch.no_grad():
            
                pbar = tqdm.tqdm(DataLoader(val_dataset, batch_size=args.batch_size), "val", total=len(val_dataset) // args.batch_size)
                for _, x, y, u, mask in pbar:

                    x, y, u, mask = x.to(device), y.to(device), u.to(device), mask.to(device)   # B, T, N, F
                    
                    # impute
                    x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=mask)

                    x_rec = x_rec * std_val + mean_val
                    loss = loss_fn(x_rec, y, mask)

                    losses.append(loss.cpu().item())
                    pbar.set_postfix(loss=loss.cpu().item())

            print(f"val_loss={np.mean(losses)} after epoch {epoch}")
            early_stopping(np.mean(losses), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    else:
        model.load_state_dict(torch.load(f"{save_path}rec_{args.dataset_name}_{args.missing_type}_{args.missing_rate}_{args.seed}_{args.s_hidden_dim}_{args.num_layers}.pt", map_location="cpu"))

        model = model.to(device)

        losses = []
        metric = MaskedMAE(compute_on_step=False).to(device)
        with torch.no_grad():
            pbar = tqdm.tqdm(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False), "test", total=len(test_dataset)//args.batch_size)
            for _, x, y, u, mask in pbar:
                x, y, u, mask = x.to(device), y.to(device), u.to(device), mask.to(device)   # B, N, T, F

                # impute
                x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=mask)
                        
                if args.scaler == 'minmax':
                    test_y_hat = x_rec.detach().squeeze(-1) * (max_val - min_val) + min_val
                elif args.scaler == 'zscore':
                    test_y_hat = x_rec.detach().squeeze(-1) * std_val + mean_val
                else:
                    raise ValueError(f"Invalid scaler type: {args.scaler}.")

                test_y = y.squeeze(-1)
                eval_mask = torch.tensor(mask, dtype=torch.float32).squeeze(-1)
                for b in range(x.shape[0]):
                    losses.append(metric(test_y_hat[b, ], test_y[b, ], eval_mask[b, ]).item())

        print("######################## Testing score ########################")
        print(f"{args.model_name}: {args.dataset_name}-{args.missing_type}-{args.missing_rate}-{args.seed}")
        print(f"mae：{np.mean(losses)}")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
