import os
import tqdm
import copy
import random
import numpy as np

import torch
import pytorch_lightning as pl
from tsl import logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.nn.metrics import MaskedMetric, MaskedMAE
from tsl.ops.imputation import add_missing_values
from tsl.utils.parser_utils import ArgParser
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from utils import *
from model import opcr


def get_model_classes(args, device):

    if args.model_name == 'opcr':
        base_model = opcr.Model
    else:
        raise ValueError(f'Model {args.model_name} not available.')

    model = base_model(in_dim=args.in_dim,
                    out_dim=args.in_dim,
                    in_len=args.window,
                    hidden_dim=args.hidden_dim,
                    time_dim=args.time_dim,
                    num_layers=args.num_layers,
                    s_layers=args.s_layers,
                    dropout=args.dropout,
                    num_nodes=args.num_nodes,
                    device=args.device)
        
    model = model.to(device)

    return model


def get_dataset(dataset_name: str, missing_rate: float, missing_type: str, seed: int):

    if dataset_name == "air36" or dataset_name == "air":
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')

    if missing_type == "point":
        p = missing_rate
        if dataset_name == "la":
            return add_missing_values(MetrLA(), p_fault=0, p_noise=p, min_seq=12, max_seq=12 * 4, seed=seed)
        elif dataset_name == 'bay':
            return add_missing_values(PemsBay(), p_fault=0, p_noise=p, min_seq=12, max_seq=12 * 4, seed=seed)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}.")
    elif missing_type == "spatial":
        p, q = missing_rate, 0.25     
        if dataset_name == "la":
            return add_t4c_missing(MetrLA(), p=p, q=q, seed=seed)
        elif dataset_name == 'bay':
            return add_t4c_missing(PemsBay(), p=p, q=q, seed=seed)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}.")
    else:
        raise ValueError("Invalid missing type.")
    

def add_t4c_missing(dataset: PandasDataset, p: float, q: float, seed: int):

    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)

    rand = random.random
    logger.info(f'Generating mask with base missing rate={p}')

    # point missing
    mask = rand(shape) < q
    # mask p*N nodes
    mask_node = rand((dataset.n_nodes))
    mask[:, mask_node<p, :] = True
        
    eval_mask = mask.astype('uint8')

    # Convert to missing values dataset
    assert isinstance(dataset, PandasDataset)

    # Dynamically inherit from MissingValuesDataset
    bases = tuple([dataset.__class__, MissingValuesMixin])
    cls_name = "MissingValues%s" % dataset.__class__.__name__
    dataset.__class__ = type(cls_name, tuple(bases), {})

    # Change dataset name
    dataset.name = "MissingValues%s" % dataset.name

    dataset.set_eval_mask(eval_mask)

    # Store evaluation mask params in dataset
    dataset.seed = seed
    dataset.random = random

    return dataset


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    parser.add_argument("--model-state", type=str, default='train')
    parser.add_argument("--model-name", type=str, default='opcr')
    parser.add_argument("--dataset-name", type=str, default='bay')

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    # config
    parser.add_argument('--window', type=int, default=24)     
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--missing-type', type=str, default="point", help="spatial, point")  
    parser.add_argument('--missing-rate', type=float, default=0.95)  

    parser.add_argument('--in-dim', type=int, default=1)
    parser.add_argument('--out-dim', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--s-layers', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
    parser.add_argument('--save-dir', type=str, default="./save/traffic")  

    args = parser.parse_args()

    return args


def preprocess(batch):

    x = (batch.input.x).transpose(1, 2)     # (B, T, N, F) -> (B, N, T, F)
    eval_mask = (batch.input.eval_mask).transpose(1, 2)     # (B, T, N, F) -> (B, N, T, F) True=missing
    x = torch.where(eval_mask, torch.zeros_like(x), x)

    y = (batch.target.y).transpose(1, 2)
    u = batch.input.u

    return x, y, u, eval_mask


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    # save
    save_path = f"{args.save_dir}/{args.dataset_name}/{args.model_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(dataset_name=args.dataset_name, missing_rate=args.missing_rate, missing_type=args.missing_type, seed=args.seed)

    # time embedding
    time_emb = dataset.datetime_encoded(['day', 'week']).values
    exog_map = {'global_temporal_encoding': time_emb}

    input_map = {'u': 'temporal_encoding', 'x': 'data'}
    args.time_dim = time_emb.shape[-1]

    adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                   include_self=False,
                                   force_symmetric=True)

    # instantiate dataset
    torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)
    
    args.num_nodes = torch_dataset.n_nodes

    node_embed = torch.arange(args.num_nodes).to(device)
    args.node_embed_dim = None
    edge_index = (torch_dataset.edge_index).to(device)

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len, test_len=args.test_len)

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size)
    dm.setup()

    # model
    model = get_model_classes(args, device)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    early_stopping = EarlyStopping(patience=40, verbose=True, path=f"{save_path}{args.dataset_name}_{args.missing_type}_{args.missing_rate}_{args.seed}.pt")

    if args.model_state == "train":
        for epoch in range(args.epochs):

            ########################################
            # training                             #
            ########################################
            model.train()
            pbar = tqdm.tqdm(dm.train_dataloader(batch_size=args.batch_size), total=len(dm.train_dataloader(batch_size=args.batch_size)))

            losses = []
            for batch in pbar:

                batch = batch.to(device)
                x, y, u, eval_mask = preprocess(batch)

                # impute
                x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=eval_mask)
                
                x_rec = batch.transform['y'].inverse_transform(x_rec)
                loss = loss_fn(x_rec, y, eval_mask)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.cpu().item())
                pbar.set_postfix(loss=loss.cpu().item())

            print(f"train_loss={np.mean(losses)} after epoch {epoch}")
            
            ########################################
            # validation                           #
            ########################################
            model.eval()
            losses = []
            with torch.no_grad():
                pbar = tqdm.tqdm(dm.val_dataloader(batch_size=args.batch_size), total=len(dm.val_dataloader(batch_size=args.batch_size)))
                for batch in pbar:

                    batch = batch.to(device)
                    x, y, u, eval_mask = preprocess(batch)

                    # impute
                    x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=eval_mask)

                    x_rec = batch.transform['y'].inverse_transform(x_rec)
                    loss = loss_fn(x_rec, y, eval_mask)

                    losses.append(loss.cpu().item())
                    pbar.set_postfix(loss=loss.cpu().item())

            print(f"val_loss={np.mean(losses)} after epoch {epoch}")
            early_stopping(np.mean(losses), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    else:
        ########################################
        # testing                              #
        ########################################
        model.load_state_dict(torch.load(f"{save_path}{args.dataset_name}_{args.missing_type}_{args.missing_rate}_{args.seed}.pt", map_location="cpu"))
        model = model.to(device)
        model.eval()

        metric = MaskedMAE(compute_on_step=False).to(device)
        losses = []
        with torch.no_grad():
            pbar = tqdm.tqdm(dm.test_dataloader(batch_size=args.batch_size), total=len(dm.test_dataloader(batch_size=args.batch_size)))
            for batch in pbar:

                batch = batch.to(device)
                x, y, u, eval_mask = preprocess(batch)

                # impute
                x_rec = model(node_embed=node_embed, x=x, ex=u, edge_index=edge_index, mask=eval_mask)

                y_hat = batch.transform['y'].inverse_transform(x_rec)

                for b in range(x.shape[0]):
                    losses.append(metric(y_hat[b, ], y[b, ], eval_mask[b, ]).item())

        print("######################## Testing score ########################")
        print(f"{args.model_name}: {args.dataset_name}-{args.missing_type}-{args.missing_rate}-{args.seed}")
        print(f"mae：{np.mean(losses)}")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
