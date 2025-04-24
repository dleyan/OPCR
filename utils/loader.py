from torch.utils.data import Dataset
import numpy as np
from tsl.ops.similarities import top_k
import datetime


def get_dataloader(args, X, mask_X, Ex, Mask, TimeFirst=True):

    train_end = args.train_len
    val_end = args.train_len + args.val_len
    test_end = args.train_len + args.val_len + args.test_len

    val_start, test_start = train_end, val_end 

    train_dataset = Loader(X = X[:val_start, ],
                           mask_X = mask_X[:val_start, ],
                           Ex = Ex[:val_start, ],
                           Mask = Mask[:val_start, ],
                           window=args.in_len, step=args.step, TimeFirst=TimeFirst)

    val_dataset = Loader(X = X[val_start:val_end, ],
                         mask_X = mask_X[val_start:val_end, ],
                         Ex = Ex[val_start:val_end, ],
                         Mask = Mask[val_start:val_end, ],
                         window=args.in_len, step=args.step, TimeFirst=TimeFirst)

    test_dataset = Loader(X = X[test_start:test_end, ],
                          mask_X = mask_X[test_start:test_end, ],
                          Ex = Ex[test_start:test_end, ],
                          Mask = Mask[test_start:test_end, ],
                          window=args.in_len, step=args.step, TimeFirst=TimeFirst, model_state="test")

    return train_dataset, val_dataset, test_dataset


def get_dataset(data_dir: str, dataset_name: str, threshold: float, knn: int):

    if dataset_name == 'pvus':
        
        data = np.load(f"{data_dir}/pvus/pvus.npy")
        distance = np.load(f"{data_dir}/pvus/distance.npy")

        data = data.astype('float32')
        distance = -distance.astype('float32')

        # 5min -> 30min
        agg_data = data.reshape(data.shape[0], -1, 6)
        agg_data = np.mean(agg_data, axis=-1)

        time = datetime.datetime(2006, 1, 1, 0, 0, 0)

        # k adjacent nodes
        top_k_adj = top_k(distance, knn, include_self=True)
        # < 30km
        top_k_adj[distance >= -threshold] = 1
        top_k_adj[top_k_adj.T!=0] = 1
        distance[top_k_adj==0] = 0

        for i in range(distance.shape[0]):
            distance[i, i] = 0
        
        idxs = np.nonzero(distance.T)
        edge_index = np.stack(idxs)
        edge_weight = distance[idxs]    

    elif dataset_name == 'cer':

        data = np.load(f"{data_dir}/cer/cer.npy")
        distance = np.load(f"{data_dir}/cer/distance.npy")

        drop_len = (31+30+31+30+31+17) * 48
        agg_data = data[:, drop_len:].astype('float32')
        distance = distance.astype('float32')
        distance[np.isnan(distance)] = 0

        time = datetime.datetime(2010, 1, 1, 0, 0, 0)

        # k adjacent nodes
        top_k_adj = top_k(distance, knn, include_self=True)
        top_k_adj[top_k_adj.T!=0] = 1
        distance[top_k_adj==0] = 0

        for i in range(distance.shape[0]):
            distance[i, i] = 0
        
        idxs = np.nonzero(distance)
        edge_index = np.stack(idxs)
        edge_weight = distance[idxs]

    else:
        ValueError(f"Invalid dataset name: {dataset_name}.")
    
    weeks = []
    times = []
    for _ in range(agg_data.shape[1]):
        weeks.append(time.weekday())
        times.append(time.hour)
        time = time + datetime.timedelta(minutes=30)

    ex_feature = pre_time(weeks, times)
    ex_feature = ex_feature.astype('float32')

    length = 181 * 24 * 2
    
    return agg_data[:, :length, np.newaxis], ex_feature[:length, ], edge_index, edge_weight


def pre_time(weeks, hours):

    week_onehot = np.zeros((len(weeks), 7))
    hour_onehot = np.zeros((len(hours), 24))

    for idx, (w, h) in enumerate(zip(weeks, hours)):
        week_onehot[idx, w] = 1
        hour_onehot[idx, h] = 1

    return np.concatenate([week_onehot, hour_onehot], axis=-1)


class Loader(Dataset):
    def __init__(self, X, mask_X, Ex, Mask, window, step, TimeFirst=True, model_state="train"):
        super(Loader, self).__init__()

        self.window = window
        self.step = step
        
        self.X = X.nan_to_num(0)
        self.mask_X = mask_X.nan_to_num(0)

        self.Ex = Ex
        self.Mask = Mask

        self.TimeFirst = TimeFirst
        self.model_state = model_state

    def __len__(self):

        length = (len(self.X) - self.window) // self.step + 1

        return length

    def __getitem__(self, index):

        index *= self.step

        x = self.mask_X[index:index+self.window, ]
        u = self.Ex[index:index+self.window, ]
        mask = self.Mask[index:index+self.window, ]

        y = self.X[index:index+self.window, ]

        if self.TimeFirst:
            return index, x, y, u, mask
        else:
            return index, x.transpose(0, 1), y.transpose(0, 1), u, mask.transpose(0, 1)
