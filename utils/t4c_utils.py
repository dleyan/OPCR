import torch
import tqdm
import numpy as np
from pathlib import Path
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.t4c22_config import *


def get_statistics(city, DATADIR):
    """
    :param city: city name
    :param day_t: list of valid time
    :return: data_statistics: dict   {"max", "min", "mean", "std", "processed_max", "processed_min"}
    """

    if city == "london":
        return {'max': 5664.0, 'min': 0.0, 'mean': 358.5881411332204, 'std': 290.43725310680674, 'processed_max': 18.2669812571039, 'processed_min': -1.2346492651937855}
    elif city == "madrid":
        return {'max': 60521.0, 'min': 0.0, 'mean': 441.2743147456362, 'std': 765.0456026450631, 'processed_max': 78.53090780148942, 'processed_min': -0.5767947861146807}
    elif city == "melbourne":
        return {'max': 10701.0, 'min': 0.0, 'mean': 176.38072533703473, 'std': 217.4335966127026, 'processed_max': 48.40383196810953, 'processed_min': -0.8111935233781185}

    day_t = [(day, t) for day in cc_dates(DATADIR, city=city, split="train") for t in range(4, 96)]

    valid_day = set([day for (day, _) in day_t])
    max_t = max([t for (_, t) in day_t])
    min_t = min([t for (_, t) in day_t])

    x_list = []
    for day in tqdm.tqdm(valid_day, total=len(valid_day)):
        fn = f"counters_{day}.parquet"
        df = pq.read_table(Path(DATADIR / "train" / city / "input" / fn)).to_pandas()
        df = df[(df["t"] >= min_t) & (df["t"] <= max_t)]
        data = df["volumes_1h"].values

        for i in range(len(data)):
            if not np.any(np.isnan(data[i])):
                x_list.extend(data[i])

    x = np.array(x_list)
    mean_v = np.nanmean(x)
    std_v = np.nanstd(x)
    max_v = np.nanmax(x)
    min_v = np.nanmin(x)
    data_statistics = {"max": max_v, "min": min_v, "mean": mean_v, "std": std_v}
    data_statistics["processed_max"] = (max_v - mean_v) / std_v
    data_statistics["processed_min"] = (min_v - mean_v) / std_v

    print({'max': data_statistics['max'],
           'min': data_statistics['min'],
           'mean': data_statistics['mean'],
           'std': data_statistics['std'],
           'processed_max': data_statistics["processed_max"],
           'processed_min': data_statistics["processed_min"]})

    return data_statistics


def get_t4c_dataset(args, DATADIR, cache="cache", impute_cache=None):

    cachedir = Path(DATADIR / cache)
    if impute_cache is None:
        impute_cachedir = None
    else:
        impute_cachedir = Path(DATADIR / impute_cache)

    select_edge_attrs = ["speed_kph", "parsed_maxspeed", "counter_distance", "importance", "highway", "oneway"]

    day_t_filter = None

    dataset = T4c22GeometricDataset(root=DATADIR,
                                    city=args.city,
                                    edge_attributes=select_edge_attrs,
                                    split="train",
                                    cachedir=cachedir,
                                    impute_cachedir=impute_cachedir,
                                    day_t_filter=day_t_filter)

    # get info
    args.num_nodes = NUM_NODES[args.city]
    args.num_edges = NUM_EDGES[args.city]
    args.num_counters = NUM_COUNTERS[args.city]

    # split dataset
    spl = int(((args.split * len(dataset)) // 2) * 2)
    train_dataset, val_dataset = \
        torch.utils.data.random_split(dataset, [spl, len(dataset) - spl])
    print("Train Dataset Size\t", len(train_dataset))
    print("Validation Dataset Size\t", len(val_dataset))

    test_dataset = T4c22GeometricDataset(root=DATADIR,
                                         city=args.city,
                                         edge_attributes=select_edge_attrs,
                                         split="test",
                                         cachedir=cachedir,
                                         impute_cachedir=impute_cachedir,
                                         day_t_filter=day_t_filter)

    print("================= DATASET INFO =================")
    print(f"Training Size: {len(train_dataset)}")
    print(f"Validation Size: {len(val_dataset)}")
    print(f"Testing Size: {len(test_dataset)}")

    return args, dataset, train_dataset, val_dataset, test_dataset


# normalize
def minmax(x):
    min_v = torch.min(x, dim=0).values
    max_v = torch.max(x, dim=0).values

    x = x - min_v
    x = x / (max_v - min_v)

    return x


# edge attributes
def get_edge_attr(df_edges, edge_attributes):
    """
    :param df_edges: data frame    num_row=num_edges
    :return: attrs: edge_attr     (num_edges, num_attrs)
    :return: num_attrs     (num_edges, num_attrs)
    """
    if edge_attributes is None:
        return None, 0

    num_edges = len(df_edges)

    speed_kph = torch.FloatTensor(df_edges["speed_kph"].values)
    parsed_maxspeed = torch.FloatTensor(df_edges["parsed_maxspeed"].values)
    length_meters = torch.FloatTensor(df_edges["length_meters"].values)
    counter_distance = torch.FloatTensor(df_edges["counter_distance"].values)

    # highway -> one hot
    highway_list = []
    for temp in df_edges["highway"].unique():
        if "[" not in temp:
            highway_list.append(temp)
        else:
            for temp1 in temp[1:-1].split(", "):
                highway_list.append(temp1)
    highway_dict = {}
    for i, key in enumerate(set(highway_list)):
        highway_dict[key] = i

    highway = torch.zeros(size=(num_edges, len(highway_dict)), dtype=torch.float)
    for i, temp in enumerate(df_edges["highway"]):
        if "[" not in temp:
            highway[i, highway_dict[temp]] = 1
        else:
            for temp1 in temp[1:-1].split(", "):
                highway[i, highway_dict[temp1]] = 1

    # oneway -> one hot
    oneway = torch.zeros(size=(num_edges, 1), dtype=torch.float)
    oneway[df_edges["oneway"].values] = 1

    # lanes -> one hot
    lanes = torch.zeros((num_edges, 1 + int(max(df_edges["lanes"].values))), dtype=torch.float)
    lanes[[i for i in range(num_edges)], [int(j) for j in df_edges["lanes"].values]] = 1

    # tunnel -> one hot
    tunnel = torch.zeros((num_edges, 1 + int(max(df_edges["tunnel"].values))), dtype=torch.float)
    tunnel[[i for i in range(num_edges)], [int(j) for j in df_edges["tunnel"].values]] = 1

    # importance -> one hot
    importance = torch.zeros((num_edges, 1 + int(max(df_edges["importance"].values))), dtype=torch.float)
    importance[[i for i in range(num_edges)], [int(j) for j in df_edges["importance"].values]] = 1

    edge_attr = []
    for attr in edge_attributes:
        if attr == "speed_kph":
            edge_attr.append(minmax(speed_kph).unsqueeze(-1))
        elif attr == "parsed_maxspeed":
            edge_attr.append(minmax(parsed_maxspeed).unsqueeze(-1))
        elif attr == "importance":
            edge_attr.append(importance)
        elif attr == "highway":
            edge_attr.append(highway)
        elif attr == "oneway":
            edge_attr.append(oneway)
        elif attr == "lanes":
            edge_attr.append(lanes)
        elif attr == "tunnel":
            edge_attr.append(tunnel)
        elif attr == "length_meters":
            edge_attr.append(minmax(length_meters).unsqueeze(-1))
        elif attr == "counter_distance":
            edge_attr.append(minmax(counter_distance).unsqueeze(-1))

    attrs = torch.cat(edge_attr, dim=1)
    num_attrs = attrs.shape[-1]

    return attrs, num_attrs


# node-node graph -> edge-edge graph
def get_edge_graph(n2n_index):
    """
    :param n2n_index: n2n_index    (2, num_edges)
    :return: e2e_index   (2, num_edges_in_e2e)
    :return: e2e_attr_index   num_edges_in_e2e*(central node index)
    """
    num_edges = n2n_index.shape[1]
    node2edge_dict = {}
    for i in tqdm.tqdm(range(num_edges), "preprocess edge attributes", total=num_edges):
        if n2n_index[0, i] not in node2edge_dict:
            node2edge_dict[n2n_index[0, i]] = [i]
        else:
            node2edge_dict[n2n_index[0, i]].append(i)

        if n2n_index[1, i] not in node2edge_dict:
            node2edge_dict[n2n_index[1, i]] = [i]
        else:
            node2edge_dict[n2n_index[1, i]].append(i)

    index = []      # (2, num_edges_in_edge_edge)
    attr_index = []         # central_nodes

    for node, edge_list in node2edge_dict.items():

        for i in edge_list:
            for j in edge_list:
                index.append([i, j])
                attr_index.append(node)

    e2e_index = torch.LongTensor(index).transpose(0, 1)
    e2e_attr_index = torch.LongTensor(attr_index)

    return e2e_index, e2e_attr_index


def normalize_adj(n2n_index, normalize_type="single"):

    # bi edge
    bi_n2n_index = [[i, j] for i, j in zip(n2n_index[0, :], n2n_index[1, :]) if i > j] + \
        [[j, i] for i, j in zip(n2n_index[0, :], n2n_index[1, :]) if i > j]
    bi_n2n_index = torch.LongTensor(bi_n2n_index).transpose(0, 1)

    # sparse adj
    adj = torch.sparse_coo_tensor(indices=bi_n2n_index, values=torch.ones(bi_n2n_index.shape[1]))

    # adj + I
    eye_indices = torch.arange(adj.shape[0]).unsqueeze(0).repeat(2, 1)
    eye_values = torch.ones(adj.shape[0])
    eye_adj = torch.sparse_coo_tensor(indices=eye_indices, values=eye_values, size=adj.shape, device=adj.device)
    adj = eye_adj + adj

    # D
    degree = torch.sparse.sum(adj, 1).to_dense()     # (num_node)

    # D^(-1)A
    if normalize_type == "single":
        degree = torch.pow(degree, -1)
        d_hat = torch.sparse_coo_tensor(indices=eye_indices, values=degree, size=adj.shape, device=adj.device)
        norm_adj = torch.sparse.mm(d_hat, adj)

    # D^(-0.5)A_bD^(-0.5)
    else:
        degree = torch.pow(degree, -0.5)
        d_hat = torch.sparse_coo_tensor(indices=eye_indices, values=degree, size=adj.shape)
        norm_adj = torch.sparse.mm(d_hat, adj)
        norm_adj = torch.sparse.mm(norm_adj, d_hat)

    return norm_adj
