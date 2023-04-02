import copy

import torch
from torch_geometric.utils import subgraph
from torch_geometric.loader import ClusterData

from torch_sparse import matmul, SparseTensor


def get_accuracy(pred, target):
    pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
    target = target.argmax(dim=1) if len(target.size()) > 1 else target

    acc = (pred == target).sum().item() / target.numel()

    return acc * 100


def get_k_hop_edge_index(edge_index, k, n):
    adj_sp = SparseTensor(row=edge_index[0], col=edge_index[1],
                          value=torch.ones(edge_index.shape[1], device=edge_index.device),
                          sparse_sizes=(n, n))
    k_adj_sp = copy.deepcopy(adj_sp)

    for i in range(k):
        k_adj_sp = matmul(k_adj_sp, adj_sp)

    row, col, _ = k_adj_sp.coo()
    edge_index = torch.stack([row, col], dim=0)

    return edge_index


# METIS clustering to get bags
def get_clusters(data, num_clusters, display=False):
    data.n_id = torch.arange(data.num_nodes)
    data.cluster_id = torch.full((data.num_nodes,), -1)

    cluster_data = ClusterData(data, num_parts=num_clusters, recursive=False, log=False)
    partptr = cluster_data.partptr.tolist()
    data.num_clusters = cluster_data.num_parts

    for i, subdata in enumerate(cluster_data):
        data.cluster_id[subdata.n_id] = i

        # save edge_index; node indices are reset in every cluster
        try:
            data.cluster_edge_index = torch.cat((data.cluster_edge_index, subdata.edge_index), dim=1)
        except:
            data.cluster_edge_index = subdata.edge_index

    # Generate a cluster mask for faster access
    for i in range(data.num_clusters):
        cluster_mask = torch.tensor([True if data.cluster_id[indx] == i
                                     else False for indx in range(data.num_nodes)]).to(data.y.device)
        cluster_mask = torch.unsqueeze(cluster_mask & data.train_mask, 0)

        try:
            data.cluster_mask = torch.cat((data.cluster_mask, cluster_mask), dim=0)
        except:
            data.cluster_mask = cluster_mask

    if display:
        num_nodes = []
        num_train = []
        for i, subdata in enumerate(cluster_data):
            num_nodes.append(subdata.num_nodes)
            num_train.append(subdata.train_mask.sum())

        print("No. of clusters: {}".format(data.num_clusters))
        print("No. of nodes in cluster: Avg:{}, Min:{}, Max:{}".format(int(sum(num_nodes)/len(num_nodes)),
                                                                       min(num_nodes),
                                                                       max(num_nodes)))
        print("No. of train nodes in cluster: Avg:{}, Min:{}, Max:{}".format(int(sum(num_train)/len(num_train)),
                                                                       min(num_train),
                                                                       max(num_train)))
    return data

