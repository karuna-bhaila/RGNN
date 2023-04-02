import copy
import sys

import torch
import torch.nn.functional as F
from torch.nn import Dropout, ReLU

from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv, GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_sparse import matmul, SparseTensor, fill_diag


class Classification(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.layer = torch.nn.Linear(hidden_channels, out_channels)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        y = F.softmax(self.layer(embeds), dim=1)
        return y


class GNN(torch.nn.Module):
    def __init__(self, _args):
        super().__init__()
        self.num_layers = _args.num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(p=_args.dropout)
        self.activation = ReLU(inplace=True)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

    def init_weights(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)


class SAGE(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GraphSAGE'

        for i in range(0, self.num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == self.num_layers-1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=False, root_weight=True))


class GCN(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GCN'

        for i in range(0, self.num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == self.num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_channels, hidden_channels, dropout=_args.dropout))


class GAT(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GAT'

        heads = 4
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads=1, concat=False))


class FeatureReconstructor(MessagePassing):
    def __init__(self, _args, self_loops=True):
        super().__init__(aggr='mean')

        self.num_layers = _args.xhops
        self.m = _args.m
        self.self_loops = self_loops

    def forward(self, data):
        if self.num_layers <= 0:
            return data

        x = data.x
        n, d = x.shape
        edge_index = data.edge_index

        if self.self_loops:
            if isinstance(data.edge_index, SparseTensor):
                edge_index = fill_diag(edge_index, 1.)
            else:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        else:
            edge_index = data.edge_index

        # Propagate in k-hop neighborhood to get proportions
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x)

        # Obtain unbiased estimates
        if self.m is None:
            P = data.P_X.repeat(n, 1).to(x.device)
            pi_x = x.add(torch.sub(P, 1))
            pi_x.true_divide_(torch.sub(P.mul(2), 1))
            data.x = pi_x

        elif self.m > 0:
            P = data.P_X.repeat(n, 1).to(x.device)
            pi_x = x.sub(0.5)
            pi_x.true_divide_(torch.sub(P.mul(2), 1))
            pi_x = pi_x.mul(d/self.m)
            pi_x = pi_x.add(0.5)
            data.x = pi_x

        return data

    def message_and_aggregate(self, adj_t, x):  # noqa
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class LabelReconstructor(MessagePassing):
    def __init__(self, _args, self_loops=True):
        super().__init__(aggr='mean')
        self.num_layers = _args.yhops
        self.self_loops = self_loops

    @torch.no_grad()
    def forward(self, data, train_and_val=True):

        if self.num_layers <= 0:
            return data

        y = data.y
        edge_index = data.edge_index
        mask = data.train_mask | data.val_mask if train_and_val else data.train_mask
        edge_weight = None

        if y.dtype == torch.long and y.size(0) == y.numel():
            y = F.one_hot(y.view(-1)).to(torch.float)

        n, m = y.size()

        out = torch.zeros_like(y)
        out[mask] = y[mask]

        if self.self_loops:
            if isinstance(data.edge_index, SparseTensor):
                edge_index = fill_diag(edge_index, 1.)
            else:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        else:
            edge_index = data.edge_index

        for _ in range(self.num_layers):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)

        # reconstruction
        y_r = torch.matmul(torch.inverse(data.P_Y).add(0.0), out.t()).t()
        y_r[y_r < 0] = 1e-5
        y_r.true_divide_(torch.sum(y_r, dim=1, keepdim=True))
        y_r = F.one_hot(y_r.argmax(dim=1), num_classes=m)
        y_r = y_r.float()

        data.y[mask] = y_r[mask]

        return data

    def message(self, x_j, edge_weight): # noqa
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x): # noqa
        return matmul(adj_t, x, reduce=self.aggr)




