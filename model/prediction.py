import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class EdgePrediction(nn.Module):

    def __init__(self, in_dim, num_nodes, num_edges, num_attrs, hidden_dim, out_dim, device, num_layers=3):

        super(EdgePrediction, self).__init__()

        self.node_idx = torch.arange(0, num_nodes).to(device)
        self.edge_idx = torch.arange(0, num_edges).to(device)

        self.edge_embed = nn.Embedding(num_edges, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)

        self.time_embed = nn.Embedding(96, hidden_dim)
        self.week_embed = nn.Embedding(7, hidden_dim)

        self.node_lin = nn.Linear(in_dim, hidden_dim)

        self.attr_lin1 = nn.Linear(num_attrs, hidden_dim)
        self.attr_lin2 = nn.Linear(num_attrs, hidden_dim)

        self.conv1 = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim) for _ in range(3)
        ])

        self.conv2 = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim) for _ in range(3)
        ])

        self.gnn_lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gnn_lin2 = nn.Linear(hidden_dim * 2, hidden_dim)

        dims = [hidden_dim * 6] + [hidden_dim] * num_layers + [out_dim]
        self.lins = nn.ModuleList([
            nn.Linear(dim1, dim2) for dim1, dim2 in zip(dims[:-1], dims[1:])
        ])

    def forward(self,
                x_rec: torch.Tensor,    # imputation
                edge_index: torch.Tensor,   # edge index (2, E)
                attr: torch.Tensor,     # edge attribution
                cur_t: torch.Tensor,        # time [0, 96)
                cur_w: torch.Tensor):       # weekday [0, 7)

        # attr embedding
        attr_f = self.attr_lin1(attr)

        # node embedding
        node_embed = self.node_embed(self.node_idx)
        pre_data = node_embed
        for conv in self.conv1:
            node_embed = conv(node_embed, edge_index, attr_f)
            node_embed = F.gelu(node_embed) + pre_data

        x_i = torch.index_select(node_embed, 0, edge_index[0])
        x_j = torch.index_select(node_embed, 0, edge_index[1])
        x1 = torch.cat([x_i, x_j], dim=1)
        x1 = self.gnn_lin1(x1)

        # node feature
        node_f = F.gelu(self.node_lin(x_rec))
        pre_data = node_f
        for conv in self.conv2:
            node_f = conv(node_f, edge_index, attr_f)
            node_f = F.gelu(node_f) + pre_data

        x_i = torch.index_select(node_f, 0, edge_index[0])
        x_j = torch.index_select(node_f, 0, edge_index[1])
        x2 = torch.cat([x_i, x_j], dim=1)
        x2 = self.gnn_lin2(x2)

        # time embedding
        time_embed = self.time_embed(cur_t).repeat(x2.shape[0], 1)
        week_embed = self.week_embed(cur_w).repeat(x2.shape[0], 1)

        edge_embed = self.edge_embed(self.edge_idx)

        xf = torch.cat([edge_embed,
                        self.attr_lin2(attr),
                        x1,
                        x2,
                        time_embed,
                        week_embed], dim=1)

        for lin in self.lins[:-1]:
            xf = lin(xf)
            xf = F.gelu(xf)

        out = self.lins[-1](xf)

        return out


class SegmentPrediction(nn.Module):

    def __init__(self, in_dim, num_nodes, num_edges, num_segments, num_attrs, hidden_dim, out_dim, device, num_layers=3):

        super(SegmentPrediction, self).__init__()

        self.node_idx = torch.arange(0, num_nodes).to(device)
        self.edge_idx = torch.arange(0, num_edges).to(device)
        self.segment_idx = torch.arange(0, num_segments).to(device)

        self.edge_embed = nn.Embedding(num_edges, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.segment_embed = nn.Embedding(num_segments, hidden_dim)

        self.time_embed = nn.Embedding(96, hidden_dim)
        self.week_embed = nn.Embedding(7, hidden_dim)

        self.node_lin = nn.Linear(in_dim, hidden_dim)

        self.attr_lin1 = nn.Linear(num_attrs, hidden_dim)
        self.attr_lin2 = nn.Linear(num_attrs, hidden_dim)

        self.conv1 = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim) for _ in range(3)
        ])

        self.conv2 = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim) for _ in range(3)
        ])

        self.conv3 = GATv2Conv(hidden_dim * 7, hidden_dim, edge_dim=1)

        self.gnn_lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gnn_lin2 = nn.Linear(hidden_dim * 2, hidden_dim)

        dims = [hidden_dim * 7] + [hidden_dim] * num_layers
        self.lins = nn.ModuleList([
            nn.Linear(dim1, dim2) for dim1, dim2 in zip(dims[:-1], dims[1:])
        ])
        self.out_lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self,
                x_rec: torch.Tensor,    # imputation
                edge_index: torch.Tensor,   # edge index
                attr: torch.Tensor,     # edge attribution
                segment_edge: torch.sparse.Tensor,
                segment_node: torch.sparse.Tensor,
                segment_segment: torch.sparse.Tensor,
                cur_t: torch.Tensor,        # time [0, 96)
                cur_w: torch.Tensor):       # weekday [0, 7)

        # attr embedding
        attr_f = self.attr_lin1(attr)

        # node embedding
        node_embed = self.node_embed(self.node_idx)
        pre_data = node_embed
        for conv in self.conv1:
            node_embed = conv(node_embed, edge_index, attr_f)
            node_embed = F.gelu(node_embed) + pre_data

        # node feature
        node_f = F.gelu(self.node_lin(x_rec))
        pre_data = node_f
        for conv in self.conv2:
            node_f = conv(node_f, edge_index, attr_f)
            node_f = F.gelu(node_f) + pre_data

        edge_embed = self.edge_embed(self.edge_idx)
        segment_embed = self.segment_embed(self.segment_idx)

        x1_edge = torch.sparse.mm(segment_edge, edge_embed)
        x2_edge = torch.sparse.mm(segment_edge, self.attr_lin2(attr))

        x1_node = torch.sparse.mm(segment_node, node_embed)
        x2_node = torch.sparse.mm(segment_node, node_f)

        # time embedding
        time_embed = self.time_embed(cur_t).repeat(x1_edge.shape[0], 1)
        week_embed = self.week_embed(cur_w).repeat(x1_edge.shape[0], 1)

        xf = torch.cat([segment_embed,
                        x1_edge,
                        x2_edge,
                        x1_node,
                        x2_node,
                        time_embed,
                        week_embed], dim=1)

        xf_gnn = self.conv3(xf, segment_segment._indices(), segment_segment._values())
        xf_gnn = F.gelu(xf_gnn)

        for lin in self.lins[:-1]:
            xf = lin(xf)
            xf = F.gelu(xf)

        out = self.out_lin(torch.cat([xf, xf_gnn], dim=1))
        out = F.sigmoid(out) * 3600

        return out
