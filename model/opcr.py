import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, SimpleConv
from .layers import PositionalEmbedding


class SpatialAttention(nn.Module):

    def __init__(self, num_nodes, in_len, hidden_dim, num_layers, dropout):

        super(SpatialAttention, self).__init__()
        
        self.in_lin = nn.Linear(in_len*hidden_dim, hidden_dim)

        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        self.gnns = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, aggr='mean')] * num_layers)

        # attention
        self.q_lin = nn.Linear(hidden_dim, hidden_dim) 
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.out_lin = nn.Linear(hidden_dim, in_len*hidden_dim)
    
    def add_pos_embedding(self, xf, nf, edge_index):

        B, N, T, _ = xf.shape

        xf = xf.reshape(B, N, -1)   # (B, N, T, F) -> (B, N, -1)
        xf = self.in_lin(xf)    # (B, N, -1) -> (B, N, F)
        xf = F.gelu(xf)

        # compute spatial embedding #
        nf = self.node_embedding(nf)    # (N, F)
        for gnn in self.gnns:
            nf = gnn(nf, edge_index)
            nf = F.gelu(nf)

        return xf, nf
    
    def get_mask(self, mask):

        B, N, T, _ = mask.shape   # (B, N, T, F)

        # compute spatial masking #
        s_masking = torch.all(mask.reshape(B, N, -1), dim=-1)   # (B, N, T, F) -> (B, N, -1) -> (B, N)
        s_masking = s_masking.unsqueeze(1).repeat(1, N, 1)     # (B, N) -> (B, 1, N) -> (B, N, N)

        return s_masking
    
    def compute_conf(self, scores, s_masking):
        
        # (B, N, N)
        conf_scores = scores.clone().detach()

        all_scores = torch.softmax(conf_scores, dim=-1)
        all_scores = all_scores * s_masking     # missing confidence

        conf = 1-torch.sum(all_scores, dim=-1)      # (B, N)

        return conf
    
    def compute_A(self, xf, nf, mask):

        B, N, _ = xf.shape
        
        # compute spatial attention #
        q = self.q_lin(nf)      # (N, F)
        k = self.k_lin(nf)
        v = self.v_lin(xf)      # (B, N, F)

        D = q.shape[-1]
        scale = 1. / np.sqrt(D)

        scores = torch.mm(q, k.transpose(0, 1))    # (N, N) 
        scores = scale * scores

        scores = scores.unsqueeze(0).repeat(B, 1, 1)    # (N, N) -> (1, N, N) -> (B, N, N)
        s_masking = self.get_mask(mask)

        conf = self.compute_conf(scores, s_masking)

        scores = scores.masked_fill_(s_masking, -np.inf)

        # compute attention & value matrix #
        A = self.dropout(torch.softmax(scores, dim=-1))
        A = A.nan_to_num(0)

        rec = torch.bmm(A, v)   # (B, N, N) * (B, N, F) -> (B, N, F)

        return rec, conf
    
    def forward(self,
                node_embed: torch.Tensor,     # (N, F)
                xf: torch.Tensor,        # (B, N, T, F)
                edge_index: torch.Tensor,   # (2, E)
                mask: torch.Tensor    # (B, N, T, F) True=missing
    ):
        B, N, T, _ = xf.shape

        # add position embedding  #
        xf, nf = self.add_pos_embedding(xf=xf, nf=node_embed, edge_index=edge_index)

        rec, conf = self.compute_A(xf=xf, nf=nf, mask=mask)

        rec = self.out_lin(rec)    # (B, N, F) -> (B, N, T*F)       
        rec = rec.reshape(B, N, T, -1)  # (B, N, T*F) -> (B, N, T, F)

        return rec, conf


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim, time_dim, num_nodes, dropout=0):

        super(TemporalAttention, self).__init__()

        self.in_lin = nn.Linear(hidden_dim*3, hidden_dim)

        self.pos_embedding = PositionalEmbedding(d_model=hidden_dim)
        self.time_embedding = nn.Linear(time_dim, hidden_dim)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        # attention
        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)

        self.out_lin = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
    
    def add_pos_embedding(self, xf, ex, nf):

        B, N, T, _ = xf.shape

        time_embed = self.time_embedding(ex).unsqueeze(1)       # (B, 1, T, F)
        node_embed = self.node_embedding(nf).unsqueeze(0).unsqueeze(2)    # (N, F) -> (1, N, 1, F)

        pos_embed = self.pos_embedding(xf)
        xf = xf + pos_embed

        xf = self.in_lin(torch.cat([xf, time_embed.repeat(1, N, 1, 1), node_embed.repeat(B, 1, T, 1)], dim=-1))
        xf = F.gelu(xf)

        return xf

    def compute_A(self, xf, mask):

        q = self.q_lin(xf)
        k = self.k_lin(xf)
        v = self.v_lin(xf)

        D = q.shape[-1]
        scale = 1. / np.sqrt(D)

        scores = torch.einsum("bnsf, bnft -> bnst", q, k.transpose(2, 3))  # (B, N, T, T)
        scores = scale * scores

        # compute attention & value matrix #
        A = torch.softmax(scores, dim=-1)
        A = A.nan_to_num(0)     # (B, N, T, T)
        
        t_masking = torch.all(mask, dim=-1).unsqueeze(2)     # (B, N, T) -> (B, N, 1, T)
        conf = 1-torch.sum(A.clone().detach() * t_masking, dim=-1)      # (B, N, T, T) -> (B, N, T)

        rec = torch.einsum("bnst, bntf -> bnsf", A, v)

        return rec, conf
    
    def forward(self, xf, ex, nf, mask):      # True=missing

        B, N, T, _ = xf.shape       # (B, N, T, F)

        # add position embedding
        xf = self.add_pos_embedding(xf, ex, nf)

        # compute temporal attention
        rec, conf = self.compute_A(xf=xf, mask=mask)

        rec = self.out_lin(rec)

        return rec, conf


class PropagationLayer(nn.Module):
    
    def __init__(self, hidden_dim, in_len, device):

        super(PropagationLayer, self).__init__()

        self.lin = nn.Linear(hidden_dim, hidden_dim)

        self.lin_s = nn.Linear(hidden_dim, hidden_dim)
        self.lin_t = nn.Linear(hidden_dim, hidden_dim)

        self.gnn = SimpleConv(aggr="mean")

        ones = torch.ones(in_len, in_len).to(device)
        self.t_masking = torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2) + torch.triu(ones, diagonal=-1) - torch.triu(ones, diagonal=0)
        self.t_masking = self.t_masking / torch.sum(self.t_masking, dim=-1, keepdim=True)

        self.out_lin = nn.Linear(hidden_dim*3, hidden_dim)

    def t_prop(self, xf, conf=None):

        xf = self.lin_t(xf)
        
        xf = xf * conf.unsqueeze(-1)
        t_xf = torch.einsum("st, bntf -> bnsf", self.t_masking, xf)       # (B, N, T, F) 
                
        return t_xf
    
    def s_prop(self, xf, edge_index, conf=None):

        xf = self.lin_s(xf)     # (B, N, T, F)

        xf = xf.transpose(1, 2)     # (B, N, T, F) -> (B, T, N, F)
        conf = conf.transpose(1, 2)     # (B, N, T) -> (B, T, N)
        xf = xf * conf.unsqueeze(-1)
        s_xf = self.gnn(xf, edge_index=edge_index)

        s_xf = s_xf.transpose(1, 2)

        return s_xf
    
    def forward(self, xf, conf, edge_index):

        t_xf = self.t_prop(xf=xf, conf=conf)

        s_xf = self.s_prop(xf, edge_index=edge_index, conf=conf)

        xf = self.lin(xf)
        
        fin_xf = self.out_lin(torch.cat([xf, t_xf, s_xf], dim=-1))
        
        return fin_xf


class Model(nn.Module):

    def __init__(self, num_nodes, in_len, in_dim, out_dim, hidden_dim, s_layers, num_layers, time_dim, device, dropout=0):

        super(Model, self).__init__()

        self.in_lin = nn.Linear(in_dim, hidden_dim)

        self.s_attn = SpatialAttention(
                            num_nodes=num_nodes,
                            in_len=in_len,
                            hidden_dim=hidden_dim,
                            num_layers=s_layers,
                            dropout=dropout)
        
        self.t_attn = TemporalAttention(
                            hidden_dim=hidden_dim,
                            time_dim=time_dim,
                            num_nodes=num_nodes,
                            dropout=dropout)
        
        self.stage1_lin = nn.Linear(hidden_dim, hidden_dim)

        self.stage2_lin = nn.Linear(hidden_dim, out_dim)

        self.stage2 = nn.ModuleList([PropagationLayer(hidden_dim=hidden_dim, in_len=in_len, device=device)] * num_layers)

    def forward(self,
                node_embed: torch.Tensor,     # (N, F)
                x: torch.Tensor,        # (B, N, T, F)
                ex: torch.Tensor,
                edge_index: torch.Tensor,   # (2, E)
                mask: torch.Tensor,    # (B, N, T, F) True=missing
                ):
        
        B, N, T, _ = x.shape   # (B, N, T, F)
                                 
        xf = self.in_lin(x)
        xf = F.gelu(xf)

        s_rec, s_conf = self.s_attn(xf=xf, mask=mask, node_embed=node_embed, edge_index=edge_index)
        t_rec, t_conf = self.t_attn(xf=xf, ex=ex, mask=mask, nf=node_embed)

        s_conf = s_conf.unsqueeze(-1).repeat(1, 1, T)    # (B, N) -> (B, N, 1) -> (B, N, T)
        conf = (s_conf + t_conf) / 2
        conf[mask.squeeze(-1)==0] = 1

        out = F.gelu(s_rec+t_rec)
        out = self.stage1_lin(out)

        out = torch.where(mask, out, xf)
        for prop in self.stage2:
            out = prop(xf=out, conf=conf, edge_index=edge_index)
            out = F.gelu(out)
            out = torch.where(mask, out, xf)

        out = self.stage2_lin(out)

        return out
