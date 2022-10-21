import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from torch_scatter import scatter
from gnn_utils import *

class WS(nn.Module):
    def __init__(self, n_in_nodes, in_dim, out_dim):
        super().__init__()
        self.w = nn.Parameter(th.rand(size=[n_in_nodes]))
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h, idx, norm=None):
        if norm == 'softmax':
            w = F.softmax(self.w)
        elif norm == 'act':
            w = F.relu(self.w)
        w = self.w.repeat(int(len(h) / len(self.w))).unsqueeze(dim=1)
        h = scatter(w * h, idx, dim=0)
        return self.lin(h)

class GAT(nn.Module):
    def __init__(self, n_layers, n_heads_list, in_dim, hid_dim, out_dim, dropout, neg_slope) -> None:
        super().__init__()
        if len(n_heads_list) == 1:
            n_heads_list = n_heads_list * n_layers
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(
                dglnn.GATConv(
                    in_dim, out_dim, n_heads_list[0], 
                    feat_drop=dropout, attn_drop=dropout, 
                    negative_slope=neg_slope
                )
            )
        else:
            self.layers.append(
                dglnn.GATConv(
                    in_dim, hid_dim, n_heads_list[0], 
                    feat_drop=dropout, attn_drop=dropout, 
                    negative_slope=neg_slope
                )
            )
            for i in range(1, n_layers-1):
                self.layers.append(
                    dglnn.GATConv(
                        hid_dim * n_heads_list[i-1], hid_dim, n_heads_list[i], 
                        feat_drop=dropout, attn_drop=dropout, 
                        negative_slope=neg_slope
                    )
                )
            self.layers.append(
                dglnn.GATConv(
                    hid_dim * n_heads_list[-2], out_dim, n_heads_list[-1], 
                    feat_drop=dropout, attn_drop=dropout, 
                    negative_slope=neg_slope
                )
            )
        self.act = F.relu # TODO: perhaps not the best. see how the original GAT is implemented
    
    def forward(self, g, h, attn=False):
        if attn:
            attn_list = []
        for layer in self.layers[:-1]:
            if not attn:
                h = self.act(layer(g, h, attn)).flatten(1)
            else:
                h, attn_emb =  self.act(layer(g, h, attn))
                h = h.flatten(1)
                attn_list.append(attn_emb)
        if not attn:
            h = self.layers[-1](g, h, attn).mean(dim=1)
            return h
        else:
            h, attn_emb =  self.act(layer(g, h, attn))
            h = h.mean(dim=1)
            attn_list.append(attn_emb)
            return h, attn_list
        