import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch

HIDDEN_LAYTER = 128

class GCNCell(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(GCNCell, self).__init__()
        
        self.gconv = GCNConv(in_channels=input_dim,
                             out_channels= output_dim,
                             bias=True,
                             improved = True)

    def forward(self, cur_state, edge_index):
        conv = self.gconv(cur_state, edge_index)
        return conv

class GATCell(nn.Module):

    def __init__(self, input_dim, output_dim, head=8):

        super(GATCell, self).__init__()
        
        self.gconv = GATConv(in_channels=input_dim,
                             out_channels=output_dim,
                             heads=head,
                             concat = False,
                             bias=True)

    def forward(self, cur_state, edge_index):
        conv = self.gconv(cur_state, edge_index)
        return conv