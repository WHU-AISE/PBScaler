import torch.nn.functional as F
import torch
import torch_geometric.transforms as T

from RL.common.MPNN import MPNN
from torch.nn import BatchNorm1d, Linear, init

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GAT, self).__init__()

#         self.conv1 = GATConv(in_channels, 8, heads=8)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(8 * 8, 256, heads=1, concat=False)
#         # Multilayer Perceptron
#         self.mlp = Seq(Linear(256, 512),
#                        ReLU(),
#                        Linear(512, out_channels))

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x =  F.relu(self.conv2(x, edge_index))
#         x = self.mlp(x)
#         return F.log_softmax(x, dim=-1)

HIDDEN_LAYER = 100

class GraphPolicyNet(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.mpnn = MPNN(in_channels, HIDDEN_LAYER)

        self.fc1 = Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc1.weight.data = init.normal_(self.fc1.weight.data, std=0.1)
        self.bn1 = BatchNorm1d(HIDDEN_LAYER)

        self.out = Linear(HIDDEN_LAYER, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.mpnn(x, edge_index))
        x = F.relu(self.fc1(x))
        x_norm = self.bn1(x)
        x = self.out(x_norm)
        return F.softmax(x,dim=1)