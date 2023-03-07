from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import add_self_loops
from torch import nn

HIDDEN_LAYER = 128

class MPNN(MessagePassing):
    def __init__(self,in_channels, out_channels):
        super().__init__(aggr='add')

        self.fc1 = Linear(in_channels, HIDDEN_LAYER)
        self.fc1.weight.data = nn.init.normal_(self.fc1.weight.data, std=0.1)
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER)

        self.fc2 =  Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc2.weight.data = nn.init.normal_(self.fc2.weight.data, std=0.1)
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER)

        self.fc3 = Linear(HIDDEN_LAYER, out_channels)
        self.fc3.weight.data = nn.init.normal_(self.fc3.weight.data, std=0.1)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        h1 = self.relu1(self.fc1(x))
        h1_norm = self.bn1(h1)
        h2 = self.relu2(self.fc2(h1_norm))
        h2_norm = self.bn2(h2)
        h3 = self.relu3(h2_norm)
        return self.propagate(edge_index, x=h3)