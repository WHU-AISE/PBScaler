

import numpy as np
import torch

from RL.common.MPNN import MPNN
from RL.common.GAT import GATCell, GCNCell

from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Sigmoid

HIDDEN_LAYER = 128

class STATE_MODEL(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.mpnn = GATCell(in_channels, HIDDEN_LAYER)

        self.cpu_predictor = Linear(HIDDEN_LAYER, 1)
        init.normal_(self.cpu_predictor.weight, std=0.01)

        self.mem_predictor = Linear(HIDDEN_LAYER, 1)
        init.normal_(self.mem_predictor.weight, std=0.01)

        self.p90_predictor = Linear(HIDDEN_LAYER, 1)
        init.normal_(self.p90_predictor.weight, std=0.01)

        self.Sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_embedding = F.relu(self.mpnn(x, edge_index))
        cpu = self.Sigmoid(self.cpu_predictor(node_embedding))
        mem = self.Sigmoid(self.mem_predictor(node_embedding))
        p90 = self.Sigmoid(self.p90_predictor(node_embedding))
        return cpu, mem, p90