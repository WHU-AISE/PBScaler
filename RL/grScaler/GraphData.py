from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class GraphData:
    def __init__(self, states, edge_index):
        self.graphs = []
        for state in states:
            self.graphs.append(self.__build_graph_data(state, edge_index))

    def __build_graph_data(self, state, edge_index):
        return Data(x=state, edge_index=edge_index)

    def get_batch_data(self, batch_size):
        data_loader = DataLoader(dataset=self.graphs, batch_size=batch_size,shuffle=True)
        batch_data = None
        for _, data in enumerate(data_loader):
            batch_data = data
            break
        return batch_data
