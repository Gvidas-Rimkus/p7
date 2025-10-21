import os

from graph_builders.base_grape import get_base_grape_graph
from models.base_grape import GRAPELayer

data = get_base_grape_graph(features_path="data/features.csv", targets_path="data/targets.csv")
####
import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
num_front = 100
num_back = 100
num_nodes = data.num_nodes
front_idx = torch.arange(num_front)
back_idx = torch.arange(num_nodes - num_back, num_nodes)
subset_idx = torch.cat([front_idx, back_idx])
edge_index, edge_attr = subgraph(subset_idx, data.edge_index, data.edge_attr, relabel_nodes=True)
x = data.x[subset_idx]
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
####

model = GRAPELayer(node_dim=data.x.shape[1], edge_dim=data.edge_attr.shape[1])
model.forward(h=data.x, e=data.edge_attr, edge_index=data.edge_index)