import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prediction_model import MLPNet

class GRAPEModel(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers, dropout):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([GRAPELayer(node_dim, edge_dim) for _ in range(num_layers)])
        self.edge_head = MLPNet(input_dim=2*node_dim, output_dim=1, hidden_layer_sizes=(64,))
        self.node_head = MLPNet(input_dim=node_dim, output_dim=1, hidden_layer_sizes=(64,))

    def drop_edges(self, edge_index, edge_attr, drop_rate):
        if drop_rate == 0.0: 
            return edge_index, edge_attr
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) >= drop_rate
        return edge_index[:, mask], edge_attr[mask]

    def forward(self, h, e, edge_index, edge_dropout_rate=0.0):
        if self.training and edge_dropout_rate > 0.0:
            edge_index, e = self.drop_edges(edge_index, e, edge_dropout_rate)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        src, dst = edge_index #TODO det her skal jo være missing edge_index -> skal få graph_builder til at returnere det også
        edge_input = torch.cat([h[src], h[dst]], dim=-1)
        edge_pred = self.edge_head(edge_input).squeeze(-1)
        node_pred = self.node_head(h).squeeze(-1)

        return h, edge_pred, node_pred
    
#Se 3.3 i GRAPE paperet hvis nedenstående er svær at forstå
class GRAPELayer(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.P = nn.Linear(node_dim + edge_dim, node_dim)
        self.Q = nn.Linear(2 * node_dim, node_dim)
        self.W = nn.Linear(2 * node_dim + edge_dim, edge_dim)

    def forward(self, h, e, edge_index):
        source, target = edge_index
        h_u = h[source]
        h_v = h[target]

        message_input = torch.cat([h_u, e], dim=-1)
        messages = F.relu(self.P(message_input))

        #den her mean aggregering kan vidst gøres noget mere effektiv med torch_scatter, men jeg har problemer med at hente det og magter ikke at bøvle lige nu :)
        aggregated = torch.zeros_like(h)
        aggregated = aggregated.index_add(0, target, messages)
        deg = torch.bincount(target, minlength=h.size(0)).unsqueeze(1)
        aggregated = aggregated / deg

        node_input = torch.cat([h, aggregated], dim=-1)
        h_new = F.relu(self.Q(node_input))

        edge_input = torch.cat([e, h_u, h_v], dim=-1)
        e_new = F.relu(self.W(edge_input))
        return h_new, e_new