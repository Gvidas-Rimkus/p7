import torch
import torch.nn as nn

from models.grape_layer import GRAPELayer
from models.prediction_model import MLPNet

class GRAPEModel(nn.Module):
    def __init__(self, data, node_dims, edge_dims, edge_dropout_rate=0.3):
        super().__init__()
        self.node_dim = data.x.shape[1]
        self.edge_dim = data.edge_attr.shape[1]
        self.num_prediction_nodes = data.y.shape[0]
        self.num_prediction_classes = data.y.shape[1]
        self.edge_dropout_rate = edge_dropout_rate

        node_dims[0] = self.node_dim
        edge_dims[0] = self.edge_dim

        # self.layers = nn.ModuleList([GRAPELayer(self.node_dim, self.edge_dim) for _ in range(self.num_layers)])
        self.layers = nn.ModuleList([
            GRAPELayer(node_dims[i], edge_dims[i], node_dims[i+1], edge_dims[i+1])
            for i in range(len(node_dims)-1)
        ])
        self.edge_head = MLPNet(input_dim=2 * node_dims[-1], output_dim=1, hidden_layer_sizes=(64,))
        self.node_head = MLPNet(input_dim=node_dims[-1], output_dim=self.num_prediction_classes, hidden_layer_sizes=(64,))

    def drop_edges(self, edge_index, edge_attr):
        if self.edge_dropout_rate == 0.0: 
            return edge_index, edge_attr, None, None

        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_dropout_rate
        return (
            edge_index[:, mask],
            edge_attr[mask],
            edge_index[:, ~mask],
            edge_attr[~mask]
        )

    def forward(self, data):
        h = data.x
        y = data.y

        edge_index = data.edge_index[:, data.train_edge_mask]
        e = data.edge_attr[data.train_edge_mask]

        if self.training and self.edge_dropout_rate > 0.0:
            edge_index, e, dropped_edge_index, dropped_e = self.drop_edges(edge_index, e)
        else:
            dropped_edge_index = edge_index
            dropped_e = e

        # tilføjer message passing mellem sites, når grape_site_connectivity_graph trænes.
        if hasattr(data, 'site2site_edge_index'):
            edge_index = torch.cat([edge_index, data.site2site_edge_index], dim=1)
            e = torch.cat([e, data.site2site_edge_attr], dim=0)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        #Edge prediction for de droppede edges
        src, dst = dropped_edge_index
        edge_input = torch.cat([h[src], h[dst]], dim=-1)
        edge_pred = self.edge_head(edge_input).squeeze(-1)

        #Node prediction for observations noderne
        node_pred = self.node_head(h[data.train_mask, :]).squeeze(-1)

        return edge_pred, dropped_e.squeeze(-1), node_pred, y[data.train_mask, :]
    
    def evaluate(self, data, split):
        if split == "validation":
            eval_edge_index = data.edge_index[:, data.val_edge_mask]
            e_true = data.edge_attr[data.val_edge_mask]
            node_mask = data.val_mask
            y_true = data.y[data.val_mask, :]

        else:
            eval_edge_index = data.edge_index[:, data.test_edge_mask]
            e_true = data.edge_attr[data.test_edge_mask]
            node_mask = data.test_mask
            y_true = data.y[data.test_mask, :]
        
        #message passing altid kun på train_split så der ikke leakes
        edge_index = data.edge_index[:, data.train_edge_mask]
        e = data.edge_attr[data.train_edge_mask]

        h = data.x
        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        src, dst = eval_edge_index
        edge_input = torch.cat([h[src], h[dst]], dim=-1)
        edge_pred = self.edge_head(edge_input).squeeze(-1)
        node_pred = self.node_head(h[node_mask, :]).squeeze(-1)

        return edge_pred, e_true.squeeze(-1), node_pred, y_true