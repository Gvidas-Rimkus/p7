import torch
import torch.nn as nn

from models.grape_layer import GRAPELayer
from models.prediction_model import MLPNet

class GRAPEModel(nn.Module):
    def __init__(self, data, node_dims, edge_dims, node_head_dims, edge_head_dims, edge_dropout_rate=0.3, new=False):
        super().__init__()
        self.node_dim = data.x.shape[1]
        self.edge_dim = data.edge_attr.shape[1]
        self.num_prediction_nodes = data.y.shape[0]
        self.num_prediction_classes = data.y.shape[1]
        self.edge_dropout_rate = edge_dropout_rate
        self.new = new

        node_dims[0] = self.node_dim
        edge_dims[0] = self.edge_dim

        self.layers = nn.ModuleList([
            GRAPELayer(node_dims[i], edge_dims[i], node_dims[i+1], edge_dims[i+1])
            for i in range(len(node_dims)-1)
        ])
        self.node_head = MLPNet(input_dim=node_dims[-1], output_dim=self.num_prediction_classes, hidden_layer_sizes=node_head_dims)
        self.node_head_new = MLPNet(input_dim=data.x.shape[1], output_dim=self.num_prediction_classes, hidden_layer_sizes=node_head_dims)
        self.edge_head = MLPNet(input_dim=2 * node_dims[-1], output_dim=1, hidden_layer_sizes=edge_head_dims)

    def drop_edges(self, edge_index, edge_attr):
        if self.edge_dropout_rate == 0.0: 
            mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            return edge_index, edge_attr, None, None

        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_dropout_rate
        return (
            edge_index[:, mask],
            edge_attr[mask],
            edge_index[:, ~mask],
            edge_attr[~mask],
            mask
        )

    def reconstruct_dense_features(self, h, data):
        n_obs = data.x.shape[0] - data.x.shape[1]
        n_feat = data.x.shape[1]
        h_obs = h[:n_obs]
        h_feat = h[n_obs:]

        h_obs_exp = h_obs.repeat_interleave(n_feat, dim=0)
        h_feat_exp = h_feat.repeat(n_obs, 1)
        edge_inputs = torch.cat([h_obs_exp, h_feat_exp], dim=-1)
        edge_preds = self.edge_head(edge_inputs).squeeze(-1)

        edge_preds[data.edge2dense_idx] = data.edge_attr[data.edge_mask_obs_to_feat].squeeze(-1)

        hat_D = edge_preds.view(n_obs, n_feat)
        return hat_D

    def forward(self, data):
        h = data.x
        y = data.y

        edge_index = data.edge_index[:, data.train_edge_mask]
        e = data.edge_attr[data.train_edge_mask]
        edge_is_cont = data.edge_is_cont[data.train_edge_mask]

        if self.training and self.edge_dropout_rate > 0.0:
            edge_index, e, dropped_edge_index, dropped_e, keep_mask = self.drop_edges(edge_index, e)
            dropped_is_cont = edge_is_cont[~keep_mask]    
        else:
            dropped_edge_index = edge_index
            dropped_e = e
            dropped_is_cont = edge_is_cont

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
        if self.new:
            hat_D = self.reconstruct_dense_features(h, data)
            padding_size = y.shape[0] - hat_D.shape[0]
            padding = torch.zeros(padding_size, hat_D.shape[1], device=hat_D.device, dtype=hat_D.dtype)
            hat_D_padded = torch.cat([hat_D, padding], dim = 0)
            node_pred = self.node_head_new(hat_D_padded[data.train_mask, :]).squeeze(-1)
        else: 
            node_pred = self.node_head(h[data.train_mask, :]).squeeze(-1)

        return edge_pred, dropped_e.squeeze(-1), node_pred, y[data.train_mask, :], dropped_is_cont
    
    def evaluate(self, data, split):
        if split == "validation":
            eval_edge_index = data.edge_index[:, data.val_edge_mask]
            e_true = data.edge_attr[data.val_edge_mask]
            node_mask = data.val_mask
            y_true = data.y[data.val_mask, :]
            edge_is_cont = data.edge_is_cont[data.val_edge_mask]

        else:
            eval_edge_index = data.edge_index[:, data.test_edge_mask]
            e_true = data.edge_attr[data.test_edge_mask]
            node_mask = data.test_mask
            y_true = data.y[data.test_mask, :]
            edge_is_cont = data.edge_is_cont[data.test_mask]
        #message passing altid kun på train_split så der ikke leakes
        edge_index = data.edge_index[:, data.train_edge_mask]
        e = data.edge_attr[data.train_edge_mask]

        h = data.x
        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        src, dst = eval_edge_index
        edge_input = torch.cat([h[src], h[dst]], dim=-1)
        edge_pred = self.edge_head(edge_input).squeeze(-1)
        if self.new:
            hat_D = self.reconstruct_dense_features(h, data)
            padding_size = data.y.shape[0] - hat_D.shape[0]
            padding = torch.zeros(padding_size, hat_D.shape[1], device=hat_D.device, dtype=hat_D.dtype)
            hat_D_padded = torch.cat([hat_D, padding], dim = 0)
            node_pred = self.node_head_new(hat_D_padded[node_mask, :]).squeeze(-1)
        else: 
            node_pred = self.node_head(h[node_mask, :]).squeeze(-1)

        return edge_pred, e_true.squeeze(-1), node_pred, y_true, edge_is_cont