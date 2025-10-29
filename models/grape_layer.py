import torch
import torch.nn as nn
import torch.nn.functional as F

#Se 3.3 i GRAPE paperet hvis nedenstående er svær at forstå
class GRAPELayer(nn.Module):
    def __init__(self, node_dim_in, edge_dim_in, node_dim_out, edge_dim_out):
        super().__init__()
        self.P = nn.Linear(node_dim_in + edge_dim_in, node_dim_out)
        self.Q = nn.Linear(node_dim_in + node_dim_out, node_dim_out)
        self.W = nn.Linear(2 * node_dim_in + edge_dim_in, edge_dim_out)

    def forward(self, h, e, edge_index):
        source, target = edge_index
        h_u = h[source]
        h_v = h[target]

        message_input = torch.cat([h_u, e], dim=-1)
        messages = F.relu(self.P(message_input))

        N = h.size(0)
        n_out = messages.size(-1)
        aggregated = h.new_zeros(N, n_out)
        aggregated.index_add_(0, target, messages)
        deg = torch.bincount(target, minlength=N).unsqueeze(1).clamp_min(1)
        aggregated = aggregated / deg

        node_input = torch.cat([h, aggregated], dim=-1)
        h_new = F.relu(self.Q(node_input))

        edge_input = torch.cat([e, h_u, h_v], dim=-1)
        e_new = F.relu(self.W(edge_input))
        return h_new, e_new