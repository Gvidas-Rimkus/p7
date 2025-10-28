import torch
import torch.nn as nn
import torch.nn.functional as F

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
        deg_safe = deg.clamp_min(1)#sørger for at ingen degrees kan være 0 efter edge dropout
        aggregated = aggregated / deg_safe

        node_input = torch.cat([h, aggregated], dim=-1)
        h_new = F.relu(self.Q(node_input))

        edge_input = torch.cat([e, h_u, h_v], dim=-1)
        e_new = F.relu(self.W(edge_input))
        return h_new, e_new