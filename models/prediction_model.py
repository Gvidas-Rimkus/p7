import torch
import torch.nn as nn
import numpy as np

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes=(64,)):
        super(MLPNet, self).__init__()
        layers = nn.ModuleList()
        for layer_size in hidden_layer_sizes:
            hidden_dim = layer_size
            layer = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            layers.append(layer)
            input_dim = hidden_dim
        layer = nn.Sequential(nn.Linear(input_dim, output_dim))
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs,-1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var 