import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, num_layers, backbone='gcn', normalize='none',
                 dropout=0.0):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize

        self.activation = activation()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * num_layers

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == 'sage':
                self.layers.append(SAGEConv(in_dim, out_dim, aggr='mean', normalize=True, root_weight=True))
            elif backbone == 'gat':
                self.layers.append(GATConv(in_dim, out_dim, heads=1))
            elif backbone == 'gcn':
                self.layers.append(GCNConv(in_dim, out_dim, ))
            elif backbone == 'gin':
                self.layers.append(GINConv(nn.Linear(in_dim, out_dim)))
            elif backbone == 'mlp':
                self.layers.append(nn.Linear(in_dim, out_dim))
            self.norms.append(nn.BatchNorm1d(out_dim))

        self.lin = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return self.predict(z)

    def encode(self, x, edge_index, edge_attr=None):
        z = x

        for i, conv in enumerate(self.layers):
            if self.backbone != 'mlp':
                z = conv(z, edge_index, edge_attr)
            else:
                z = conv(z)
            z = self.activation(z)
            if self.normalize != 'none':
                z = self.norms[i](z)

            z = self.dropout(z)

        return z

    def predict(self, x):
        return self.lin(x)

    def reset_lin(self, num_classes):
        device = next(self.lin.parameters()).device

        self.lin = nn.Linear(self.hidden_dim, num_classes).to(device)
        self.lin.reset_parameters()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_params(self):
        for param in self.parameters():
            param.requires_grad = True
