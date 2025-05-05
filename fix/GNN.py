import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

class ScaledMLP(nn.Module):
    """MLP with NTK-style scaling and ReLU activations."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim, bias=False, device=device))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False, device=device))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False, device=device))
            self.layers.append(nn.Linear(hidden_dim, output_dim, bias=False, device=device))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            fan_in = layer.in_features
            x = layer(x) / math.sqrt(fan_in)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

class GNN(nn.Module):
    """GNN modified to reflect the GNTK structure."""
    def __init__(self, name: str, gnn_type: str, input_dim: int, hidden_dim: int, 
                 output_dim: int, num_layers: int, num_mlp_layers: int, 
                 softmax: bool, device: str):
        super().__init__()
        self.name = name
        self.gnn_type = gnn_type
        assert gnn_type == 'GNN', f"Unsupported GNN type: {gnn_type}"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.softmax = softmax
        self.device = device
        self.mlps = nn.ModuleList()
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize MLPs for each layer based on GNTK structure."""
        for l in range(self.num_layers):
            input_dim = self.input_dim if l == 0 else self.hidden_dim
            output_dim = self.output_dim if l == self.num_layers - 1 else self.hidden_dim
            self.mlps.append(ScaledMLP(input_dim, self.hidden_dim, output_dim, 
                                     self.num_mlp_layers, self.device))

    def forward(self, data: Data, return_intermediate: bool = False):
        """Forward pass mimicking GNTK: initial aggregation, then MLP + aggregation per layer."""
        
        intermediate_outputs = []
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        num_nodes = x.shape[0]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=self.device)

        # Build sparse adjacency matrix
        A = torch.sparse_coo_tensor(edge_index, edge_weight, 
                                  (num_nodes, num_nodes), device=self.device)

        # Initial aggregation
        h = torch.spmm(A, x)
        if return_intermediate: intermediate_outputs = [h] 

        # Layer-wise computation
        for l in range(self.num_layers):
            h = self.mlps[l](h)  # Apply MLP with ReLU
            if l < self.num_layers - 1:
                h = torch.spmm(A, h)  # Aggregate except for the last layer
            if return_intermediate:
                intermediate_outputs.append(h)

        if self.softmax:
            h = F.log_softmax(h, dim=1)

        return intermediate_outputs if return_intermediate else h