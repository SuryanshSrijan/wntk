import torch
from torch_geometric.data import Data

def LSIGF(weight: torch.nn.ParameterList, sparse_matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    
    k = len(weight)
    
    diffused_signals = [x]
    
    for i in range(1, k):
        x = torch.spmm(sparse_matrix, x)
        diffused_signals.append(x)
    
    output = [z @ w / torch.sqrt(torch.tensor(x.shape[1])) for z, w in zip(diffused_signals, weight)]
    output = torch.stack(output)
    return torch.sum(output, dim=0)

class GraphFilter(torch.nn.Module):
    
    def __init__(self, fan_in: int, fan_out: int, k: int, normalize: bool = True, device: str = 'cpu'):
        
        super(GraphFilter, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k = k
        self.normalize = normalize
        self.device = device
        
        self.weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(self.fan_in, self.fan_out, device=self.device)) for _ in range(self.k)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        
        if edge_weight is None:
            edge_weight = torch.ones(num_edges, device=self.device)
        
        sparse_matrix = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
        
        if self.normalize:
            
            s = torch.linalg.svdvals(sparse_matrix.to_dense().cpu())
            sparse_matrix = sparse_matrix / s[0]
        
        return LSIGF(self.weight, sparse_matrix, x)

class GNN(torch.nn.Module):
    
    def __init__(self, name: str, gnn_type: str, architecture: list[list[int]], softmax: bool, device: str):
        
        super(GNN, self).__init__()
        self.name = name
        self.gnn_type = gnn_type
        
        assert gnn_type in ['GNN'], f"Unsupported GNN type: {gnn_type}"
        
        self.feature_list = architecture[0]
        self.mlp_list = architecture[1]
        self.K_list = architecture[2]
        self.softmax = softmax
        self.device = device
        
        self.num_layers = len(self.feature_list) - 1
        self.num_mlp_layers = len(self.mlp_list) - 1
        
        # if self.K_list is not None: 
        assert self.num_layers == len(self.K_list), "Mismatch between number of layers and K_list length"
        
        self.layers = torch.nn.ModuleList()
        self.mlp_layers = torch.nn.ModuleList()
        self._initialize_layers()
    
    def _initialize_layers(self):
        for i in range(self.num_layers):
            if self.gnn_type == 'GNN':
                assert self.K_list is not None, "K_list must be provided for GNN type"
                self.layers.append(GraphFilter(self.feature_list[i], self.feature_list[i+1], self.K_list[i], device=self.device))
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        
        for i in range(self.num_mlp_layers):
            self.mlp_layers.append(torch.nn.Linear(self.mlp_list[i], self.mlp_list[i+1], bias=False, device=self.device))
            # torch.nn.init.xavier_uniform_(self.mlp_layers[i].weight, gain=1.0)
    
    def forward(self, data: Data, return_intermediate: bool = False):
        
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        assert isinstance(x, torch.Tensor)
        
        intermediate_outputs = [x]
        
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index=edge_index, edge_weight=edge_weight)
            x = torch.nn.functional.relu(x)
            
            if return_intermediate:
                intermediate_outputs.append(x)
        
        for i in range(self.num_mlp_layers):
            x = self.mlp_layers[i](x) / torch.sqrt(torch.tensor(self.mlp_list[i]))
            x = torch.nn.functional.relu(x)
            
            if return_intermediate:
                intermediate_outputs.append(x)
        
        if self.softmax:
            x = torch.nn.functional.log_softmax(x, dim=1)
        
        if return_intermediate:
            return intermediate_outputs
        
        return x
    
    def get_weights(self):
        
        weight_list = []
        
        fweights = torch.empty([self.feature_list[0], self.feature_list[1], self.K_list[0]], device=self.device)
        
        for i in range(self.num_layers):
            if self.gnn_type == 'GNN':
                fweights[:, :, i] = self.layers[0].weight[i].clone()
        
        weight_list.append(fweights)
        
        mlp_weights = torch.empty([self.mlp_list[0], self.mlp_list[1], self.num_mlp_layers], device=self.device)
        
        for i in range(self.num_mlp_layers):
            mlp_weights[:, :, i] = torch.transpose(self.mlp_layers[i].weight, 0, 1)
        
        weight_list.append(mlp_weights)
        return weight_list