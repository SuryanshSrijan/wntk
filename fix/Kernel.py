import torch
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from typing import List, Optional, Union
from torch_geometric.data import Data

EPS = 1e-8

class GNTKernelRegression:
    """
    Graph Neural Tangent Kernel Regression with integrated GNTK computation
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_mlp_layers: int,
                 jk: bool = False,
                 logistic: bool = False, 
                 device: str = 'cpu'):
        """
        Args:
            num_layers: Number of GNN layers
            num_mlp_layers: Number of MLP layers per GNN layer
            jk: Whether to use jumping knowledge (sum over all layers)
            logistic: Whether to use logistic regression
            device: Computation device
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.logistic = logistic
        self.device = device
        self.kernel = None
        self.reg = None
        
    def _next_diag(self, S: torch.Tensor) -> tuple:
        diag = torch.sqrt(torch.diag(S))
        S = S / (EPS + diag[:, None]) / (EPS + diag[None, :])
        S = torch.clamp(S, -1, 1)
        
        # Compute derivative of ReLU activation
        DS = (torch.pi - torch.arccos(S)) / torch.pi
        S = (S * (torch.pi - torch.arccos(S)) + torch.sqrt(1 - S**2)) / torch.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def _adj(self, S: torch.Tensor, adj_block: torch.Tensor, N1: int, N2: int | None = None) -> torch.Tensor:
        
        if N2 is None: N2 = N1
        
        # Convert to sparse
        if adj_block.layout == torch.strided:
            adj_block = adj_block.to_sparse()
        
        S_flat = S.reshape(-1, 1)
        result = torch.sparse.mm(adj_block, S_flat)
        
        return result.reshape(N1, N2)

    def _sparse_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        
        if A.layout == torch.strided:
            A = A.to_sparse()
        if B.layout == torch.strided:
            B = B.to_sparse()
        
        idxA = A.indices()
        valA = A.values()
        idxB = B.indices()
        valB = B.values()
        
        n, m = A.shape[0], B.shape[0]
        rows = (idxA[0] * m).view(-1, 1) + idxB[0].view(1, -1)
        cols = (idxA[1] * m).view(-1, 1) + idxB[1].view(1, -1)
        
        indices = torch.stack([
            rows.reshape(-1),
            cols.reshape(-1)
        ])
        
        values = (valA.view(-1, 1) * valB.view(1, -1)).reshape(-1)
        
        return torch.sparse_coo_tensor(
            indices,
            values,
            (n*m, n*m),
            device=self.device
        ).coalesce()

    def compute_diagonal(self, features: torch.Tensor, adj: torch.Tensor) -> List[torch.Tensor]:
        N = adj.size(0)
        
        # Convert to sparse if needed
        if adj.layout == torch.strided:
            adj = adj.to_sparse()
        
        adj_block = self._sparse_kron(adj, adj)
        sigma = torch.matmul(features, features.T)
        
        sigma = self._adj(sigma, adj_block, N)
        
        ntk = sigma.clone()
        diag_list = []
        
        for layer in range(1, self.num_layers):
            for _ in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self._next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            
            if layer != self.num_layers - 1:
                sigma = self._adj(sigma, adj_block, N)
                ntk = self._adj(ntk, adj_block, N)
        
        return diag_list

    def _next(self, S: torch.Tensor, diag1: torch.Tensor, diag2: torch.Tensor) -> tuple:
        
        S = S / (EPS + diag1[:, None]) / (EPS + diag2[None, :])
        S = torch.clamp(S, -1, 1)
        DS = (torch.pi - torch.arccos(S)) / torch.pi
        S = (S * (torch.pi - torch.arccos(S)) + torch.sqrt(1 - S**2)) / torch.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def compute_gntk(self, 
                   features1: torch.Tensor, 
                   features2: torch.Tensor,
                   adj1: torch.Tensor,
                   adj2: torch.Tensor,
                   diag_list1: List[torch.Tensor],
                   diag_list2: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute GNTK between two graphs
        Args:
            features1/2: Node features [N x F]
            adj1/2: Adjacency matrices [N x N]
            diag_list1/2: Diagonal elements from compute_diagonal()
        Returns:
            GNTK value
        """
        n1, n2 = adj1.shape[0], adj2.shape[0]
        
        adj_block = self._sparse_kron(adj1, adj2)
        
        jump_ntk = 0
        sigma = features1 @ features2.T
        jump_ntk += sigma
        sigma = self._adj(sigma, adj_block, n1, n2)
        ntk = sigma.clone()
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self._next(
                    sigma,
                    diag_list1[(layer-1)*self.num_mlp_layers + mlp_layer],
                    diag_list2[(layer-1)*self.num_mlp_layers + mlp_layer]
                    )
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            if layer != self.num_layers - 1:
                sigma = self._adj(sigma, adj_block, n1, n2)
                ntk = self._adj(ntk, adj_block, n1, n2)
        
        if self.jk:
            return jump_ntk
        else:
            return ntk
    
    def fit(self, 
            train_features: List[torch.Tensor], 
            train_adjs: List[torch.Tensor],
            y_train: torch.Tensor) -> Union[LogisticRegression, Ridge]:
        """
        Fit kernel regression model
        Args:
            train_features: List of node feature matrices
            train_adjs: List of adjacency matrices
            y_train: Target values
        """
        # Precompute diagonals for all training graphs
        diag_lists = [self.compute_diagonal(f, a) for f, a in zip(train_features, train_adjs)]

        kernel_matrix = self.compute_gntk(
            train_features[0], train_features[0],
            train_adjs[0], train_adjs[0],
            diag_lists[0], diag_lists[0]
        )
        
        # Fit regression model
        y_train = y_train.cpu().numpy()
        kernel_matrix_np = kernel_matrix.cpu().numpy()
        
        if self.logistic:
            self.reg = LogisticRegression(
                penalty=None,
                fit_intercept=False,
                max_iter=1000,
            ).fit(kernel_matrix_np, y_train)
        else:
            self.reg = Ridge(
                alpha=0.0,
                fit_intercept=False
            ).fit(kernel_matrix_np, y_train)
        
        self.kernel = kernel_matrix
        return self.reg
    
    def predict(self, 
               train_features: List[torch.Tensor],
               train_adjs: List[torch.Tensor],
               y_train: torch.Tensor,
               test_features: List[torch.Tensor],
               test_adjs: List[torch.Tensor]) -> torch.Tensor:
        """
        Make predictions
        Args:
            train_features/test_features: Node features
            train_adjs/test_adjs: Adjacency matrices
            y_train: Training targets
        Returns:
            Predictions
        """
        if self.reg is None:
            self.fit(train_features, train_adjs, y_train)
        
        # Precompute diagonals
        train_diags = [self.compute_diagonal(f, a) for f, a in zip(train_features, train_adjs)]
        test_diags = [self.compute_diagonal(f, a) for f, a in zip(test_features, test_adjs)]
        
        # Compute test-train kernel matrix
        n_train = len(train_features)
        n_test = len(test_features)
        kernel_test = torch.zeros((n_test, n_train), device=self.device)
        
        kernel_test = self.compute_gntk(
            test_features[0], train_features[0],
            test_adjs[0], train_adjs[0],
            test_diags[0], train_diags[0]
        )
        
        # Make predictions
        if self.logistic:
            assert isinstance(self.reg, LogisticRegression)
            preds = self.reg.predict_proba(kernel_test.cpu().numpy())
        else:
            assert isinstance(self.reg, Ridge)
            preds = self.reg.predict(kernel_test.cpu().numpy())
        
        return torch.from_numpy(preds).to(self.device)
    
    def get_eigenvalues(self, k: int = 6) -> torch.Tensor:
        """Compute top eigenvalues of the kernel matrix"""
        if self.kernel is None:
            raise ValueError("Must call fit() first")
        
        eigvals = torch.linalg.eigvalsh(self.kernel)
        return eigvals[-k:].flip(0)