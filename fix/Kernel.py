import torch
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from typing import List, Optional, Union

EPS = 1e-9

class GNTKernelRegression:
    """
    Graph Neural Tangent Kernel Regression with integrated GNTK computation
    """
    
    def __init__(self, 
                 num_layers: int, 
                 num_mlp_layers: int,
                 jk: bool = False,
                 scale: str = 'degree',
                 logistic: bool = False, 
                 device: str = 'cpu'):
        """
        Args:
            num_layers: Number of GNN layers
            num_mlp_layers: Number of MLP layers per GNN layer
            jk: Whether to use jumping knowledge (sum over all layers)
            scale: Scaling method for adjacency ('uniform' or 'degree')
            logistic: Whether to use logistic regression
            device: Computation device
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        self.logistic = logistic
        self.device = device
        self.kernel = None
        self.reg = None
        
        assert scale in ['uniform', 'degree'], "Scale must be 'uniform' or 'degree'"
        
    def _next_diag(self, S: torch.Tensor) -> tuple:
        """Process diagonal elements through a normal layer"""
        diag = torch.sqrt(torch.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = torch.clamp(S, -1, 1)
        
        # Compute derivative of ReLU activation
        DS = (torch.pi - torch.arccos(S)) / torch.pi
        S = (S * (torch.pi - torch.arccos(S)) + torch.sqrt(1 - S**2)) / torch.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag
    
    def _adj_diag(self, S: torch.Tensor, adj_block: torch.Tensor, N: int, scale_mat: torch.Tensor) -> torch.Tensor:
        """Process diagonal elements through adjacency layer"""
        return (adj_block @ S.reshape(-1)).reshape(N, N) * scale_mat
    
    def _next(self, S: torch.Tensor, diag1: torch.Tensor, diag2: torch.Tensor) -> tuple:
        """Process all elements through a normal layer"""
        S = S / diag1[:, None] / diag2[None, :]
        S = torch.clamp(S, -1, 1)
        DS = (torch.pi - torch.arccos(S)) / torch.pi
        S = (S * (torch.pi - torch.arccos(S)) + torch.sqrt(1 - S**2)) / torch.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def _adj(self, S: torch.Tensor, adj_block: torch.Tensor, N1: int, N2: int, scale_mat: torch.Tensor) -> torch.Tensor:
        """Process all elements through adjacency layer"""
        return (adj_block @ S.reshape(-1)).reshape(N1, N2) * scale_mat
    
    def compute_diagonal(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute diagonal elements of GNTK for a single graph
        Args:
            features: Node features [N x F]
            adj_matrix: Adjacency matrix [N x N]
        Returns:
            List of diagonal elements for each layer
        """
        N = adj_matrix.shape[0]
        if self.scale == 'uniform':
            scale_mat = torch.Tensor(1.0)
        else:
            scale_mat = 1.0 / (adj_matrix.sum(1) * adj_matrix.sum(0))
        
        adj_block = torch.kron(adj_matrix, adj_matrix)
        sigma = features @ features.T
        sigma = self._adj_diag(sigma, adj_block, N, scale_mat)
        diag_list = []
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, _, diag = self._next_diag(sigma)
                diag_list.append(diag)
            
            if layer != self.num_layers - 1:
                sigma = self._adj_diag(sigma, adj_block, N, scale_mat)
                
        return diag_list
    
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
        
        if self.scale == 'uniform':
            scale_mat = torch.tensor(1.0)
        else:
            scale_mat = 1.0 / (adj1.sum(1) * adj2.sum(0))
        
        adj_block = torch.kron(adj1, adj2)
        
        jump_ntk = 0
        sigma = features1 @ features2.T
        jump_ntk += sigma
        sigma = self._adj(sigma, adj_block, n1, n2, scale_mat)
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
                sigma = self._adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self._adj(ntk, adj_block, n1, n2, scale_mat)
        
        if self.jk:
            return jump_ntk.sum() * 2
        else:
            return ntk.sum() * 2
    
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
        
        # Compute kernel matrix
        n = len(train_features)
        kernel_matrix = torch.zeros((n, n), device=self.device)
        
        for i in range(n):
            for j in range(i, n):
                kernel_matrix[i,j] = self.compute_gntk(
                    train_features[i], train_features[j],
                    train_adjs[i], train_adjs[j],
                    diag_lists[i], diag_lists[j]
                )
                if i != j:
                    kernel_matrix[j,i] = kernel_matrix[i,j]
        
        # Fit regression model
        y_train = y_train.cpu().numpy()
        kernel_matrix_np = kernel_matrix.cpu().numpy()
        
        if self.logistic:
            self.reg = LogisticRegression(
                penalty=None,
                fit_intercept=False,
                max_iter=1000,
                multi_class='multinomial'
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
        
        for i in range(n_test):
            for j in range(n_train):
                kernel_test[i,j] = self.compute_gntk(
                    test_features[i], train_features[j],
                    test_adjs[i], train_adjs[j],
                    test_diags[i], train_diags[j]
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