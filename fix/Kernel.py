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
        S = S / (EPS + diag[:, None]) / (EPS + diag[None, :])
        S = torch.clamp(S, -1, 1)
        
        # Compute derivative of ReLU activation
        DS = (torch.pi - torch.arccos(S)) / torch.pi
        S = (S * (torch.pi - torch.arccos(S)) + torch.sqrt(1 - S**2)) / torch.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag
    
    def _adj_diag(self, S: torch.Tensor, adj_block: torch.Tensor, N: int, scale_mat: torch.Tensor) -> torch.Tensor:
        """Sparse version of adjacency propagation for diagonal elements"""
        # Ensure S is 2D
        if S.dim() == 1:
            S = S.unsqueeze(1)
        
        # Convert to sparse if needed
        if adj_block.layout == torch.strided:
            adj_block = adj_block.to_sparse()
        
        # Flatten S properly
        S_flat = S.reshape(-1, 1)
        
        # Check dimensions
        if adj_block.shape[1] != S_flat.shape[0]:
            raise ValueError(f"Dimension mismatch: adj_block {adj_block.shape} vs S_flat {S_flat.shape}")
        
        # Perform sparse-dense matmul
        result = torch.sparse.mm(adj_block, S_flat)
        
        # Apply scaling if needed
        if isinstance(scale_mat, torch.Tensor):
            result = result * scale_mat.reshape(-1, 1)
        
        return result.reshape(N, N)

    def _adj(self, S: torch.Tensor, adj_block: torch.Tensor, N1: int, N2: int, scale_mat: torch.Tensor) -> torch.Tensor:
        """Sparse version of adjacency propagation for all elements"""
        # Convert to sparse if needed
        if adj_block.layout == torch.strided:
            adj_block = adj_block.to_sparse()
        
        # Reshape and perform sparse multiplication
        S_flat = S.reshape(-1, 1)
        result = torch.sparse.mm(adj_block, S_flat)
        
        # Apply scaling if needed
        if isinstance(scale_mat, torch.Tensor):
            result = result * scale_mat.reshape(-1, 1)
        
        return result.reshape(N1, N2)

    def _sparse_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Fixed sparse Kronecker product with dimension checks"""
        # Ensure sparse format
        if A.layout == torch.strided:
            A = A.to_sparse()
        if B.layout == torch.strided:
            B = B.to_sparse()
        
        # Get sparse components
        idxA = A.indices()
        valA = A.values()
        idxB = B.indices()
        valB = B.values()
        
        # Compute output dimensions
        n, m = A.shape[0], B.shape[0]
        out_size = n * m
        
        # Compute Kronecker indices
        rows = (idxA[0] * m).view(-1, 1) + idxB[0].view(1, -1)
        cols = (idxA[1] * m).view(-1, 1) + idxB[1].view(1, -1)
        
        # Flatten and stack indices
        indices = torch.stack([
            rows.reshape(-1),
            cols.reshape(-1)
        ])
        
        # Compute Kronecker values
        values = (valA.view(-1, 1) * valB.view(1, -1)).reshape(-1)
        
        # Create sparse tensor
        return torch.sparse_coo_tensor(
            indices,
            values,
            (out_size, out_size),
            device=self.device
        ).coalesce()

    def compute_diagonal(self, features: torch.Tensor, adj: torch.Tensor) -> List[torch.Tensor]:
        N = adj.size(0)
        
        # Convert to sparse if needed
        if adj.layout == torch.strided:
            adj = adj.to_sparse()
        
        # Compute scaling matrix
        scale_mat = torch.tensor(1.0, device=self.device)
        # if self.scale == 'uniform':
        #     scale_mat = torch.tensor(1.0)
        # else:
        #     degrees = torch.sparse.sum(adj, dim=1).to_dense()
        #     scale_mat = 1.0 / (degrees.unsqueeze(1) * degrees.unsqueeze(0))
        
        # Compute sparse Kronecker product with checks
        try:
            adj_block = self._sparse_kron(adj, adj)
        except Exception as e:
            print(f"Error in sparse_kron: {e}")
            print(f"Adj shape: {adj.shape}")
            raise
        
        # Initialize covariance
        sigma = torch.matmul(features, features.T)
        
        # Verify dimensions before propagation
        if sigma.numel() != N * N:
            raise ValueError(f"Sigma size {sigma.shape} doesn't match expected {N}x{N}")
        
        if adj_block.shape[1] != N * N:
            raise ValueError(f"adj_block cols {adj_block.shape[1]} != N² {N*N}")
        
        # First propagation
        try:
            sigma = self._adj_diag(sigma, adj_block, N, scale_mat)
        except Exception as e:
            print(f"Error in first _adj_diag: {e}")
            print(f"sigma shape: {sigma.shape}")
            print(f"adj_block shape: {adj_block.shape}")
            raise
        
        # Rest of the computation...
        ntk = sigma.clone()
        diag_list = []
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self._next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            
            if layer != self.num_layers - 1:
                sigma = self._adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self._adj_diag(ntk, adj_block, N, scale_mat)
        
        return diag_list

    def _next(self, S: torch.Tensor, diag1: torch.Tensor, diag2: torch.Tensor) -> tuple:
        """Process all elements through a normal layer"""
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
        
        scale_mat = torch.tensor(1.0, device=self.device)
        # if self.scale == 'uniform':
        #     scale_mat = torch.tensor(1.0)
        # else:
        #     scale_mat = 1.0 / (adj1.sum(1) * adj2.sum(0))
        
        adj_block = self._sparse_kron(adj1, adj2)
        
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
            return jump_ntk
        else:
            return ntk
    
    def fit(self, 
            train_features: List[List[torch.Tensor]], 
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
        diag_lists = [self.compute_diagonal(f[0], a) for f, a in zip(train_features, train_adjs)]
        
        # # Compute kernel matrix
        # n = len(train_features)
        # kernel_matrix = torch.zeros((n, n), device=self.device)
        
        # for i in range(n):
        #     for j in range(i, n):
        #         kernel_matrix[i,j] = self.compute_gntk(
        #             train_features[i][0], train_features[j][0],
        #             train_adjs[i], train_adjs[j],
        #             diag_lists[i], diag_lists[j]
        #         )
        #         if i != j:
        #             kernel_matrix[j,i] = kernel_matrix[i,j]
        kernel_matrix = self.compute_gntk(
            train_features[0][0], train_features[0][0],
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
               train_features: List[List[torch.Tensor]],
               train_adjs: List[torch.Tensor],
               y_train: torch.Tensor,
               test_features: List[List[torch.Tensor]],
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
        train_diags = [self.compute_diagonal(f[0], a) for f, a in zip(train_features, train_adjs)]
        test_diags = [self.compute_diagonal(f[0], a) for f, a in zip(test_features, test_adjs)]
        
        # Compute test-train kernel matrix
        n_train = len(train_features)
        n_test = len(test_features)
        kernel_test = torch.zeros((n_test, n_train), device=self.device)
        
        kernel_test = self.compute_gntk(
            test_features[0][0], train_features[0][0],
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