import torch
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from typing import List, Optional, Union
from torch_geometric.data import Data

# Constants
EPSILON = 1e-8  # Small constant for numerical stability

class GNTKernelRegression:
    """Graph Neural Tangent Kernel (GNTK) for regression and classification tasks.
    
    This class implements the GNTK computation with support for:
    - Multiple GNN and MLP layers
    - Jumping knowledge connections
    - Different adjacency scaling methods
    - Both regression and classification tasks
    """

    def __init__(self, 
                 num_layers: int, 
                 num_mlp_layers: int,
                 jk: bool = False,
                 scale: str = 'degree',
                 logistic: bool = False, 
                 device: str = 'cpu'):
        """Initialize GNTK model.
        
        Args:
            num_layers: Number of GNN layers in the architecture
            num_mlp_layers: Number of MLP layers per GNN layer
            jk: Whether to use jumping knowledge (sum over all layers)
            scale: Scaling method for adjacency ('uniform' or 'degree')
            logistic: Whether to perform classification (True) or regression (False)
            device: Computation device ('cpu' or 'cuda')
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        self.logistic = logistic
        self.device = device
        
        # Model state
        self.kernel_matrix = None
        self.regression_model = None
        
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.scale not in ['uniform', 'degree']:
            raise ValueError("Scale must be either 'uniform' or 'degree'")
        if self.num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        if self.num_mlp_layers < 1:
            raise ValueError("Number of MLP layers must be at least 1")

    def _process_diagonal_elements(self, covariance: torch.Tensor) -> tuple:
        """Process diagonal elements through a normal layer.
        
        Args:
            covariance: Input covariance matrix
            
        Returns:
            Tuple of (processed covariance, derivative, diagonal elements)
        """
        diag = torch.sqrt(torch.diag(covariance))
        normalized = covariance / (EPSILON + diag[:, None]) / (EPSILON + diag[None, :])
        clamped = torch.clamp(normalized, -1, 1)
        
        # Compute ReLU derivative
        derivative = (torch.pi - torch.arccos(clamped)) / torch.pi
        processed = (clamped * (torch.pi - torch.arccos(clamped)) + 
                    torch.sqrt(1 - clamped**2)) / torch.pi
        scaled = processed * diag[:, None] * diag[None, :]
        
        return scaled, derivative, diag

    def _process_all_elements(self, 
                            covariance: torch.Tensor, 
                            diag1: torch.Tensor,
                            diag2: torch.Tensor) -> tuple:
        """Process all matrix elements through a normal layer.
        
        Args:
            covariance: Input covariance matrix
            diag1: Diagonal elements from first graph
            diag2: Diagonal elements from second graph
            
        Returns:
            Tuple of (processed covariance, derivative)
        """
        normalized = covariance / (EPSILON + diag1[:, None]) / (EPSILON + diag2[None, :])
        clamped = torch.clamp(normalized, -1, 1)
        
        derivative = (torch.pi - torch.arccos(clamped)) / torch.pi
        processed = (clamped * (torch.pi - torch.arccos(clamped)) + 
                   torch.sqrt(1 - clamped**2)) / torch.pi
        scaled = processed * diag1[:, None] * diag2[None, :]
        
        return scaled, derivative

    def _sparse_kronecker_product(self, 
                                matrix1: torch.Tensor, 
                                matrix2: torch.Tensor) -> torch.Tensor:
        """Compute sparse Kronecker product between two matrices.
        
        Args:
            matrix1: First sparse matrix
            matrix2: Second sparse matrix
            
        Returns:
            Kronecker product as sparse matrix
        """
        # Convert to sparse if needed
        matrix1 = matrix1.to_sparse() if matrix1.layout == torch.strided else matrix1
        matrix2 = matrix2.to_sparse() if matrix2.layout == torch.strided else matrix2
        
        # Get sparse components
        idx1, val1 = matrix1.indices(), matrix1.values()
        idx2, val2 = matrix2.indices(), matrix2.values()
        
        # Compute output dimensions
        n, m = matrix1.shape[0], matrix2.shape[0]
        output_size = n * m
        
        # Compute Kronecker indices and values
        rows = (idx1[0] * m).view(-1, 1) + idx2[0].view(1, -1)
        cols = (idx1[1] * m).view(-1, 1) + idx2[1].view(1, -1)
        values = (val1.view(-1, 1) * val2.view(1, -1)).reshape(-1)
        
        # Create and coalesce sparse tensor
        return torch.sparse_coo_tensor(
            torch.stack([rows.reshape(-1), cols.reshape(-1)]),
            values,
            (output_size, output_size),
            device=self.device
        ).coalesce()

    def _propagate_adjacency(self, 
                           matrix: torch.Tensor,
                           adjacency: torch.Tensor,
                           scale_factor: torch.Tensor = None) -> torch.Tensor:
        """Perform adjacency propagation with optional scaling.
        
        Args:
            matrix: Input matrix to propagate
            adjacency: Sparse adjacency matrix
            scale_factor: Optional scaling matrix
            
        Returns:
            Propagated matrix
        """
        # Ensure sparse format
        adjacency = adjacency.to_sparse() if adjacency.layout == torch.strided else adjacency
        
        # Flatten and propagate
        flat_matrix = matrix.reshape(-1, 1)
        result = torch.sparse.mm(adjacency, flat_matrix)
        
        # Apply scaling if provided
        if scale_factor is not None:
            result = result * scale_factor.reshape(-1, 1)
            
        return result.reshape(matrix.shape)

    def compute_diagonal_components(self, 
                                  features: torch.Tensor, 
                                  adjacency: torch.Tensor) -> List[torch.Tensor]:
        """Compute diagonal components needed for GNTK computation.
        
        Args:
            features: Node feature matrix [N x F]
            adjacency: Graph adjacency matrix [N x N]
            
        Returns:
            List of diagonal components for each layer
        """
        num_nodes = adjacency.size(0)
        adjacency = adjacency.to(self.device)
        
        # Compute initial covariance
        covariance = torch.matmul(features, features.T)
        
        # Create adjacency block matrix
        adjacency_block = self._sparse_kronecker_product(adjacency, adjacency)
        
        # First propagation
        covariance = self._propagate_adjacency(covariance, adjacency_block)
        
        # Initialize storage
        diagonal_components = []
        ntk = covariance.clone()
        
        # Process through layers
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                covariance, derivative, diag = self._process_diagonal_elements(covariance)
                diagonal_components.append(diag)
                ntk = ntk * derivative + covariance
            
            if layer != self.num_layers - 1:
                covariance = self._propagate_adjacency(covariance, adjacency_block)
                ntk = self._propagate_adjacency(ntk, adjacency_block)
        
        return diagonal_components

    def compute_kernel(self,
                      features1: torch.Tensor,
                      features2: torch.Tensor,
                      adjacency1: torch.Tensor,
                      adjacency2: torch.Tensor,
                      diag1: List[torch.Tensor],
                      diag2: List[torch.Tensor]) -> torch.Tensor:
        """Compute the GNTK between two graphs.
        
        Args:
            features1/2: Node feature matrices [N x F]
            adjacency1/2: Adjacency matrices [N x N]
            diag1/2: Diagonal components from compute_diagonal_components()
            
        Returns:
            Computed kernel matrix
        """
        num_nodes1, num_nodes2 = adjacency1.shape[0], adjacency2.shape[0]
        
        # Create adjacency block matrix
        adjacency_block = self._sparse_kronecker_product(adjacency1, adjacency2)
        
        # Initialize kernel computation
        covariance = features1 @ features2.T
        kernel_value = covariance.clone()
        if self.jk:
            accumulated_kernel = covariance.clone()
        
        # First propagation
        covariance = self._propagate_adjacency(covariance, adjacency_block)
        kernel_value = covariance.clone()
        
        # Process through layers
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                idx = (layer-1)*self.num_mlp_layers + mlp_layer
                covariance, derivative = self._process_all_elements(
                    covariance, diag1[idx], diag2[idx]
                )
                kernel_value = kernel_value * derivative + covariance
            
            if self.jk:
                accumulated_kernel += kernel_value
            
            if layer != self.num_layers - 1:
                covariance = self._propagate_adjacency(covariance, adjacency_block)
                kernel_value = self._propagate_adjacency(kernel_value, adjacency_block)
        
        return accumulated_kernel if self.jk else kernel_value

    def fit(self, 
            train_features: List[List[torch.Tensor]], 
            train_adjacency: List[torch.Tensor],
            train_labels: torch.Tensor) -> Union[LogisticRegression, Ridge]:
        """Fit the kernel regression model.
        
        Args:
            train_features: List of node feature matrices
            train_adjacency: List of adjacency matrices
            train_labels: Target labels
            
        Returns:
            Fitted regression model
        """
        # Precompute diagonal components for all training graphs
        diagonal_components = [
            self.compute_diagonal_components(f[0], a) 
            for f, a in zip(train_features, train_adjacency)
        ]
        
        # Compute kernel matrix (simplified for single graph case)
        self.kernel_matrix = self.compute_kernel(
            train_features[0][0], train_features[0][0],
            train_adjacency[0], train_adjacency[0],
            diagonal_components[0], diagonal_components[0]
        )
        
        # Prepare data for sklearn
        kernel_np = self.kernel_matrix.cpu().numpy()
        labels_np = train_labels.cpu().numpy()
        
        # Fit appropriate model
        if self.logistic:
            self.regression_model = LogisticRegression(
                penalty=None,
                fit_intercept=False,
                max_iter=1000,
            ).fit(kernel_np, labels_np)
        else:
            self.regression_model = Ridge(
                alpha=0.0,
                fit_intercept=False
            ).fit(kernel_np, labels_np)
        
        return self.regression_model

    def predict(self, 
                train_features: List[List[torch.Tensor]],
                train_adjacency: List[torch.Tensor],
                train_labels: torch.Tensor,
                test_features: List[List[torch.Tensor]],
                test_adjacency: List[torch.Tensor]) -> torch.Tensor:
        """Make predictions using the trained model.
        
        Args:
            train_features/test_features: Node features
            train_adjacency/test_adjacency: Adjacency matrices
            train_labels: Training targets
            
        Returns:
            Model predictions
        """
        if self.regression_model is None:
            self.fit(train_features, train_adjacency, train_labels)
        
        # Precompute diagonal components
        train_diag = [self.compute_diagonal_components(f[0], a) 
                     for f, a in zip(train_features, train_adjacency)]
        test_diag = [self.compute_diagonal_components(f[0], a) 
                    for f, a in zip(test_features, test_adjacency)]
        
        # Compute test-train kernel matrix
        kernel_test = self.compute_kernel(
            test_features[0][0], train_features[0][0],
            test_adjacency[0], train_adjacency[0],
            test_diag[0], train_diag[0]
        )
        
        # Make predictions
        if self.logistic:
            predictions = self.regression_model.predict_proba(kernel_test.cpu().numpy())
        else:
            predictions = self.regression_model.predict(kernel_test.cpu().numpy())
        
        return torch.from_numpy(predictions).to(self.device)

    def get_top_eigenvalues(self, num_eigenvalues: int = 6) -> torch.Tensor:
        """Compute the top eigenvalues of the kernel matrix.
        
        Args:
            num_eigenvalues: Number of top eigenvalues to return
            
        Returns:
            Tensor of top eigenvalues in descending order
        """
        if self.kernel_matrix is None:
            raise ValueError("Must call fit() before computing eigenvalues")
            
        eigenvalues = torch.linalg.eigvalsh(self.kernel_matrix)
        return eigenvalues[-num_eigenvalues:].flip(0)