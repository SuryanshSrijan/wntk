import torch

EPS  = 1e-9

class KernelRegression:
    
    def __init__(self, num_flayers: int, num_filters: list[int], filter_widths: list[int], adj_matrix: torch.Tensor, logistic: bool = False, device: str = 'cpu'):
        
        self.num_flayers = num_flayers
        self.num_filters = num_filters
        self.filter_widths = filter_widths
        self.adj_matrix = adj_matrix
        self.logistic = logistic
        self.kernel = None
        self.reg = None
        self.device = device
        
    def compute_gradient(self, x: list[torch.Tensor], weights: list[torch.Tensor], S: torch.Tensor | None = None) -> list[torch.Tensor]:
        
        assert False, "This function is not implemented yet."
    
    def predict(
        self,
        x_train: list[torch.Tensor],
        weights: list[torch.Tensor],
        y_train: torch.Tensor,
        x_test: list[torch.Tensor],
        adj_matrix: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        
        assert False, "This function is not implemented yet."
    
    def get_eigenvalues(self) -> torch.Tensor:
        
        assert False, "This function is not implemented yet."