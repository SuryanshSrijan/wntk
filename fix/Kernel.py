import torch
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

EPS  = 1e-9

class KernelRegression:
    
    def __init__(self, num_layers: int, num_filters: list[int], filter_widths: list[int], adj_matrix: torch.Tensor, logistic: bool = False, device: str = 'cpu'):
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_widths = filter_widths
        self.adj_matrix = adj_matrix
        self.logistic = logistic
        self.kernel = None
        self.reg = None
        self.device = device
        
    def compute_gradient(self, x: list[torch.Tensor], weights: list[torch.Tensor], adj_mat: torch.Tensor | None = None) -> torch.Tensor:
        
        num_signals = x[0].shape[0]
        num_nodes = x[0].shape[1]
        
        for i in range(self.num_layers + 1):
            assert x[i].ndim == 3
            assert x[i].shape[0] == num_signals
            assert x[i].shape[1] == num_nodes
            assert x[i].shape[2] == self.filter_widths[i]
        
        for i in range(self.num_layers):
            assert weights[i].ndim == 3
            assert weights[i].shape[0] == self.filter_widths[i]
            assert weights[i].shape[1] == self.filter_widths[i + 1]
            assert weights[i].shape[2] == self.num_filters[i]
        
        if adj_mat is None:
            adj_mat = self.adj_matrix
        
        assert adj_mat.ndim == 2
        assert adj_mat.shape[0] == num_nodes
        assert adj_mat.shape[1] == num_nodes
        
        max_num_filters = max(self.num_filters)
        adj_mat_powers = [torch.eye(num_nodes, device=self.device)]
        for i in range(1, max_num_filters):
            adj_mat_powers.append(torch.matmul(adj_mat_powers[-1], adj_mat))
        
        grads = []
        
        for l in range(self.num_layers):
            
            grad_l = torch.zeros(
                num_signals, num_nodes, self.filter_widths[l + 1], 
                self.filter_widths[l], self.num_filters[l],
                device=self.device
            )
            
            for k in range(self.num_filters[l]):
                
                grad_l[:, :, :, :, k] = torch.einsum(
                    'nm,smf->snf',
                    adj_mat_powers[k],
                    x[l]
                ).unsqueeze(2).expand(-1, -1, self.filter_widths[l + 1], -1)
        
            if l < self.num_layers - 1:
                grad_l = grad_l * (grad_l > 0).float()
            
            for l2 in range(l + 1, self.num_layers):
                
                grad_reshaped = grad_l.reshape(
                    num_signals, num_nodes,
                    self.filter_widths[l + 1] * self.num_filters[l],
                    self.filter_widths[l] 
                )
                
                grad_l = torch.einsum(
                    'snil,ijk->snjk',
                    grad_reshaped,
                    weights[l2].reshape(self.filter_widths[l2], -1)
                )
                
                if l2 < self.num_layers - 1:
                    grad_l = grad_l * (grad_l > 0).float()
            
            grad_l = grad_l.reshape(
                num_signals, num_nodes, self.filter_widths[-1], -1
            )
            
            if self.logistic:
                
                y = torch.nn.functional.softmax(x[-1])
                y_flat = y.reshape(-1, self.filter_widths[-1])
                soft = torch.einsum(
                    'bi,bj->bij',
                    y_flat, y_flat
                )
                diag = torch.diag_embed(y_flat * (1 - y_flat))
                grad_l = torch.einsum(
                    'bij,bj->bi',
                    diag - soft,
                    grad_l.reshape(-1, self.filter_widths[-1])
                ).reshape(grad_l.shape)
            
            grads.append(grad_l)
        
        return torch.cat(grads, dim=-1)
    
    def fit(self, x_train: list[torch.Tensor], weights: list[torch.Tensor], y_train: torch.Tensor) -> LogisticRegression | Ridge:
        
        grads = self.compute_gradient(x_train, weights)
        num_signals, num_nodes, out_dim = grads.shape[:3]
        ntk_features = grads.reshape(-1, grads.shape[-1])
        
        kernel_matrix = ntk_features @ ntk_features.T
        
        y_train_flat = y_train.reshape(-1).cpu().numpy()
        
        kernel_matrix_sparse = sparse.csr_matrix(kernel_matrix.cpu().numpy())

        if self.logistic:
            self.reg = LogisticRegression(
                penalty=None,
                fit_intercept=False,
                max_iter=1000,
                multi_class='multinomial',
            )
        
        else:
            self.reg = Ridge(
                alpha=0.0,
                fit_intercept=False,
                solver='lsqr',
            )
        
        self.kernel = kernel_matrix
        self.reg.fit(kernel_matrix_sparse, y_train_flat)
        
        return self.reg
    
    
    def predict(
        self,
        x_train: list[torch.Tensor],
        weights: list[torch.Tensor],
        y_train: torch.Tensor,
        x_test: list[torch.Tensor],
        adj_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if self.reg is None:
            self.fit(x_train, weights, y_train)
        
        test_grads = self.compute_gradient(x_test, weights, adj_matrix)
        num_test_signals = test_grads.shape[0]
        
        if self.logistic:
            assert self.kernel is not None and isinstance(self.reg, LogisticRegression)
            phi_test = test_grads.reshape(num_test_signals, -1)
            K_test = (phi_test @ self.kernel.T).cpu().numpy()
            preds = self.reg.predict_proba(sparse.csr_matrix(K_test))
            
            return torch.from_numpy(preds).to(self.device).reshape(num_test_signals, -1)
        
        else:
            assert self.kernel is not None and isinstance(self.reg, Ridge)
            phi_test = test_grads.reshape(-1, test_grads.shape[-1])
            K_test = (phi_test @ self.kernel.T).cpu().numpy()
        
        assert False, "This function is not implemented yet."
    
    def get_eigenvalues(self) -> torch.Tensor:
        
        assert False, "This function is not implemented yet."