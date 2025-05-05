import os
import datetime
import argparse
import pickle as pkl
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.utils

from GNN import GNN
from Trainer import train, test
from Kernel import GNTKernelRegression

# Constants and Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONSTANTS_FILE = os.path.join(os.path.dirname(__file__), 'constants.yaml')

def load_constants():
    """Load configuration constants from YAML file"""
    with open(CONSTANTS_FILE, 'r') as file:
        return yaml.safe_load(file)

CONSTANTS = load_constants()

class GraphExperiment:
    """Main class for running graph experiments comparing GNN and GNTK performance"""
    
    def __init__(self):
        self.args = self.parse_args()
        self.set_seed(self.args.seed)
        self.dataset = None
        self.data = None
        self.results = None

    @staticmethod
    def parse_args():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='Citeseer', 
                          choices=['Cora', 'Citeseer', 'PubMed'])
        parser.add_argument('--seed', type=int, default=786)
        parser.add_argument('--jk', action='store_true', 
                          help='Use jumping knowledge in GNTK')
        parser.add_argument('--scale', type=str, default='degree', 
                          choices=['uniform', 'degree'])
        return parser.parse_args()

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_dataset(self):
        """Load and prepare the graph dataset"""
        self.dataset = Planetoid(root=f'/tmp/{self.args.dataset}', 
                               name=self.args.dataset, split='full')
        self.data = self.dataset[0].to(DEVICE)

    def create_balanced_masks(self, y: torch.Tensor, num_classes: int, 
                            sample_size: int) -> Tuple[torch.Tensor, ...]:
        """Create balanced train/val/test masks with equal class distribution"""
        train_mask = torch.zeros_like(y, dtype=torch.bool)
        val_mask = torch.zeros_like(y, dtype=torch.bool)
        test_mask = torch.zeros_like(y, dtype=torch.bool)
        
        for cls in range(num_classes):
            cls_indices = (y == cls).nonzero(as_tuple=True)[0]
            permuted = cls_indices[torch.randperm(len(cls_indices))]
            
            # Split indices into train/val/test
            splits = torch.tensor([0.5, 0.25, 0.25]) * len(cls_indices)
            splits = splits.int().cumsum(0)
            
            train_mask[permuted[:splits[0]]] = True
            val_mask[permuted[splits[0]:splits[1]]] = True
            test_mask[permuted[splits[1]:]] = True
        
        return train_mask, val_mask, test_mask

    def sample_subgraph(self, sample_size: int) -> Data:
        """Sample a k-hop subgraph with balanced classes"""
        while True:
            # Select random seed node
            seed = torch.randint(0, self.data.num_nodes, (1,), device=DEVICE)
            
            # Get k-hop neighborhood
            node_idx, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                seed.tolist(), num_hops=sample_size, 
                edge_index=self.data.edge_index,
                relabel_nodes=True, num_nodes=self.data.num_nodes
            )

            if len(node_idx) < sample_size:
                continue

            # Create masks for selected nodes
            selected = node_idx[:sample_size]
            selected_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            selected_mask[selected] = True
            
            # Create subgraph
            edge_index, _ = torch_geometric.utils.subgraph(
                selected_mask, self.data.edge_index, relabel_nodes=True
            )
            
            # Create balanced masks
            train_mask, val_mask, test_mask = self.create_balanced_masks(
                self.data.y[selected], self.dataset.num_classes, sample_size
            )
            
            return Data(
                x=self.data.x[selected],
                edge_index=edge_index,
                y=self.data.y[selected],
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            ).to(DEVICE)

    def get_features(self, data: Data, mask: torch.Tensor, 
                    model: GNN) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract intermediate features from GNN for kernel computation"""
        features = model(data, return_intermediate=True)
        y = data.y[mask]
        
        # Process features for kernel input
        processed_features = [f[mask].unsqueeze(0) for f in features]
        return processed_features, y

    def prepare_adjacency(self, data: Data) -> torch.Tensor:
        """Create dense adjacency matrix with normalized edge weights"""
        edge_weights = torch.full((data.edge_index.size(1),), 
                                1.0/data.num_nodes, device=DEVICE)
        return torch.sparse_coo_tensor(
            data.edge_index, edge_weights,
            (data.num_nodes, data.num_nodes), device=DEVICE
        ).to_dense()

    def run_gnn_experiment(self, sub_data: Data, arch: List[List[int]]) -> Tuple[float, float]:
        """Train and evaluate GNN model"""
        gnn = GNN(f"gnn{len(arch)}", 'GNN', arch, False, device=DEVICE)
        original_gnn = copy.deepcopy(gnn)
        
        # Train GNN
        _, _, best_gnn, _ = train(
            sub_data, sub_data.train_mask, sub_data.val_mask,
            gnn, torch.nn.CrossEntropyLoss(), 
            CONSTANTS['train_args'][self.args.dataset], 
            logistic=True
        )
        
        # Evaluate performance
        test_acc = test(sub_data, sub_data.test_mask, best_gnn, logistic=True)
        transfer_acc = test(self.data, self.data.test_mask, best_gnn, logistic=True)
        
        return test_acc, transfer_acc, original_gnn

    def run_gntk_experiment(self, sub_data: Data, original_gnn: GNN, 
                           sub_adj: torch.Tensor, num_layers: int) -> Tuple[float, float]:
        """Train and evaluate GNTK model"""
        # Extract features for kernel
        train_feats, train_y = self.get_features(sub_data, sub_data.train_mask, original_gnn)
        test_feats, test_y = self.get_features(sub_data, sub_data.test_mask, original_gnn)
        full_feats, full_y = self.get_features(self.data, self.data.test_mask, original_gnn)

        # Initialize and fit GNTK
        gntk = GNTKernelRegression(
            num_layers=num_layers,
            num_mlp_layers=1,
            jk=self.args.jk,
            scale=self.args.scale,
            logistic=True,
            device=DEVICE
        )
        
        gntk.fit(
            train_features=[[f[0] for f in train_feats]],
            train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
            y_train=train_y
        )

        # Make predictions
        test_preds = gntk.predict(
            train_features=[[f[0] for f in train_feats]],
            train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
            y_train=train_y,
            test_features=[[f[0] for f in test_feats]],
            test_adjs=[sub_adj[sub_data.test_mask][:, sub_data.test_mask]]
        )
        
        transfer_preds = gntk.predict(
            train_features=[[f[0] for f in train_feats]],
            train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
            y_train=train_y,
            test_features=[[f[0] for f in full_feats]],
            test_adjs=[self.prepare_adjacency(self.data)[self.data.test_mask][:, self.data.test_mask]]
        )

        # Calculate accuracies
        test_acc = (test_preds.argmax(1) == test_y).float().mean().item()
        transfer_acc = (transfer_preds.argmax(1) == full_y).float().mean().item()
        
        return test_acc, transfer_acc

    def initialize_results_structure(self):
        """Create results dictionary with proper dimensions"""
        train_sizes = CONSTANTS['train_sample_sizes'][self.args.dataset]
        architectures = CONSTANTS['GNN_architecture'][self.args.dataset]
        realizations = CONSTANTS['num_realizations']
        
        return {
            'gnn': np.zeros((realizations, len(train_sizes), len(architectures))),
            'gnn_transfer': np.zeros((realizations, len(train_sizes), len(architectures))),
            'gntk': np.zeros((realizations, len(train_sizes), len(architectures))),
            'gntk_transfer': np.zeros((realizations, len(train_sizes), len(architectures)))
        }

    def run_experiments(self):
        """Main experiment loop across realizations, sample sizes and architectures"""
        self.results = self.initialize_results_structure()
        train_sizes = CONSTANTS['train_sample_sizes'][self.args.dataset]
        architectures = CONSTANTS['GNN_architecture'][self.args.dataset]

        for rlz in range(CONSTANTS['num_realizations']):
            print(f"\nRealization {rlz+1}/{CONSTANTS['num_realizations']}")
            
            for size_idx, sample_size in enumerate(train_sizes):
                print(f"\nSample size: {sample_size}")
                sub_data = self.sample_subgraph(sample_size)
                sub_adj = self.prepare_adjacency(sub_data)

                for arch_idx, arch in enumerate(architectures):
                    print(f"Architecture {arch_idx+1}/{len(architectures)}")
                    num_layers = len(arch) - 1

                    # Run GNN experiments
                    gnn_test, gnn_transfer, original_gnn = self.run_gnn_experiment(sub_data, arch)
                    self.results['gnn'][rlz, size_idx, arch_idx] = gnn_test
                    self.results['gnn_transfer'][rlz, size_idx, arch_idx] = gnn_transfer

                    # Run GNTK experiments
                    gntk_test, gntk_transfer = self.run_gntk_experiment(
                        sub_data, original_gnn, sub_adj, num_layers
                    )
                    self.results['gntk'][rlz, size_idx, arch_idx] = gntk_test
                    self.results['gntk_transfer'][rlz, size_idx, arch_idx] = gntk_transfer

    def save_results(self):
        """Save results to pickle file with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{self.args.dataset}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pkl.dump(self.results, f)
        
        print("\nFinal Results:")
        for key, arr in self.results.items():
            print(f"{key} mean: {np.mean(arr, axis=0)}")

    def run(self):
        """Main execution method"""
        self.load_dataset()
        self.run_experiments()
        self.save_results()

if __name__ == '__main__':
    experiment = GraphExperiment()
    experiment.run()