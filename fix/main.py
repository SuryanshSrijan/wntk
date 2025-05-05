import os
import datetime
import numpy as np
import torch
import pickle as pkl
import argparse
import yaml
import copy

from typing import List, Tuple, Dict, Any
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.utils

import matplotlib.pyplot as plt

from GNN import GNN
from Trainer import train, test
from Kernel import GNTKernelRegression  # Changed to use our new GNTK implementation

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load constants
constants_file = os.path.join(os.path.dirname(__file__), 'constants.yaml')
with open(constants_file, 'r') as file:
    CONSTANTS = yaml.safe_load(file)

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_balanced_masks(y: torch.Tensor, num_classes: int, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create balanced train/val/test masks"""
    train_mask = torch.zeros_like(y, dtype=torch.bool)
    val_mask = torch.zeros_like(y, dtype=torch.bool)
    test_mask = torch.zeros_like(y, dtype=torch.bool)
    
    for cls in range(num_classes):
        cls_indices = (y == cls).nonzero(as_tuple=True)[0]
        num_cls = cls_indices.numel()
        
        num_train = num_cls // 2
        num_val = num_cls // 4
        num_test = num_cls - num_train - num_val
        
        selected_indices = cls_indices[torch.randperm(num_cls)]
        train_mask[selected_indices[:num_train]] = True
        val_mask[selected_indices[num_train:num_train + num_val]] = True
        test_mask[selected_indices[num_train + num_val:num_train + num_val + num_test]] = True
    
    return train_mask, val_mask, test_mask

def sample_subgraph(data: Data, sample_size: int) -> Data:
    """Sample a k-hop subgraph with balanced classes"""
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    
    num_classes = data.y.max().item() + 1
    
    assert isinstance(num_classes, int)
    
    while True:
        assert isinstance(data.num_nodes, int)
        seed = torch.randint(0, data.num_nodes, (1,), device=DEVICE)
        node_idx, edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            seed.tolist(), num_hops=sample_size, edge_index=data.edge_index,
            relabel_nodes=True, num_nodes=data.num_nodes
        )

        if node_idx.size(0) < sample_size:
            continue

        selected = node_idx[:sample_size] 
        selected_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        selected_mask[selected] = True
        
        # edge_mask = selected_mask[data.edge_index[0]] & selected_mask[data.edge_index[1]]
        # edge_index = data.edge_index[:, edge_mask]

        relabeled_edge_index, *_ = torch_geometric.utils.subgraph(
            selected_mask, data.edge_index, relabel_nodes=True
        )
        
        train_mask, val_mask, test_mask = create_balanced_masks(
            data.y[selected], num_classes, sample_size
        )
        
        sampled_data = Data(
            x=data.x[selected],
            edge_index=relabeled_edge_index,
            y=data.y[selected],
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        ).to(DEVICE)
        
        assert isinstance(sampled_data.y, torch.Tensor)
        if all((sampled_data.y[sampled_data.train_mask] == cls).sum() > 0 for cls in range(num_classes)):
            return sampled_data

def get_features(data: Data, mask: torch.Tensor, model: GNN) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Extract intermediate features from GNN"""
    features = model(data, return_intermediate=True)
    
    assert isinstance(data.y, torch.Tensor)
    y = data.y[mask]
    
    # Process features for kernel
    processed_features = []
    for feat in features:
        feat_masked = feat[mask].unsqueeze(0)  # Add batch dimension
        processed_features.append(feat_masked)
    
    return processed_features, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer', choices=['Cora', 'Citeseer', 'PubMed'])
    parser.add_argument('--seed', type=int, default=786)
    parser.add_argument('--jk', action='store_true', help='Use jumping knowledge in GNTK')
    parser.add_argument('--scale', type=str, default='degree', choices=['uniform', 'degree'])
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load dataset
    dataset = Planetoid(root=f'/tmp/{args.dataset}', name=args.dataset, split='full')
    data = dataset[0]
    assert isinstance(data, Data)
    data = data.to(DEVICE)
    
    # Get parameters from constants
    train_subgraph_sizes: List[int] = CONSTANTS['train_sample_sizes'][args.dataset]
    feature_limit: int = CONSTANTS['feature_limit'][args.dataset]    
    if feature_limit == -1: 
        feature_limit = dataset.num_features
    
    num_classes: int = dataset.num_classes
    num_features: int = dataset.num_features
    
    assert isinstance(data.num_nodes, int)
    assert isinstance(data.edge_index, torch.Tensor)
    
    # Prepare full graph adjacency
    edge_weights = torch.full((data.num_edges,), 1.0 / data.num_nodes, device=DEVICE)
    full_adj = torch.sparse_coo_tensor(
        data.edge_index, edge_weights, 
        (data.num_nodes, data.num_nodes), device=DEVICE
    ).to_dense()
    
    # Prepare GNN architectures
    gnn_architectures: List[List[List[int]]] = CONSTANTS['GNN_architecture'][args.dataset]
    for arch in gnn_architectures:
        arch[0][0] = num_features  # Set input dimension
        arch[-2][-1] = num_classes  # Set output dimension
    
    num_realizations = CONSTANTS['num_realizations']
    train_args = CONSTANTS['train_args']
    
    # Initialize result storage
    results = {
        'gnn': np.zeros((num_realizations, len(train_subgraph_sizes), len(gnn_architectures))),
        'gnn_transfer': np.zeros((num_realizations, len(train_subgraph_sizes), len(gnn_architectures))),
        'gntk': np.zeros((num_realizations, len(train_subgraph_sizes), len(gnn_architectures))),
        'gntk_transfer': np.zeros((num_realizations, len(train_subgraph_sizes), len(gnn_architectures)))
    }
    
    for rlz in range(num_realizations):
        print(f"\nRealization {rlz+1}/{num_realizations}")
        
        for size_idx, sample_size in enumerate(train_subgraph_sizes):
            print(f"\nSample size: {sample_size}")
            
            assert isinstance(data, Data)
            # Sample subgraph
            sub_data = sample_subgraph(data, sample_size)
            
            # Create subgraph adjacency
            assert isinstance(sub_data.edge_index, torch.Tensor)
            assert isinstance(sub_data.num_nodes, int)
            edge_weights = torch.full((sub_data.edge_index.size(1),), 1.0/sub_data.num_nodes, device=DEVICE)
            sub_adj = torch.sparse_coo_tensor(
                sub_data.edge_index, edge_weights,
                (sub_data.num_nodes, sub_data.num_nodes), device=DEVICE
            ).to_dense()
            
            # Train and evaluate each GNN architecture
            for arch_idx, arch in enumerate(gnn_architectures):
                print(f"Architecture {arch_idx+1}/{len(gnn_architectures)}")
                
                # breakpoint()
                # Initialize GNN
                gnn = GNN(f"gnn{arch_idx}", 'GNN', arch, False, device=DEVICE)
                original_gnn = copy.deepcopy(gnn)
                
                # Train GNN
                _, _, best_gnn, _ = train(
                    sub_data, sub_data.train_mask, sub_data.val_mask,
                    gnn, torch.nn.CrossEntropyLoss(), train_args, logistic=True
                )
                
                # Evaluate GNN
                gnn_test_acc = test(sub_data, sub_data.test_mask, best_gnn, logistic=True)
                gnn_transfer_acc = test(data, data.test_mask, best_gnn, logistic=True)
                
                results['gnn'][rlz, size_idx, arch_idx] = gnn_test_acc
                results['gnn_transfer'][rlz, size_idx, arch_idx] = gnn_transfer_acc
                
                # Prepare GNTK
                num_layers = len(arch) - 1  # Number of GNN layers
                num_mlp_layers = 1 
                
                # Get features for kernel
                train_feats, train_y = get_features(sub_data, sub_data.train_mask, original_gnn)
                test_feats, test_y = get_features(sub_data, sub_data.test_mask, original_gnn)
                full_feats, full_y = get_features(data, data.test_mask, original_gnn)
                
                # breakpoint()
                # Initialize GNTK
                gntk = GNTKernelRegression(
                    num_layers=num_layers,
                    num_mlp_layers=num_mlp_layers,
                    jk=args.jk,
                    scale=args.scale,
                    logistic=True,
                    device=DEVICE
                )
                
                # Fit GNTK
                gntk.fit(
                    train_features=[[f[0] for f in train_feats]],  # Remove batch dim
                    train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
                    y_train=train_y
                )
                
                # Evaluate GNTK
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
                    test_adjs=[full_adj[data.test_mask][:, data.test_mask]]
                )
                
                # Calculate accuracies
                gntk_test_acc = (test_preds.argmax(1) == test_y).float().mean().item()
                gntk_transfer_acc = (transfer_preds.argmax(1) == full_y).float().mean().item()
                
                results['gntk'][rlz, size_idx, arch_idx] = gntk_test_acc
                results['gntk_transfer'][rlz, size_idx, arch_idx] = gntk_transfer_acc
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{args.dataset}_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pkl.dump(results, f)
    
    print("\nFinal Results:")
    for key in results:
        print(f"{key}: {np.mean(results[key], axis=0)}")  # Average over realizations

if __name__ == '__main__':
    main()