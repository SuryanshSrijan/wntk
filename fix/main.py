import os
import datetime
import numpy as np
import torch
import pickle as pkl
import argparse
import yaml
import itertools

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
    
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.num_nodes, int)
    
    # num_classes = data.y.max().item() + 1
    
    # assert isinstance(num_classes, int)
    
    ratio = sample_size / data.num_nodes
    
    train_nodes = data.train_mask.nonzero().squeeze()
    num_train = train_nodes.numel()
    num_sub_train = int(num_train * ratio)
    sub_train_nodes = train_nodes[torch.randperm(num_train)[:num_sub_train]]
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[sub_train_nodes] = True
    
    val_nodes = data.val_mask.nonzero().squeeze()
    num_val = val_nodes.numel()
    num_sub_val = int(num_val * ratio)
    sub_val_nodes = val_nodes[torch.randperm(num_val)[:num_sub_val]]
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    val_mask[sub_val_nodes] = True
    
    test_nodes = data.test_mask.nonzero().squeeze()
    num_test = test_nodes.numel()
    num_sub_test = sample_size - num_sub_train - num_sub_val
    sub_test_nodes = test_nodes[torch.randperm(num_test)[:num_sub_test]]
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    test_mask[sub_test_nodes] = True
    
    edge_index, edge_weight = torch_geometric.utils.subgraph(
        train_mask | val_mask | test_mask, data.edge_index, data.edge_weight, relabel_nodes=True, num_nodes=data.num_nodes
    )
    
    return Data(
        x=data.x[train_mask | val_mask | test_mask],
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=data.y[train_mask | val_mask | test_mask],
        train_mask=train_mask[train_mask | val_mask | test_mask],
        val_mask=val_mask[train_mask | val_mask | test_mask],
        test_mask=test_mask[train_mask | val_mask | test_mask],
    )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer', choices=['Cora', 'Citeseer', 'PubMed'])
    parser.add_argument('--seed', type=int, default=786)
    parser.add_argument('--jk', action='store_true', help='Use jumping knowledge in GNTK')
    parser.add_argument('--scale', type=str, default='degree', choices=['uniform', 'degree'])
    parser.add_argument('--save_dir', type=str, default='experiments', help='Directory to save results')
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, args.dataset, timestamp)
    os.makedirs(save_dir, exist_ok=True)
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
    
    assert isinstance(data.x, torch.Tensor)
    data.x = data.x[:, :feature_limit]
    
    num_classes: int = dataset.num_classes
    num_features: int = feature_limit
    
    assert isinstance(data.num_nodes, int)
    assert isinstance(data.edge_index, torch.Tensor)
    
    # Prepare full graph adjacency
    edge_weights = torch.full((data.num_edges,), 1.0 / data.num_nodes, device=DEVICE)
    full_adj = torch.sparse_coo_tensor(
        data.edge_index, edge_weights, 
        (data.num_nodes, data.num_nodes), device=DEVICE
    ).to_dense()
    
    # Prepare GNN architectures
    gnn_hidden_dims: List[int] = CONSTANTS['GNN_hidden_dims']
    gnn_num_layers: List[int] = CONSTANTS['GNN_num_layers']
    gnn_num_mlp_layers: List[int] = CONSTANTS['GNN_num_mlp_layers']
    gnn_architectures: List[List[int]] = []
    for (hidden_dim, num_layers, num_mlp_layers) in itertools.product(
        gnn_hidden_dims, gnn_num_layers, gnn_num_mlp_layers
    ):
        gnn_architectures.append([num_features, hidden_dim, num_classes, num_layers, num_mlp_layers])
    
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
                
                print(f"\nArchitecture {arch}")
                in_dim, hidden_dim, out_dim, num_layers, num_mlp_layers = arch
                # Initialize GNN
                gnn = GNN(f"gmm{arch_idx}", 'GNN', in_dim, hidden_dim, out_dim, num_layers, num_mlp_layers, False, device=DEVICE)
                
                # Train GNN
                val_losses, losses, best_gnn, _ = train(
                    sub_data, sub_data.train_mask, sub_data.val_mask,
                    gnn, torch.nn.CrossEntropyLoss(), train_args, logistic=True
                )
                
                fig = plt.figure(figsize=(10, 5))
                plt.plot(np.arange(10, len(losses)+1, 10), val_losses, label='Validation Loss')
                plt.plot(np.arange(1, len(losses)+1), losses, label='Training Loss')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title(f"Training and Validation Loss for Architecture {arch}")
                fig.savefig(os.path.join(save_dir, f"loss_plot_{sample_size}_{arch_idx+1}.png"))
                plt.close(fig)
                
                # Evaluate GNN
                assert isinstance(sub_data.y, torch.Tensor)
                assert isinstance(data.y, torch.Tensor)
                gnn_test_preds, gnn_test_loss = test(sub_data, sub_data.test_mask, best_gnn, logistic=True)
                gnn_transfer_preds, gnn_transf_loss = test(data, data.test_mask, best_gnn, logistic=True)
                gnn_test_acc = (gnn_test_preds.argmax(1) == sub_data.y[sub_data.test_mask]).float().mean().item()
                gnn_transf_acc = (gnn_transfer_preds.argmax(1) == data.y[data.test_mask]).float().mean().item()
                
                print(f"Test Loss: {gnn_test_loss:.4f}, Transfer Loss: {gnn_transf_loss:.4f}")
                print(f"Test Accuracy: {gnn_test_acc:.4f}, Transfer Accuracy: {gnn_transf_acc:.4f}")
                
                results['gnn'][rlz, size_idx, arch_idx] = gnn_test_loss
                results['gnn_transfer'][rlz, size_idx, arch_idx] = gnn_transf_loss
                
                # Get features for kernel
                assert isinstance(sub_data.x, torch.Tensor)
                train_feats, train_y = sub_data.x[sub_data.train_mask], sub_data.y[sub_data.train_mask]
                test_feats, test_y = sub_data.x[sub_data.test_mask], sub_data.y[sub_data.test_mask]
                full_feats, full_y = data.x[data.test_mask], data.y[data.test_mask]
                
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
                    train_features=[train_feats], 
                    train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
                    y_train=train_y
                )
                
                # Evaluate GNTK
                test_preds = gntk.predict(
                    train_features=[train_feats],
                    train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
                    y_train=train_y,
                    test_features=[test_feats],
                    test_adjs=[sub_adj[sub_data.test_mask][:, sub_data.test_mask]]
                )
                
                transfer_preds = gntk.predict(
                    train_features=[train_feats],
                    train_adjs=[sub_adj[sub_data.train_mask][:, sub_data.train_mask]],
                    y_train=train_y,
                    test_features=[full_feats],
                    test_adjs=[full_adj[data.test_mask][:, data.test_mask]]
                )
                
                # Calculate accuracies
                gntk_test_loss = torch.nn.functional.cross_entropy(test_preds, test_y).item()
                gntk_transfer_loss = torch.nn.functional.cross_entropy(transfer_preds, full_y).item()
                gntk_test_acc = (test_preds.argmax(1) == test_y).float().mean().item()
                gntk_transfer_acc = (transfer_preds.argmax(1) == full_y).float().mean().item()
                
                print(f"GNTK Test Loss: {gntk_test_loss:.4f}, GNTK Transfer Loss: {gntk_transfer_loss:.4f}")
                print(f"GNTK Test Accuracy: {gntk_test_acc:.4f}, GNTK Transfer Accuracy: {gntk_transfer_acc:.4f}")
                
                results['gntk'][rlz, size_idx, arch_idx] = gntk_test_loss
                results['gntk_transfer'][rlz, size_idx, arch_idx] = gntk_transfer_loss
    
    results_file = os.path.join(save_dir, f"results_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pkl.dump(results, f)
    
    print("\nFinal Results:")
    for key in results:
        print(f"{key}: {np.mean(results[key], axis=0)}")  # Average over realizations

    kernel_transf = np.abs((results['gntk'] - results['gntk_transfer']) / results['gntk_transfer'])
    kernel_transf_avg = np.mean(kernel_transf, axis=0)
    kernel_transf_std = np.std(kernel_transf, axis=0)
    
    for i in range(len(gnn_architectures)):
        fig = plt.figure()
        plt.title(f'Kernel transferability, arch: {gnn_architectures[i]}')
        plt.xlabel('Training graph size')
        plt.ylabel('Transferability error')
        plt.errorbar(train_subgraph_sizes, kernel_transf_avg[:,i], kernel_transf_std[:,i])
        fig.savefig(os.path.join(save_dir,'transf_kernel' + str(i) + '.png'), bbox_inches = 'tight')
        plt.close()

    # GNN and kernel transferability plot

    gnn_transf = np.abs((results['gnn'] - results['gnn_transfer']) / results['gnn_transfer'])
    gnn_transf_avg = np.mean(gnn_transf,axis=0)
    gnn_transf_std = np.std(gnn_transf,axis=0)

    for i in range(len(gnn_architectures)):
        fig = plt.figure()
        plt.title(f'Kernel transferability, arch: {gnn_architectures[i]}')
        plt.xlabel('Training graph size')
        plt.ylabel('Transferability error')
        plt.errorbar(train_subgraph_sizes, kernel_transf_avg[:,i], kernel_transf_std[:,i], label='GNTK')
        plt.errorbar(train_subgraph_sizes, gnn_transf_avg[:,i], gnn_transf_std[:,i], label='GNN')
        plt.legend()
        #plt.show()
        fig.savefig(os.path.join(save_dir,'transf_kernel_gnn' + str(i) + '.png'), bbox_inches = 'tight')
        plt.close()

if __name__ == '__main__':
    main()