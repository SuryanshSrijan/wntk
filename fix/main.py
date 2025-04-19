import os
import datetime
import numpy as np
import torch
import pickle as pkl
import argparse
import yaml

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

import matplotlib.pyplot as plt

from GNN import GNN
from Trainer import train, test
from Kernel import KernelRegression

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

constants_file = os.path.join(os.path.dirname(__file__), 'constants.yaml')
with open(constants_file, 'r') as file:
    CONSTANTS = yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer')
    parser.add_argument('--seed', type=int, default=786)
    args = parser.parse_args()
    
    assert args.dataset in ['Cora', 'Citeseer', 'PubMed'], "Dataset not supported. Choose from ['Cora', 'Citeseer', 'PubMed']"
    set_seed(args.seed)
    
    dataset = Planetoid(root=f'/tmp/{args.dataset}', name=args.dataset, split='full').to(DEVICE)
    
    train_subgraph_size: list[int] = CONSTANTS['train_sample_sizes'][args.dataset]
    
    feature_limit: int = CONSTANTS['feature_limit'][args.dataset]    
    if feature_limit == -1: 
        feature_limit = dataset.num_features
    
    assert isinstance(dataset._data, Data)
    num_classes: int = dataset.num_classes
    num_features: int = dataset.num_features
    
    assert isinstance(dataset._data.num_nodes, int)
    num_nodes = dataset._data.num_nodes
    
    assert isinstance(dataset._data.edge_index, torch.Tensor)
    edge_list = dataset._data.edge_index.clone().to(DEVICE)
    
    assert edge_list.ndim == 2 and edge_list.shape[0] == 2
    assert edge_list.shape[1] == dataset._data.num_edges

    num_edges: int = edge_list.shape[1]
    edge_weights: torch.Tensor = torch.full((num_edges,), 1.0 / num_nodes, device=DEVICE)
    all_adjacency_matrix: torch.Tensor = torch.sparse_coo_tensor(edge_list, edge_weights, (num_nodes, num_nodes), device=DEVICE).to_dense()
    
    assert isinstance(dataset._data.x, torch.Tensor)
    assert isinstance(dataset._data.y, torch.Tensor)
    all_test_data = dataset._data.subgraph(dataset._data.test_mask)
    # all_test_data.x = all_test_data.x[:, :feature_limit]
    loader_all_test = DataLoader([all_test_data], batch_size=1, shuffle=False)
    
    GNN_architectures = CONSTANTS['GNN_architecture'][args.dataset]
    num_models = len(GNN_architectures)
    
    for i in range(num_models):
        GNN_architectures[i][0][0] = num_features
        GNN_architectures[i][-2][-1] = num_classes    
    
    num_realizations = CONSTANTS['num_realizations']
    
    train_args = CONSTANTS['train_args'][args.dataset]
    
    gnn_results             = np.zeros((num_realizations, len(train_subgraph_size), num_models))
    gnn_transf_results      = np.zeros((num_realizations, len(train_subgraph_size), num_models))
    kernel_results          = np.zeros((num_realizations, len(train_subgraph_size), num_models))
    kernel_transf_results   = np.zeros((num_realizations, len(train_subgraph_size), num_models))
    
    for rlz in range(num_realizations):
        
        for sample_ind, sample_size in enumerate(train_subgraph_size):
            
            while True:
                sampled_data = dataset._data.subgraph(torch.randint(0, dataset._data.num_nodes, (sample_size,)).to(DEVICE))
                
                assert isinstance(sampled_data.y, torch.Tensor)
                if all((sampled_data.y[sampled_data.train_mask] == cls).sum() > 0 for cls in range(num_classes)):
                    break
            
            assert isinstance(sampled_data.x, torch.Tensor)
            # sampled_data.x = sampled_data.x[:, :feature_limit]
                
            assert isinstance(sampled_data.edge_index, torch.Tensor)
            edge_list = sampled_data.edge_index.clone()
            
            num_edges = edge_list.shape[1]
            edge_weights = torch.full((num_edges,), 1.0 / sample_size, device=DEVICE)
            adjacency_matrix = torch.sparse_coo_tensor(edge_list, edge_weights, (sample_size, sample_size)).to_dense()
            
            train_data = sampled_data.subgraph(sampled_data.train_mask)
            # train_data.x = train_data.x[:, :feature_limit]
            
            val_data = sampled_data.subgraph(sampled_data.val_mask)
            # val_data.x = val_data.x[:, :feature_limit]
            
            test_data = sampled_data.subgraph(sampled_data.test_mask)
            # test_data.x = test_data.x[:, :feature_limit]
            
            models = [GNN(f"gnn{cnt}", 'GNN', arch, False, device=DEVICE) for cnt, arch in enumerate(GNN_architectures)]
            
            loss_fn = torch.nn.CrossEntropyLoss()
            
            for model_ind, model in enumerate(models):
                
                loader_train = DataLoader([train_data], batch_size=train_args['batch_size'], shuffle=True)
                loader_val = DataLoader([val_data], batch_size=1, shuffle=False)
                
                val_losses, losses, best_model, best_loss = train(
                    loader_train, loader_val, model, loss_fn, train_args, logistic=True
                )
                
                loader_test = DataLoader([test_data], batch_size=1, shuffle=False)
                test_loss = test(loader_test, best_model, logistic=True)
                transf_test_loss = test(loader_all_test, best_model, logistic=True)

                gnn_results[rlz, sample_ind, model_ind] = test_loss
                gnn_transf_results[rlz, sample_ind, model_ind] = transf_test_loss
    
    print('GNN results:', gnn_results)
    print('GNN transf results:', gnn_transf_results)
                
if __name__ == '__main__':
    main()