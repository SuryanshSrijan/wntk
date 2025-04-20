import os
import datetime
import numpy as np
import torch
import pickle as pkl
import argparse
import yaml
import copy

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.utils

import matplotlib.pyplot as plt

from GNN import GNN
from Trainer import train, test
from Kernel import KernelRegression

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

constants_file = os.path.join(os.path.dirname(__file__), 'constants.yaml')
with open(constants_file, 'r') as file:
    CONSTANTS = yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_balanced_masks(y: torch.Tensor, num_classes: int, sample_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
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
    num_classes = data.y.max().item() + 1
    assert isinstance(num_classes, int)
    assert isinstance(data.num_nodes, int)
    
    while True:

        seed = torch.randint(0, data.num_nodes, (1,), device=DEVICE)
        node_idx, edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            seed.tolist(), num_hops=sample_size, edge_index=data.edge_index,
            relabel_nodes=True
        )

        if node_idx.size(0) < sample_size:
            continue

        selected = node_idx[:sample_size] 
        selected_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        selected_mask[selected] = True
        
        edge_mask = selected_mask[data.edge_index[0]] & selected_mask[data.edge_index[1]]
        edge_index = data.edge_index[:, edge_mask]

        relabeled_edge_index, relabeled_mapping = torch_geometric.utils.subgraph(
            selected_mask, data.edge_index, relabel_nodes=True
        )
        
        train_mask, val_mask, test_mask = create_balanced_masks(
            data.y[selected], num_classes, sample_size
        )
        
        sampled_data = Data(
            x = data.x[selected],
            edge_index = relabeled_edge_index,
            y = data.y[selected],
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
        ).to(DEVICE)
        
        assert isinstance(sampled_data.y, torch.Tensor)
        if all((sampled_data.y[sampled_data.train_mask] == cls).sum() > 0 for cls in range(num_classes)):
            return sampled_data

def get_features(data: Data, mask: torch.Tensor, model: GNN, best_model: GNN) -> tuple[list[torch.Tensor], torch.Tensor]:
    
    features = []
    good_nodes = len(mask[mask == True])
    
    assert isinstance(data.y, torch.Tensor)
    y_test = data.y.detach()[mask]
    y_test = torch.reshape(y_test, (1, good_nodes, -1))
    
    features = model(data, return_intermediate=True)
    preds = best_model(data)[mask]
    preds = torch.reshape(preds, (1, good_nodes, -1)).detach()
    
    for i in range(len(features)):
        features[i] = torch.reshape(features[i][mask], (1, good_nodes, -1)).detach()
    
    return features, preds

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer')
    parser.add_argument('--seed', type=int, default=786)
    args = parser.parse_args()
    
    assert args.dataset in ['Cora', 'Citeseer', 'PubMed'], "Dataset not supported. Choose from ['Cora', 'Citeseer', 'PubMed']"
    set_seed(args.seed)
    
    dataset = Planetoid(root=f'/tmp/{args.dataset}', name=args.dataset, split='full').to(DEVICE)
    
    all_data = dataset[0]
    assert isinstance(all_data, Data)
    all_data = all_data.to(DEVICE)
    
    train_subgraph_size: list[int] = CONSTANTS['train_sample_sizes'][args.dataset]
    
    feature_limit: int = CONSTANTS['feature_limit'][args.dataset]    
    if feature_limit == -1: 
        feature_limit = dataset.num_features
    
    num_classes: int = dataset.num_classes
    num_features: int = dataset.num_features
    
    assert isinstance(all_data.num_nodes, int)
    num_nodes = all_data.num_nodes
    
    assert isinstance(all_data.edge_index, torch.Tensor)
    edge_list = all_data.edge_index.clone().to(DEVICE)
    
    assert edge_list.ndim == 2 and edge_list.shape[0] == 2
    assert edge_list.shape[1] == all_data.num_edges

    num_edges: int = edge_list.shape[1]
    edge_weights: torch.Tensor = torch.full((num_edges,), 1.0 / num_nodes, device=DEVICE)
    all_adjoint_matrix: torch.Tensor = torch.sparse_coo_tensor(edge_list, edge_weights, (num_nodes, num_nodes), device=DEVICE).to_dense()
        
    GNN_architectures: list[list[list[int]]] = CONSTANTS['GNN_architecture'][args.dataset]
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
            
            sampled_data = sample_subgraph(all_data, sample_size)
            
            num_subgraph_nodes = sampled_data.num_nodes
            assert isinstance(num_subgraph_nodes, int)
            print(sample_size, num_subgraph_nodes)
            assert isinstance(sampled_data.edge_index, torch.Tensor)
            edge_list = sampled_data.edge_index.clone()
            
            num_edges = edge_list.shape[1]
            edge_weights = torch.full((num_edges,), 1.0 / num_subgraph_nodes, device=DEVICE)
            adjoint_matrix = torch.sparse_coo_tensor(edge_list, edge_weights, (num_subgraph_nodes, num_subgraph_nodes)).to_dense()
            
            models = [GNN(f"gnn{cnt}", 'GNN', arch, False, device=DEVICE) for cnt, arch in enumerate(GNN_architectures)]
            
            loss_fn = torch.nn.CrossEntropyLoss()
            
            for model_ind, model in enumerate(models):
                
                original_model = copy.deepcopy(model)
                
                
                val_losses, losses, best_model, best_loss = train(
                    sampled_data, sampled_data.train_mask, sampled_data.val_mask, model, loss_fn, train_args, logistic=True
                )
                
                test_loss = test(sampled_data, sampled_data.test_mask, best_model, logistic=True)
                transf_test_loss = test(all_data, all_data.test_mask, best_model, logistic=True)

                gnn_results[rlz, sample_ind, model_ind] = test_loss
                gnn_transf_results[rlz, sample_ind, model_ind] = transf_test_loss
                                
                feats_train, _                  = get_features(sampled_data, sampled_data.train_mask, original_model, best_model)
                feats_test, test_preds          = get_features(sampled_data, sampled_data.test_mask, original_model, best_model)
                feats_all_test, all_test_preds  = get_features(all_data, all_data.test_mask, original_model, best_model)
                    
                consF = model.feature_list[:-1] + model.mlp_list
                consK = model.K_list + [1] * model.num_mlp_layers
                
                kernel = KernelRegression(
                    len(consF) - 1, consK, consF, adjoint_matrix, logistic=True
                )
                
                weight_list = original_model.get_weights()
                
                breakpoint()
                
                assert isinstance(sampled_data.y, torch.Tensor)
                kernel_preds = kernel.predict(
                    feats_train, weight_list, sampled_data.y[sampled_data.train_mask], feats_test
                )
                
                eig_val = kernel.get_eigenvalues() / num_subgraph_nodes
                
                # assert isinstance(test_data.y, torch.Tensor)
                test_loss = torch.nn.functional.cross_entropy(kernel_preds[0], sampled_data.y[sampled_data.test_mask][:, 0])
                
                kernel_transf_preds = kernel.predict(
                    feats_train, weight_list, sampled_data.y[sampled_data.train_mask], feats_all_test, all_adjoint_matrix
                )
                
                assert isinstance(all_data.y, torch.Tensor)
                all_test_loss = torch.nn.functional.cross_entropy(kernel_transf_preds[0], all_data.y[:, 0])
                
                kernel_results[rlz, sample_ind, model_ind] = test_loss
                kernel_transf_results[rlz, sample_ind, model_ind] = all_test_loss

    print('GNN results:', gnn_results)
    print('GNN transf results:', gnn_transf_results)
    print('Kernel results:', kernel_results)
    print('Kernel transf results:', kernel_transf_results)
                
if __name__ == '__main__':
    main()