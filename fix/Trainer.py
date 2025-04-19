import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN import GNN

import copy
from tqdm import trange
from typing import Iterator

def get_optimizer(args: dict, params: Iterator[torch.nn.Parameter]) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    
    weight_decay = args['weight_decay']
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args['opt'] == 'adam':
        optimizer = torch.optim.Adam(filter_fn, lr=args['lr'], weight_decay=weight_decay)
    elif args['opt'] == 'sgd':
        optimizer = torch.optim.SGD(filter_fn, lr=args['lr'], momentum=0.95, weight_decay=weight_decay)
    elif args['opt'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter_fn, lr=args['lr'], weight_decay=weight_decay)
    elif args['opt'] == 'adagrad':
        optimizer = torch.optim.Adagrad(filter_fn, lr=args['lr'], weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args['opt']}")
    if args['scheduler'] == 'none':
        return optimizer, None
    elif args['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['opt_decay_step'], gamma=args['opt_decay_rate'])
    elif args['scheduler'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['opt_restart'])
    else:
        raise ValueError(f"Unsupported optimizer scheduler: {args['opt_scheduler']}")
    return optimizer, scheduler

def train(
    loader_train: DataLoader, 
    loader_val: DataLoader, 
    model: GNN, 
    loss_fn, 
    args: dict, 
    logistic: bool
) -> tuple[list[float], list[float], GNN, float]:
    
    optimizer, scheduler = get_optimizer(args, model.parameters())
    losses = []
    val_losses = []
    best_loss = float('inf')
    best_model = None
    
    for epoch in trange(args['epochs'], desc="Training", unit="Epochs"):
        
        total_loss = 0.0
        model.train()
        
        for batch in loader_train:
            
            optimizer.zero_grad()
            
            pred = model(batch)
            label = batch.y
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs 
        
        losses.append(total_loss)
        
        if epoch % 10 == 0:
            
            val_loss = test(loader_val, model, logistic=logistic)
            val_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)   # SUS
    
    if best_model is None:
        best_model = model
        
    return val_losses, losses, best_model, best_loss


def test(
    loader_test: DataLoader,
    model: GNN,
    logistic: bool
) -> float:
    
    model.eval()
    total_loss = torch.tensor(0.0)
    num_data = 0
    for data in loader_test:
        data: Data
        with torch.no_grad():
            pred = model(data)
            label = data.y
        
        assert isinstance(label, torch.Tensor)
        
        if logistic:
            total_loss += torch.nn.functional.cross_entropy(pred, label).cpu()
        else:
            total_loss += torch.nn.functional.mse_loss(pred.view(-1, 1), label.view(-1, 1), reduction='sum').cpu()
        
        num_data += data.num_graphs
    
    return total_loss.item() / num_data if num_data > 0 else 0.0
    