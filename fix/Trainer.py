import torch
from torch_geometric.data import Data
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
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
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
        
        model.train()
            
        optimizer.zero_grad()
        
        pred = model(data)
        
        assert isinstance(data.y, torch.Tensor)
        loss = loss_fn(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            
            _, val_loss = test(data, val_mask, model, logistic=logistic)
            val_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)   # SUS
    
    if best_model is None:
        best_model = model
        
    return val_losses, losses, best_model, best_loss


def test(
    data: Data,
    test_mask: torch.Tensor,
    model: GNN,
    logistic: bool
) -> tuple[torch.Tensor, float]:
    
    model.eval()
    
    with torch.no_grad():
        pred = model(data)
        label = data.y
    
    assert isinstance(label, torch.Tensor)
    
    if logistic:
        return pred[test_mask], torch.nn.functional.cross_entropy(pred[test_mask], label[test_mask]).cpu().item()
    else:
        return pred[test_mask], torch.nn.functional.mse_loss(pred.view(-1, 1), label.view(-1, 1), reduction='sum').cpu().item()
    