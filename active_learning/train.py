import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from active_learning.evaluate import evaluate, evaluate_rmse


def train_model(
    model_class,
    dataset,
    train_indices,
    weight_decay,
    batch_size = 64,
    num_epochs = 50,  # train_model will be called with very few datapoints, set num_epochs to be enough.
    criterion = None,
    device="cuda"
):
    """
    Args:
        model_class: Model class to instantiate (e.g., CNN).
        dataset: The full dataset.
        train_indices: Indices for the training subset.
        weight_decay: Weight decay parameter for optimizer.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        criterion: Loss 'accuracy'(then will use CrossEntropyLoss) or 'rmse'. If None, uses CrossEntropyLoss for classification or MSELoss for regression.
        device: Device to use ("cuda" or "cpu").
    
    Returns:
        Trained model instance.
    """
    model = model_class()
    model = model.to(device)
    
    if criterion == 'accuracy':
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion == 'rmse':
        criterion = torch.nn.MSELoss()
    # Auto-detect loss function if not provided
    elif criterion is None:
        # Check if model is CNN_baseline (regression) or CNN (classification)
        if hasattr(model, 'get_feature'):
            # CNN_baseline: use MSE loss for regression
            criterion = torch.nn.MSELoss()
        else:
            # CNN: use CrossEntropyLoss for classification
            criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.001,
        weight_decay=weight_decay
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        batch_size=batch_size,
    )
    
    num_classes = 10  # MNIST has 10 classes
    
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)  # [B, num_classes]
            
            # Handle different loss functions
            if isinstance(criterion, torch.nn.MSELoss):
                y_onehot = torch.zeros(y.size(0), num_classes, device=device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
                loss = criterion(logits, y_onehot)
            else:
                # For classification: use class indices directly
                loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
    return model



def tune_weight_decay(
    model_class,
    dataset,
    train_indices,
    val_indices,
    weight_decays,
    criterion='accuracy',
    device='cuda'
):
    """
    Tune the weight decay parameter for the given model_class.

    For each candidate weight decay value in weight_decays, trains a fresh model instance on the training indices,
    evaluates on the validation set, and returns the weight decay achieving the best validation metric.

    Args:
        model_class: Class of the model to instantiate (should be callable with no arguments).
        dataset: The full dataset object.
        train_indices: Indices containing training data.
        val_indices: Indices containing validation data.
        weight_decays: List of weight decay candidate values to search over.
        criterion: Must be 'accuracy' or 'rmse'.
        device: Device to use for training and validation ("cuda" or "cpu", default "cuda").

    Returns:
        best_lambda: weight decay value from weight_decays that achieves the best validation metric.
    """
    best_lambda = None
    if criterion == 'accuracy':
        best_val_metric = -1  # maximize accuracy
    elif criterion == 'rmse':
        best_val_metric = float('inf')  # minimize RMSE
    else:
        raise ValueError(f"criterion must be 'accuracy' or 'rmse', got {criterion}")
    
    for wd in weight_decays:
        model = train_model(
            model_class,
            dataset,
            train_indices,
            wd,
            device=device
        )
        if criterion == 'accuracy':
            val_metric = evaluate(model, dataset, val_indices, mc_dropout=False)  
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_lambda = wd
        elif criterion == 'rmse':
            val_metric = evaluate_rmse(model, dataset, val_indices, device=device)
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_lambda = wd
    return best_lambda


