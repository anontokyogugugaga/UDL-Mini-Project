import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.mc import mc_forward


@torch.no_grad()
def evaluate(
    model,
    dataset,
    indices,
    batch_size: int = 64,
    device: str = "cuda",
    mc_dropout: bool = False,
    T: int = 20,
):
    """
    Evaluate classification accuracy of model on dataset indices.

    Args:
        model: Trained model to evaluate.
        dataset: Dataset object.
        indices: List of indices to evaluate on.
        batch_size: Batch size (default 64).
        device: Device ("cuda" or "cpu", default "cuda").
        mc_dropout: If True, use MC dropout for uncertainty (default False).
        T: Number of MC dropout samples if mc_dropout is True (default 20).

    Returns:
        Accuracy as float in [0, 1].
    """
    model.to(device)
    loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(indices),
        batch_size=batch_size,
    )
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        outs = mc_forward(model, x, T=T, MC=mc_dropout)      # [T, B, C]
        mean_outs = outs.mean(dim=0)         # [B, C]
        preds = mean_outs.argmax(dim=-1)      # [B]

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0



@torch.no_grad()
def evaluate_rmse(
    model,
    dataset,
    indices,
    batch_size=64,
    device="cuda",
):
    """
    Evaluate RMSE error of model on dataset indices.
    Computes RMSE between model logits and one-hot encoded labels.
    
    Args:
        model: Trained model to evaluate.
        dataset: Dataset object.
        indices: List of indices to evaluate on.
        batch_size: Batch size (default 64).
        device: Device ("cuda" or "cpu", default "cuda").
    
    Returns:
        RMSE as float.
    """
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(indices),
        batch_size=batch_size,
    )
    num_classes = 10  
    total_squared_error = 0.0
    total_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_onehot = torch.zeros(y.size(0), num_classes, device=device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0) # [B, 10]
        logits = model(x)  # [B, 10]
        squared_error = ((logits - y_onehot) ** 2).sum()
        total_squared_error += squared_error.item() # [1]
        total_samples += y.size(0)
    
    # RMSE = sqrt(MSE)
    mse = total_squared_error / (total_samples * num_classes) if total_samples > 0 else 0.0
    rmse = np.sqrt(mse)
    
    return rmse