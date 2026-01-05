# Head model for last layer training
# Defines a simple linear head and training functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler


class Head(nn.Module):
    """
    Simple linear head (equivalent to fc2 layer in CNN_baseline)
    """
    def __init__(self, in_features=128, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input features [B, in_features]
        Returns:
            logits: [B, out_features]
        """
        return self.linear(x)


def create_head_from_fc2(fc2_layer, cold_start=False):
    """
    Create a new Head initialized with fc2's parameters or random initialization.
    
    Args:
        fc2_layer: nn.Linear layer to copy from (used for architecture, ignored if cold_start=True)
        cold_start: If True, initialize with random weights instead of copying from fc2
    
    Returns:
        Head instance with same architecture and initial weights (from fc2 or random)
    """
    head = Head(fc2_layer.in_features, fc2_layer.out_features)
    if not cold_start:
        # Copy weights from trained fc2 layer
        head.linear.weight.data = fc2_layer.weight.data.clone()
        head.linear.bias.data = fc2_layer.bias.data.clone()
    # If cold_start=True, Head is already randomly initialized by default
    return head


def train_single_head(
    head,
    train_features,
    train_labels,
    weight_decay,
    alpha_0 = 1.0,
    num_iterations=1000,
    batch_size=64,
    device="cuda",
    use_classification=True,
    langevin=False
):
    """
    Train a single head using SGD with Robbins-Monro learning rate schedule.
    Learning rate at iteration t: alpha_t = alpha_0 / (10*t + 1000), satisfying Robbins-Monro condition.
    If langevin=True, adds Gaussian noise with variance 2*alpha_t after each SGD step to simulate Langevin dynamics.
    
    Args:
        head: Head instance to train
        train_features: Training features [N_train, feature_dim]
        train_labels: Training labels - if use_classification=True: class indices [N_train], 
                      else: one-hot [N_train, num_classes]
        weight_decay: Weight decay parameter (same as original model)
        alpha_0: Initial learning rate for Robbins-Monro schedule, default 1.0
        num_iterations: Number of SGD iterations
        batch_size: Batch size
        device: Device to use
        use_classification: If True, use CrossEntropyLoss (classification), else use MSELoss (regression)
        langevin: If True, add Gaussian noise with variance 2*alpha_t after each SGD step (default False)
    
    Returns:
        Trained head
    """
    head.train()
    head = head.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    
    N_train = train_features.shape[0]
    
    # Convert one-hot to class indices if needed for classification
    if use_classification:
        if train_labels.dim() > 1:
            # Convert one-hot to class indices
            train_labels = train_labels.argmax(dim=-1)  # [N_train]
        # Use CrossEntropyLoss for classification (consistent with base_model training)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # Use MSELoss for regression
        criterion = torch.nn.MSELoss()
    
    # SGD optimizer with weight decay
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=alpha_0,  # Will be updated manually
        weight_decay=weight_decay
    )
    
    # Training loop with Robbins-Monro schedule
    for iteration in range(num_iterations):
        # Sample random batch
        batch_indices = torch.randint(0, N_train, (batch_size,), device=device)
        feat_batch = train_features[batch_indices]  # [B, feature_dim]
        label_batch = train_labels[batch_indices]   # [B] for classification or [B, num_classes] for regression
        
        # Update learning rate: alpha_t = alpha_0 / (10*t + 1000)
        current_lr = alpha_0 / (10*iteration + 1000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = head(feat_batch)  # [B, num_classes]
        loss = criterion(logits, label_batch)
        loss.backward()
        optimizer.step()
        
        # Add Langevin noise if enabled
        if langevin:
            noise_variance = 2 * current_lr
            noise_std = np.sqrt(noise_variance)
            with torch.no_grad():
                for param in head.parameters():
                    # Add pointwise independent Gaussian noise
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)
    
    return head


def train_k_heads(
    initial_fc2,
    train_features,
    train_labels,
    weight_decay,
    K,
    alpha_0,
    num_iterations=1000,
    batch_size=64,
    device="cuda",
    use_classification=True,
    cold_start=False,
    langevin=False
):
    """
    Train K heads separately using SGD with Robbins-Monro schedule.
    Each head starts from the optimal parameters of the trained model, or from random initialization if cold_start=True.
    If langevin=True, adds Gaussian noise after each SGD step to simulate Langevin dynamics.
    
    Args:
        initial_fc2: Trained fc2 layer to initialize heads from (used for architecture, ignored if cold_start=True)
        train_features: Training features [N_train, feature_dim]
        train_labels: Training labels - if use_classification=True: class indices [N_train],
                      else: one-hot [N_train, num_classes]
        weight_decay: Weight decay parameter
        K: Number of heads to train (posterior samples)
        alpha_0: Initial learning rate for Robbins-Monro schedule
        num_iterations: Number of SGD iterations per head
        batch_size: Batch size
        device: Device to use
        use_classification: If True, use CrossEntropyLoss (classification), else use MSELoss (regression)
        cold_start: If True, initialize heads with random weights instead of copying from fc2
        langevin: If True, add Gaussian noise with variance alpha_t / 2 after each SGD step (default False)
    
    Returns:
        List of K trained heads
    """
    heads = []
    for k in range(K):
        # Create new head initialized from optimal fc2 parameters or random
        head = create_head_from_fc2(initial_fc2, cold_start=cold_start)
        
        # Train this head
        trained_head = train_single_head(
            head,
            train_features,
            train_labels,
            weight_decay,
            alpha_0,
            num_iterations,
            batch_size,
            device,
            use_classification=use_classification,
            langevin=langevin
        )
        heads.append(trained_head)
        init_type = "random" if cold_start else "optimal"
        langevin_str = " (Langevin)" if langevin else ""
        print(f"Trained head {k+1}/{K} (initialized from {init_type}{langevin_str})")
    
    return heads

