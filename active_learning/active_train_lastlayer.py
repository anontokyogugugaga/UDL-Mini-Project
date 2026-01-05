# Active Learning with Last Layer Training and Posterior Sampling
# Train CNN_baseline model, freeze feature extractor, train K heads with SGD (Robbins-Monro),
# and use them as posterior samples for pool set predictions

from utils.data_process import split_dataset_indices
from active_learning.train import train_model, tune_weight_decay
from active_learning.evaluate import evaluate, evaluate_rmse
from active_learning.acquisition import var_rat_heads, mean_std_heads

from models.cnn_baseline import CNN_baseline
from models.head import train_k_heads

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN_baseline_with_softmax(nn.Module):
    """
    Wrapper for CNN_baseline that adds softmax at the end for classification.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        logits = self.base_model(x)
        return F.softmax(logits, dim=-1)
    
    def get_feature(self, x):
        return self.base_model.get_feature(x)
    
    @property
    def fc2(self):
        return self.base_model.fc2


@torch.no_grad()
def acquire_point_indices_lastlayer(
    model,
    heads,
    dataset,
    pool_indices,
    acquisition_fn,
    K,
    batch_size=64,
    device="cuda"
):
    """
    Evaluate the acquisition function (variation ratio) on the pool set and return the indices of top-K most informative points.
    Specifically designed for CNN_baseline model with trained heads as posterior samples.
    
    Args:
        model: Trained CNN_baseline model (used as feature extractor).
        heads: List of T trained heads (posterior samples).
        dataset: The full dataset object.
        pool_indices: List of indices currently in the pool set to select from.
        acquisition_fn: Acquisition function name ("var_rat_heads", "mean_std_heads").
        K: Number of top points to acquire (int).
        batch_size: Batch size for evaluation (default 64).
        device: Device to run computation on ("cuda" or "cpu", default "cuda").
    
    Returns:
        selected_indices: List of K pool indices chosen according to acquisition_fn scores.
    """
    assert acquisition_fn in ['var_rat_heads', 'mean_std_heads'], \
        'For last-layer SGD-based inference active learning, we only implemented acquisition function "var_rat_heads" and "mean_std_heads"'
    
    model.to(device)
    model.eval()
    
    # Compute scores for pool data in batches
    scores = []
    for i in range(0, len(pool_indices), batch_size):
        batch_indices = pool_indices[i:i+batch_size]
        x_batch = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
        
        if acquisition_fn == 'var_rat_heads':
            score_batch = var_rat_heads(model, heads, x_batch)  # [B]
        elif acquisition_fn == 'mean_std_heads':
            score_batch = mean_std_heads(model, heads, x_batch)  # [B]
        scores.append(score_batch.cpu())
    
    with torch.no_grad():
        scores = torch.cat(scores)  # [len(pool_indices)]
        print(f'Next acquisition: Scores min = {scores.min()}, mean = {scores.mean()}, max = {scores.max()} \n')
        _, topk_pos = torch.topk(scores, K)
    
    selected_indices = [pool_indices[i] for i in topk_pos.cpu().tolist()]
    return selected_indices


def active_learning_lastlayer(
    dataset,
    acquisition_fn="var_rat_heads",
    split_size=(20, 100, 10000),  # Initial_training, Validation, Test_set size
    random_split=False,
    model_class=CNN_baseline,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100,
    K=10,
    T=10,  # Number of heads to train (posterior samples)
    alpha_0=0.01,  # Initial learning rate for Robbins-Monro schedule
    num_iterations=1000,  # Number of SGD iterations per head
    batch_size=64,
    device="cuda",
    cold_start=False, 
    langevin=False  
):
    """
    Run active learning on the dataset using CNN_baseline model with last layer training.
    
    This function trains a CNN_baseline model, then trains T heads separately using SGD
    with a specific Robbins-Monro learning rate schedule (alpha_t = alpha_0 / (10*t + 1000)). 
    These heads serve as posterior samples for acquisition function scoring.
    
    Args:
        dataset: The full dataset object.
        acquisition_fn: Acquisition function name (default "var_rat_heads", / "mean_std_heads").
        split_size: [3-tuple] Sizes of initial_training, validation, and test_set. All remaining in dataset is pool.
        random_split: whether dataset split is random (default False, uses seed=0).
        model_class: Model class (default CNN_baseline).
        weight_decay_candidates: candidates for weight decay to test using validation split (default [0.000001, 0.00001, 0.0001]).
        n_rounds: rounds of acquisition (default 100).
        K: Number of top points to acquire per round (default 10).
        T: Number of heads to train as posterior samples (default 10).
        alpha_0: Initial learning rate for Robbins-Monro schedule (default 0.01).
        num_iterations: Number of SGD iterations per head (default 1000).
        batch_size: Batch size (default 64).
        device: Device to use ("cuda" or "cpu", default "cuda").
        cold_start: If True, initialize heads with random weights instead of copying from optimal fc2 parameters (default False).
        langevin: If True, add Gaussian noise with variance alpha_t after each SGD step to simulate Langevin dynamics (default False).
    
    Returns:
        (Model, history): Tuple of model and accuracy history over test set.
    """
    seed = np.random.randint(0, 10000) if random_split else 0
    splits = split_dataset_indices(dataset, split_size, seed)
    train_indices = splits["train"]
    pool_indices = splits["pool"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    
    history = []
    print(f'Active Learning Last Layer starts, using Acquisition function {acquisition_fn}.  \n')
    
    for r in range(n_rounds):
        print(f"Acquisition Round {r}, |train|={len(train_indices)}, |pool|={len(pool_indices)}")
        
        # Step 1: tune weight decay using accuracy (classification)
        best_wd = tune_weight_decay(
            model_class,
            dataset,
            train_indices,
            val_indices,
            weight_decay_candidates,
            criterion='accuracy',
            device=device
        )
        print(f'Best weight_decay based on validation accuracy: {best_wd}')
        
        # Step 2: retrain with best decay weight using CrossEntropyLoss (classification)
        base_model = train_model(
            model_class,
            dataset,
            train_indices,
            best_wd,
            num_epochs=50,
            criterion='accuracy',
            device=device
        )
        # Wrap model to add softmax at the end
        model = CNN_baseline_with_softmax(base_model)
        
        # Extract training features and labels using get_feature
        print("Extracting training features...")
        model.eval()
        train_features_list = []
        train_labels_list = []
        
        with torch.no_grad():
            for idx in train_indices:
                x, y = dataset[idx]
                x_tensor = x.unsqueeze(0).to(device)  # [1, ...]
                feat = model.get_feature(x_tensor)  # [1, feature_dim]
                train_features_list.append(feat.cpu())
                # Store class index (not one-hot) for CrossEntropyLoss
                train_labels_list.append(y)
        
        train_features = torch.cat(train_features_list, dim=0)  # [N_train, feature_dim]
        train_labels = torch.tensor(train_labels_list, dtype=torch.long)  # [N_train] class indices
        
        # Store optimal fc2 parameters (starting point for head training)
        initial_fc2 = base_model.fc2
        
        # Step 3: Train T heads using SGD with Robbins-Monro schedule
        # Use CrossEntropyLoss (classification) to match base_model training
        init_type = "random" if cold_start else "optimal"
        langevin_str = " (Langevin dynamics)" if langevin else ""
        print(f"Training {T} heads with SGD (Robbins-Monro schedule, initialized from {init_type}{langevin_str})...")
        heads = train_k_heads(
            initial_fc2,
            train_features,
            train_labels,
            best_wd,
            T,
            alpha_0,
            num_iterations,
            batch_size,
            device,
            use_classification=True,  # Use CrossEntropyLoss to match base_model
            cold_start=cold_start,
            langevin=langevin
        )
        
        # Step 3.5: test performance using accuracy
        acc = evaluate(model, dataset, test_indices, device=device, mc_dropout=False)
        print(f'Test Accuracy: {acc:.6f}.')
        history.append(acc)
        
        # Step 4: acquisition using trained heads
        new_points = acquire_point_indices_lastlayer(
            model,
            heads,
            dataset,
            pool_indices,
            acquisition_fn,
            K,
            batch_size=batch_size,
            device=device
        )
        
        train_indices = train_indices + new_points
        pool_indices = [i for i in pool_indices if i not in new_points]
    
    return model, history
