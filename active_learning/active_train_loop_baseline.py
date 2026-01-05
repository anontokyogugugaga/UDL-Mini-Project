# Active Learning with CNN_baseline and Predictive Covariance Acquisition Functions
# This module implements two Bayesian inference-based acquisition functions:
# 1. Mean Field Variational Inference (MFVI) - Lemma 1
# 2. Analytic Inference (Matrix Normal) - Lemma 2

from utils.data_process import split_dataset_indices
from active_learning.train import train_model, tune_weight_decay
from active_learning.evaluate import evaluate_rmse
from active_learning.acquisition import pred_cov_mfvi, pred_cov_analytic, random_acquire

from models.cnn_baseline import CNN_baseline

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np




def acquire_point_indices_baseline(
    model,
    dataset,
    pool_indices,
    train_indices,
    acquisition_fn, 
    K,
    batch_size=64, 
    device="cuda",
    s_sq = None,
    num_iterations=100,
    lr=0.001,
):
    """
    Evaluate the acquisition function on the pool set and return the indices of top-K most informative point.
    Specifically designed for CNN_baseline model with predictive covariance acquisition functions.
    
    Args:
        model: Trained CNN_baseline model.
        dataset: The full dataset object.
        pool_indices: List of indices currently in the pool set to select from.
        train_indices: List of training indices (required for predictive_covariance functions).
        acquisition_fn: Acquisition func name ("pred_cov_mfvi" or "pred_cov_analytic").
        K: Number of top points to acquire (int).
        batch_size: Batch size for evaluation (default 64).
        device: Device to run computation on ("cuda" or "cpu", default "cuda").
        s_sq: Prior variance s2 for predictive_covariance functions (default 1).
        num_iterations: Number of iterations for MFVI optimization (default 100).
        lr: Learning rate for MFVI optimization (default 0.001).
    
    Returns:
        selected_indices: List of K pool indices chosen according to acquisition_fn scores.
    """
    assert acquisition_fn in ['pred_cov_analytic', 'pred_cov_mfvi', 'random'], \
        'For baseline CNN, acquisition func must be one of: pred_cov_analytic, pred_cov_mfvi, random'
    
    model.to(device)
    model.eval()
    
    ## Handle random acquisition separately (doesn't need train_features/train_labels)
    if acquisition_fn == 'random':
        topk_pos = torch.randperm(len(pool_indices))[:K]
        selected_indices = [pool_indices[i] for i in topk_pos.cpu().tolist()]
        return selected_indices


    ## For pred_cov-based acquisition functions, need train_features and train_labels
    acquisition_dict = {'pred_cov_analytic': pred_cov_analytic,
                        'pred_cov_mfvi': pred_cov_mfvi}
    acquisition_fn_func = acquisition_dict[acquisition_fn]

    # Extract training features and labels
    with torch.no_grad():
        train_features_list = []
        train_labels_list = []
        num_classes = 10  # MNIST has 10 classes
        
        for idx in train_indices:
            x, y = dataset[idx]
            x_tensor = x.unsqueeze(0).to(device)  # [1, ...]
            feat = model.get_feature(x_tensor)  # [1, feature_dim]
            train_features_list.append(feat.cpu())
            # One-hot encode label
            label_onehot = torch.zeros(num_classes)
            label_onehot[y] = 1.0
            train_labels_list.append(label_onehot)
        
        train_features = torch.cat(train_features_list, dim=0)  # [N_train, feature_dim]
        train_labels = torch.stack(train_labels_list)  # [N_train, num_classes]
    
    # Prepare extra kwargs based on acquisition function
    if acquisition_fn == 'pred_cov_analytic':
        extra_kwargs = {}
    elif acquisition_fn == 'pred_cov_mfvi':
        extra_kwargs = {'num_iterations': num_iterations, 'lr': lr}
    
    
    # Compute scores for pool data in batches
    scores = []
    for i in range(0, len(pool_indices), batch_size):
        batch_indices = pool_indices[i:i+batch_size]
        x_batch = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
        score_batch = acquisition_fn_func(
            model, x_batch, train_features, train_labels, s_sq, device,
            **extra_kwargs
        )
        scores.append(score_batch.cpu())
    
    with torch.no_grad():
        scores = torch.cat(scores)  # [len(pool_indices)]
        print(f'Next acquisition: Scores min = {scores.min()}, mean = {scores.mean()}, max = {scores.max()} \n')
        _, topk_pos = torch.topk(scores, K)
    
    selected_indices = [pool_indices[i] for i in topk_pos.cpu().tolist()]
    return selected_indices


def active_learning_baseline(
    dataset,
    acquisition_fn,  # "pred_cov_mfvi", "pred_cov_analytic", "random"
    split_size=(20, 100, 10000),  # Initial_training, Validation, Test_set size
    random_split=False,
    model_class=CNN_baseline,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100, 
    K=10,
    s_sq = None,      # Prior variance s2
    num_iterations = 100,  # For MFVI optimization, if using 'pred_cov_mfvi'
    lr = 0.01,      # For MFVI optimization, if using 'pred_cov_mfvi'
    device="cuda"
):
    """
    Run active learning on the dataset using CNN_baseline model and predictive covariance acquisition functions.
    
    This function is specifically designed for CNN_baseline model and uses Bayesian inference-based
    acquisition functions (MFVI or Analytic) instead of MC dropout-based methods.

    Note: The aleatoric covariance matrix Σ_y is always estimated using MLE (Lemma 0).
    
    Args:
        dataset: The full dataset object.
        acquisition_fn: Acquisition function name ("pred_cov_mfvi", "pred_cov_analytic", "random").
        split_size: [3-tuple] Sizes of initial_training, validation, and test_set. All remaining in dataset is pool.
        random_split: whether dataset split is random (default False, uses seed=0).
        weight_decay_candidates: candidates for weight decay to test using validation split (default [0.000001, 0.00001, 0.0001]).
        n_rounds: rounds of acquisition (default 100).
        K: Number of top points to acquire per round (default 10).
        s_sq: Prior variance for Bayesian inference (default None = 2/L, L size of feature layer).
        num_iterations: Number of iterations for MFVI ELBO optimization (default 100).
        lr: Learning rate for MFVI ELBO optimization (default 0.01).
        device: Device to use ("cuda" or "cpu", default "cuda").
        
    
    Returns:
        (Model, history): Tuple of model and RMSE history over test set.
    """
    seed = np.random.randint(0, 10000) if random_split else 0
    splits = split_dataset_indices(dataset, split_size, seed)
    train_indices = splits["train"]
    pool_indices = splits["pool"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    
    history = []
    print(f'Active Learning Baseline starts, using Acquisition function {acquisition_fn}.  \n')
    
    for r in range(n_rounds):
        print(f"Acquisition Round {r}, |train|={len(train_indices)}, |pool|={len(pool_indices)}")
        
        # Step 1: tune weight decay using RMSE
        best_wd = tune_weight_decay(
            CNN_baseline,
            dataset,
            train_indices,
            val_indices,
            weight_decay_candidates,
            criterion='rmse',
            device=device
        )
        print(f'Best weight_decay based on validation RMSE: {best_wd}')
        
        # Step 2: retrain with best decay weight using RMSE loss
        model = train_model(
            CNN_baseline,
            dataset,
            train_indices,
            best_wd,
            num_epochs=50,
            device=device
        )
        
        # Step 3: test performance using RMSE
        rmse = evaluate_rmse(model, dataset, test_indices, device=device)
        print(f'Test RMSE: {rmse:.6f}.')
        history.append(rmse)
        
        # Step 4: acquisition using predictive covariance
        new_points = acquire_point_indices_baseline(
            model,
            dataset,
            pool_indices,
            train_indices,
            acquisition_fn,
            K,
            device=device,
            s_sq=s_sq,
            num_iterations=num_iterations,
            lr=lr,
        )
        
        train_indices = train_indices + new_points
        pool_indices = [i for i in pool_indices if i not in new_points]
    
    return model, history


