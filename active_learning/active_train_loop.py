# Active Learning with CNN and various Acquisition Functions (entropy, mean_std, var_ratios, BALD, random)

# Train the model from scratch, doing n_rounds rounds of acquisition&training,
# each acquiring K new data from pool using acquisition_fn as judge.

from utils.data_process import split_dataset_indices
from active_learning.train import train_model, tune_weight_decay
from active_learning.evaluate import evaluate
from active_learning.acquisition import entropy, BALD, variation_ratio, mean_std, random_acquire

from models.cnn import CNN

import numpy as np
import torch





@torch.no_grad()
def acquire_point_indices(
    model,
    dataset,
    pool_indices,
    acquisition_fn, 
    K,
    T=20,
    MC = True,
    batch_size=256, 
    device="cuda"
):
    """
    Evaluate the acquisition function on the pool set and return the indices of top-K most informative point.

    Args:
        model: Trained CNN model.
        dataset: The full dataset object.
        pool_indices: List of indices currently in the pool set to select from.
        acquisition_fn: Acquisition function name as a string (must be one of entropy / BALD / variation_ratio / mean_std / random).
        K: Number of top points to acquire (int).
        T: Number of Monte Carlo dropout samples (default 20).
        MC: whether to use MC dropout to compute scores for acquisition. (default True)
        batch_size: Batch size for evaluation (default 64).
        device: Device to run computation on ("cuda" or "cpu", default "cuda").

    Returns:
        selected_indices: List of K pool indices chosen according to acquisition_fn scores.
    """
    assert acquisition_fn in ['random', 'entropy', 'BALD', 'variation_ratio', 'mean_std'], \
        'For reproduction, acquistiion function must be one of: random, entropy, BALD, variation_ratio, mean_std'
    acquisition_dict = {'entropy': entropy,
                    'BALD': BALD,
                    'variation_ratio': variation_ratio,
                    'mean_std': mean_std,
                    'random': random_acquire
                    }
    model.to(device)
    model.eval()
    scores = []

    # Obtain top-K indices for highest uncertainty scores
    if acquisition_fn == 'random':
        topk_pos = torch.randperm(len(pool_indices))[:K]
    else:
        # Compute scores
        acquisition_fn = acquisition_dict[acquisition_fn]
        for i in range(0, len(pool_indices), batch_size):
            batch_indices = pool_indices[i:i+batch_size]
            x_batch = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
            score_batch = acquisition_fn(model, x_batch, T = T, MC = MC)  # [batch_size]
            scores.append(score_batch.cpu())
        scores = torch.cat(scores)  # [len(pool_indices)]
        print(f'Next acquisition: Scores min = {scores.min()}, mean = {scores.mean()}, max = {scores.max()}')
        if scores.std() < 1e-7:
            print(f'Warning: All acquisition scores are effectively the same. Scores std = {scores.std()}')
        _, topk_pos = torch.topk(scores, K)
    
    selected_indices = [pool_indices[i] for i in topk_pos.cpu().tolist()]
    print(f'Selected indices: {topk_pos.cpu().tolist()} \n')
    return selected_indices

def active_learning(
    dataset,
    acquisition_fn, # acquisition_fn from acquisition.py
    split_size = (20, 100, 10000), # Initial_training, Validation, Test_set size; all remaining is pool
    random_split = False,
    model_class = CNN,
    weight_decay_candidates = [0.000001, 0.00001, 0.0001],
    n_rounds = 100, 
    K = 10,
    T = 20, 
    MC_acquire = True,
    MC_test = True,
    device = 'cuda',
):
    """
    Run active learning on the dataset, outputs the accuracy history.

    Args:
        dataset: The full dataset object.
        acquisition_fn: Acquisition function name as a string (must be one of entropy / BALD / variation_ratio / mean_std / random).
        split_size: [3-tuple] Sizes of initial_training, validation, and test_set. All remaining in dataset is pool.
        random_split: whether dataset split is random
        model_class: CNN (architecture defined in models/cnn.py).
        weight_decay_candidates:  candidates for wd to test using validation split after each round of acquisition.
        n_rounds: rounds of acquisition.
        K: Number of top points to acquire (int).
        T: Number of Monte Carlo dropout samples (default 20).
        MC_acquire: whether to use MC dropout to obtain scores.
        MC_test: whether to use MC dropout during evaluation of model on test set.

    Returns:
        (Model, history): Tuple of model and accuracy over test set.

    """
    seed = np.random.randint(0, 10000) if random_split else 42
    splits = split_dataset_indices(dataset, split_size, seed)
    train_indices = splits["train"]
    pool_indices = splits["pool"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    history = []
    print(f'Active Learning starts, using Acquisition function {acquisition_fn}.  \n')

    for r in range(n_rounds):
        print(f"Acquisition Round {r}, |train|={len(train_indices)}, |pool|={len(pool_indices)}")

        # Step 1: tune weight decay
        best_wd = tune_weight_decay(
            model_class,
            dataset,
            train_indices,
            val_indices,
            weight_decay_candidates,
            criterion='accuracy',
            device=device,
        )
        print(f'Best weight_decay based on validation set: {best_wd}')

        # Step 2: retrain with best decay weight
        model = train_model(
            model_class,
            dataset,
            train_indices,
            best_wd,
            num_epochs=50,
            device=device,
        )

        # Step 3: test performance
        acc = evaluate(model, dataset, test_indices, device=device,
                            mc_dropout=MC_test, T=T)
        print(f'Test Accuracy: {acc}.')
        history.append(acc)

        # Step 4: acquisition
        new_points = acquire_point_indices(
            model,
            dataset,
            pool_indices,
            acquisition_fn,
            K,
            T=T,
            MC=MC_acquire,
            device=device,
        )

        train_indices = train_indices + new_points
        pool_indices = [i for i in pool_indices if i not in new_points]

    return model, history


