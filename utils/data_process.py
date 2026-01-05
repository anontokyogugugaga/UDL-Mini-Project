import numpy as np
import torch
from utils.mc import mc_forward

def split_dataset_indices(
    dataset,  
    split_size = (20, 100, 10000),
    seed = 0
):
    """
    Randomly split dataset indices into:
    train / pool / val / test
    The initial train indices will be class-balanced.
    
    Args:
        dataset: Dataset object with __getitem__ returning (data, label)
        split_size: 3-tuple, (n_initial_train, n_val, n_test), sizes of initial_training, validation, and test_set. All remaining in dataset is pool.
        seed: Random seed
    
    Returns:
        Dictionary with keys: "train", "pool", "val", "test", and values: lists of indices.
    """
    n_initial_train, n_val, n_test = split_size

    dataset_size = len(dataset)
    rng = np.random.default_rng(seed)
    n_classes = 10  

    class_indices = {label: [] for label in range(n_classes)}
    
    for idx in range(dataset_size):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)
    
    for label in range(n_classes):
        indices = np.array(class_indices[label])
        rng.shuffle(indices)
        class_indices[label] = indices.tolist()
    
    n_per_class = n_initial_train // n_classes
    remainder = n_initial_train % n_classes
    
    train_indices = []
    for label in range(n_classes):
        n_samples = n_per_class + (1 if label < remainder else 0)
        train_indices.extend(class_indices[label][:n_samples])
        class_indices[label] = class_indices[label][n_samples:]
    
    remaining_indices = []
    for label in range(n_classes):
        remaining_indices.extend(class_indices[label])
    
    remaining_indices = np.array(remaining_indices)
    rng.shuffle(remaining_indices)
    
    test_indices = list(remaining_indices[:n_test])
    val_indices = list(remaining_indices[n_test : n_test + n_val])
    pool_indices = list(remaining_indices[n_test + n_val :])
    
    return {
        "train": train_indices,
        "pool": pool_indices,
        "val": val_indices,
        "test": test_indices,
    }