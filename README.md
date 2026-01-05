# Accurate Last-Layer Posterior Inference for Bayesian Active Learning

This repository contains the complete implementation for the paper submission **"Accurate Last-Layer Posterior Inference for Bayesian Active Learning"**.

## Overview

This project implements multiple approaches to Bayesian active learning, including:

- **Monte Carlo Dropout-based methods**: Traditional MC dropout with various acquisition functions (BALD, entropy, variation ratio, mean std)
- **Last-layer Bayesian inference**: Analytical and variational inference methods for posterior estimation
- **Last-layer Langevin-based Adaptive Basis Selective Sampling**: Training multiple classification heads as posterior samples using stochastic gradient descent with Robbins-Monro learning rate schedules.


## Installation and Setup

1. Clone this repository.
2. We recommend using the conda setup. Install requirements and dependencies by loading 
```bash
conda env create -f active_learning.yaml
```
3. The MNIST dataset will be automatically downloaded on the first run of the data loading part of the Demo notebook ```Active_Learning_Demo.ipynb```.



## Basic Usage
This repository supports three active learning pipelines, differing in how uncertainty is estimated during acquisition and inference.

#### 1. MC Dropout-based Active Learning
Bayesian CNN with MC dropout for both acquisition and evaluation.
```python
from active_learning.active_train_loop import active_learning
from models.cnn import CNN

model, history = active_learning(
    dataset=full_dataset,
    acquisition_fn="BALD",  # Options: "BALD", "entropy", "variation_ratio", "mean_std", "random"
    split_size=(20, 100, 10000),  # (train, val, test)
    model_class=CNN,
    weight_decay_candidates=[0.000001, 0.00001, 0.0001],
    n_rounds=100,
    K=10,  # Points acquired per round
    T=20,  # MC dropout samples
    MC_acquire=True,  # Use MC dropout for acquisition
    MC_test=True,    # Use MC dropout for evaluation
    device="cuda"
)
```


#### 2. Predictive Covariance-based Active Learning (Analytic or MFVI inference over last layer parameters)
Classification is reformulated as regression, and uncertainty is computed from the predictive covariance of the last layer.
```python
from active_learning.active_train_loop_baseline import active_learning_baseline
from models.cnn_baseline import CNN_baseline

model, history = active_learning_baseline(
    dataset=full_dataset,
    acquisition_fn="pred_cov_analytic",  # Options: "pred_cov_analytic", "pred_cov_mfvi", "random"
    split_size=(20, 100, 10000),
    model_class=CNN_baseline,
    n_rounds=100,
    K=10,
    s_sq=None,  # Prior variance (default: 2/L)
    num_iterations=100,  # For MFVI optimization
    lr=0.01,  # For MFVI optimization
    device="cuda"
)
```


#### 3. Langevin-based Adaptive Basis Selective Sampling (LAB2S)
Exact posterior sampling over last-layer weights using multiple heads, with optional Langevin dynamics.
```python
from active_learning.active_train_lastlayer import active_learning_lastlayer
from models.cnn_baseline import CNN_baseline

model, history = active_learning_lastlayer(
    dataset=full_dataset,
    acquisition_fn="var_rat_heads",  # Options: "var_rat_heads", "mean_std_heads"
    split_size=(20, 100, 10000),
    model_class=CNN_baseline,
    n_rounds=100,
    K=10,
    T=10,  # Number of heads (posterior samples)
    alpha_0=0.01,  # Initial learning rate for Robbins-Monro schedule
    num_iterations=1000,  # SGD iterations per head
    cold_start=False,  # Initialize from optimized weights (MAP)
    langevin=False,  # Enable Langevin dynamics
    device="cuda"
)
```


## Reproducing Experimental Results

1. We recommend reproducing all experiments by running the notebook cells in `Active_Learning_Demo.ipynb` sequentially. The notebook contains detailed explanations for the experiments.
2. Results will be saved to `Active Learning Accuracy Histories` (CSV format), where several histories already exist for direct plotting purposes if you do not wish to train any model.
3. Visualization helper functions are provided in the notebook for plotting mean $\pm$ std across multiple runs.

The notebook contains experiments labelled with section numbers corresponding to those in the work.
- **Section 3.1**: Comparison of MC dropout-based acquisition functions (BALD, entropy, variation ratio, mean std, random)
- **Section 3.2**: Comparison of Deterministic, Semi-Deterministic, and (fully) Bayesian active learning
- **Section 4**: Predictive covariance inference methods (analytic and MFVI) 
- **Section 5**: Langevin-based Adaptive Basis Selective Sampling method (LAB2S)

