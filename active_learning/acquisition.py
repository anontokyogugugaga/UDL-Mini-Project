import torch
import torch.nn.functional as F
import numpy as np
from utils.mc import mc_forward

## --------------------------------------------------------------------------- ##
## Acquisition functions used in the paper

@torch.no_grad()
def entropy(model, x, T=20, MC = True):
    """
    Predictive entropy acquisition

    Args:
        model: trained model
        x: unlabeled batch [B, ...]
        MC: whether use MC dropout to obtain scores
        T: MC samples

    Returns:
        scores: [B]
    """
    probs = mc_forward(model, x, T, MC)   # [T, B, C]  (if MC False, gets T same copies of shape [B,C] probs)
    mean_probs = probs.mean(dim=0)    # [B, C]
    entropy = -(mean_probs * mean_probs.log()).sum(dim=-1)   # [B]
    return entropy



@torch.no_grad()
def BALD(model, x, T=20, MC = True):
    """
    BALD acquisition: Mutual information between y and θ

    Args:
        model: trained model
        x: unlabeled batch [B, ...]
        MC: whether to use MC dropout to obtain scores
        T: MC samples

    Returns:
        scores: [B]
    """
    probs = mc_forward(model, x, T, MC)    # [T, B, C]

    # Predictive entropy
    mean_probs = probs.mean(dim=0)     # [B, C]
    H_pred = -(mean_probs * mean_probs.log()).sum(dim=-1)   # [B]

    # Expected entropy
    H_exp = (-(probs * probs.log()).sum(dim=-1)).mean(dim=0)   # [B]
    
    return H_pred - H_exp   


@torch.no_grad()
def variation_ratio(model, x, T=20, MC = True):
    """
    Variation ratio acquisition

    Args:
        model: trained model
        x: unlabeled batch [B, ...]
        MC: whether to use MC dropout to obtain scores
        T: MC samples

    Returns:
        scores: [B]
    """
    probs = mc_forward(model, x, T, MC)     # [T, B, C]
    mean_probs = probs.mean(dim=0)      # [B, C]
    max_prob, _ = mean_probs.max(dim=-1) # [B]
    variation_ratio = 1.0 - max_prob   # [B]
    return variation_ratio


@torch.no_grad()
def mean_std(model, x, T=20, MC = True):
    """
    Mean and standard deviation acquisition

    Args:
        model: trained model
        x: unlabeled batch [B, ...]
        MC: whether to use MC dropout to obtain scores
        T: MC samples

    Returns:
        scores: [B]
    """
    probs = mc_forward(model, x, T, MC)     # [T, B, C]
    std_probs = probs.std(dim=0, unbiased=False)        # [B, C]
    mean_std = std_probs.mean(dim=-1)   # [B]
    return mean_std


@torch.no_grad()
def random_acquire(model, x, T=20):
    return torch.rand(x.shape[0])


## --------------------------------------------------------------------------- ##
## Acquisition functions based on predictive covariance matrix by regression inference on the last layer

@torch.no_grad()
def pred_cov_analytic(model, 
                        x, 
                        train_features, 
                        train_labels, 
                        s_sq = 1,
                        device="cuda"):
    """
    Predictive covariance acquisition using Analytic Inference (Lemma 1)
    Under prior W ~ MN(0, s2 I_K, Σ_y), compute the analytic form of W posterior, 
    then computes predictive covariance for pool data points.

    Σ_y is via MLE estimation from training data Y=train_labels, Phi=train_features. (Lemma 0)

    Args:
        model: Trained CNN_baseline model (must have get_feature method).
        x: Unlabeled batch [B, ...] from pool.
        train_features: Feature matrix Phi from training data [N, K].
        train_labels: Regression/one-hot targets Y from training data [N, D].
        s_sq: [float] Prior variance for W ~ MN(0, s_sq I_K, Σ_y).
        device: Device to run computation on.

    Returns:
        scores: Predictive covariance trace scores [B].
    """
    assert hasattr(model, 'get_feature'), \
        '''To carry out predictive covariance estimation using last-layer inference, 
        model must implement a get_feature(x) method to return feature embeddings'''
    model.to(device)
    model.eval()

    pool_features = model.get_feature(x)  # [B, K]
    Phi = train_features.to(device)   # [N, K]
    Y = train_labels.to(device)      # [N, D]
    phi_star = pool_features.to(device)  # [B, K]

    N, K = Phi.shape   # number of training points, feature dim
    D = Y.shape[1]     # number of outputs / classes

    # MLE estimate for Σ_y using Y,Phi (Lemma 0)
    PhiT_Phi = Phi.T @ Phi  # [K, K]
    PhiT_Phi_reg = PhiT_Phi + 1e-6 * torch.eye(K, device=device)  # jitter for invertibility
    PhiT_Phi_inv = torch.inverse(PhiT_Phi_reg)                    # [K, K]
    W_hat = PhiT_Phi_inv @ Phi.T @ Y                              # [K, D]
    Y_pred = Phi @ W_hat              # [N, D]
    residual = Y - Y_pred             # [N, D]
    Sigma_y_hat = (residual.T @ residual) / float(N)  # [D, D]
    trace_Sigma_y = torch.trace(Sigma_y_hat).clamp(min=1e-8)      # scalar tr(Σ_y)

    # Calculate Sigma_hat
    prior_prec = 1.0 / max(s_sq, 1e-6)
    post_matrix = PhiT_Phi + prior_prec * torch.eye(K, device=device)   # [K, K]
    Sigma_hat = torch.inverse(post_matrix)                              # [K, K]

    # Compute predictive_covariance_trace
    phi_Sigma = phi_star @ Sigma_hat          # [B, K]
    quad_form = (phi_Sigma * phi_star).sum(dim=1)  # [B], each = φ*ᵀ Σ_hat φ*
    quad_form = torch.clamp(quad_form, min=0.0)
    pred_cov_trace = trace_Sigma_y * (1.0 + quad_form)   # [B]
    pred_cov_trace = torch.clamp(pred_cov_trace, min=0.0) 

    return pred_cov_trace




def pred_cov_mfvi(model, 
                    x, 
                    train_features, 
                    train_labels, 
                    s_sq = 1, 
                    device="cuda",
                    num_iterations=100, 
                    lr=0.01, ):
    """
    Predictive covariance acquisition using Mean Field Variational Inference (Lemma 2).
    Under prior W ~ MN(0, s² I_K, I_D), approximates posterior of W with mean-field
    q_{M,S}(W) with independent entries W_kd. Maximizes ELBO to find
    optimal M*, S*, then computes predictive covariance for pool data points. 

    Σ_y is via MLE estimation from training data Y=train_labels, Phi=train_features. (Lemma 0)
    
    Args:
        model: Trained CNN_baseline model (must have get_feature method)
        x: Unlabeled batch [B, ...] from pool.
        train_features: Features from training data [N, K] where K is feature dim
        train_labels: One-hot encoded labels from training data [N, D] where D is num_classes
        s_sq: [float] Prior variance s2 for W_{kd} ~ N(0, s2), default 1.
        num_iterations: Number of iterations for ELBO optimization (default 100)
        lr: Learning rate for ELBO optimization (default 0.01)
        device: Device to run computation on
    
    Returns:
        scores: Predictive covariance trace scores [B]
    """
    import torch 
    
    assert hasattr(model, 'get_feature'), \
        '''To carry out predictive covariance estimation using last-layer inference, 
        model must implement a get_feature(x) method to return feature embeddings'''
    model.to(device)
    model.eval()
    
    # Extract features without gradients (model is frozen)
    with torch.no_grad():
        phi_star = model.get_feature(x)  # [B, K]
        Phi = train_features.to(device)  # [N, K]
        Y = train_labels.to(device)  # [N, D]
    
    N, K = Phi.shape
    D = Y.shape[1]
    s_sq = 1
    
    # Estimate Σ_y via MLE (Lemma 0) before ELBO optimization
    with torch.no_grad():
        PhiT_Phi = Phi.T @ Phi  # [K, K]
        PhiT_Phi_reg = PhiT_Phi + 1e-6 * torch.eye(K, device=device)
        PhiT_Phi_inv = torch.inverse(PhiT_Phi_reg)
        W_hat = PhiT_Phi_inv @ Phi.T @ Y  # [K, D]
        Y_pred = Phi @ W_hat  # [N, D]
        residual = Y - Y_pred  # [N, D]
        Sigma_y_hat = (residual.T @ residual) / float(N)  # [D, D]
        Sigma_y_hat = Sigma_y_hat + 1e-6 * torch.eye(D, device=device)  # Regularize for invertibility
        Sigma_y_inv = torch.inverse(Sigma_y_hat)  # [D, D]
    
    Phi = Phi.detach()
    Y = Y.detach()
    Sigma_y_inv = Sigma_y_inv.detach()
    
    # Initialize mean-field parameters M and S
    # M: [K, D] - mean matrix
    # S: [K, D] - std matrix (we optimize log(S) for positivity)
    M = torch.empty(K, D, device=device, requires_grad=True)
    M.data = torch.randn(K, D, device=device) * 0.01
    log_S = torch.empty(K, D, device=device, requires_grad=True)
    log_S.data = torch.randn(K, D, device=device) * 0.01 - 2.0  # Initialize small
    
    # Optimize ELBO according to Lemma 2
    optimizer = torch.optim.Adam([M, log_S], lr=lr)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        S = torch.exp(log_S)  # [K, D]

        # Term 1 trace calculation
        Y_pred = Phi @ M  # [N, D]
        residual = Y - Y_pred  # [N, D]
        term1 = -0.5 * torch.sum((residual @ Sigma_y_inv) * residual)  
        
        # Term 2 trace calculation
        diag_PhiT_Phi = torch.sum(Phi * Phi, dim=0)   # [K], diag(Phi^T Phi)
        S_Sigma_inv = S @ Sigma_y_inv          # [K, D]
        diag_S_Sigma_ST = torch.sum(S_Sigma_inv * S, dim=1)  # [K]
        term2 = -0.5 * torch.sum(diag_S_Sigma_ST * diag_PhiT_Phi)

        # Term 3 sum calculation
        S_sq = S ** 2  # [K, D]
        M_sq = M ** 2  # [K, D]
        term3 = -0.5 * torch.sum(S_sq / s_sq + M_sq / s_sq - 1.0 + torch.log(s_sq / S_sq))
        
        # ELBO (minimize negative ELBO)
        elbo = term1 + term2 + term3
        loss = -elbo 
        
        loss.backward()
        optimizer.step()
    
    # Get optimal parameters (detach from computation graph)
    with torch.no_grad():
        M_star = M.detach()  # [K, D]
        S_star = torch.exp(log_S.detach())  # [K, D]
        
        # Compute predictive covariance for pool data
        # S_star: [K, D], M_star: [K, D], phi_star: [B, K]
        phi_S_star = phi_star @ S_star  # [B, D] 
        pred_cov_epistemic_diag = phi_S_star ** 2  # [B, D] - element [b,d] = (sum_k phi_star[b,k] * S_star[k,d])^2
        trace_Sigma_y = torch.trace(Sigma_y_hat)  # [1]
        pred_cov_trace = trace_Sigma_y + pred_cov_epistemic_diag.sum(dim=1)  # [B]
        pred_cov_trace = torch.clamp(pred_cov_trace, min=0.0)  # Ensure non-negative
    
    return pred_cov_trace




## --------------------------------------------------------------------------- ##
## Further extension use: Acquisition functions based on trained heads


@torch.no_grad()
def var_rat_heads(model, heads, x):
    """
    Variation ratio acquisition using trained heads as posterior samples.
    
    Args:
        model: Trained full CNN_baseline model (use up to penultimate layer as feature extractor)
        heads: List of T trained heads (Head instances for the final layer)
        x: Unlabeled batch [B, ...]
    
    Returns:
        scores: [B] variation ratio scores
    """
    assert hasattr(model, 'get_feature'), \
        'Model must have get_feature method (e.g. should be CNN_baseline)'
    
    model.eval()
    for head in heads:
        head.eval()
    
    device = next(model.parameters()).device
    x = x.to(device)
    
    # Extract features using model
    features = model.get_feature(x)  # [B, K]
    
    # Get predictions from all T heads
    probs_list = []
    for head in heads:
        head = head.to(device)
        logits = head(features)  # [B, C]
        probs = F.softmax(logits, dim=-1)  # [B, C]
        probs_list.append(probs)
    
    # Stack to get [T, B, C]
    probs = torch.stack(probs_list, dim=0)  # [T, B, C]
    
    # Compute variation ratio: 1 - max_prob
    mean_probs = probs.mean(dim=0)  # [B, C]
    max_prob, _ = mean_probs.max(dim=-1)  # [B]
    variation_ratio = 1.0 - max_prob  # [B]
    
    return variation_ratio


@torch.no_grad()
def mean_std_heads(model, heads, x):
    """
    Mean and standard deviation acquisition using trained heads as posterior samples.
    """
    assert hasattr(model, 'get_feature'), \
        'Model must have get_feature method (e.g. should be CNN_baseline)'
    
    model.eval()
    for head in heads:
        head.eval()
    
    device = next(model.parameters()).device
    x = x.to(device)
    
    # Extract features using model
    features = model.get_feature(x)  # [B, K]
    
    # Get predictions from all T heads
    probs_list = []
    for head in heads:
        head = head.to(device)
        logits = head(features)  # [B, C]
        probs = F.softmax(logits, dim=-1)  # [B, C]
        probs_list.append(probs)
    probs = torch.stack(probs_list, dim=0)  # [T, B, C]
    
    # Compute standard deviation
    std_probs = probs.std(dim=0, unbiased=False)  # [B, C]
    mean_std = std_probs.mean(dim=-1)  # [B]
    
    return mean_std







acquisition_dict = {'entropy': entropy,
                    'BALD': BALD,
                    'variation_ratio': variation_ratio,
                    'mean_std': mean_std,
                    'pred_cov_analytic': pred_cov_analytic,
                    'pred_cov_mfvi': pred_cov_mfvi,
                    'var_rat_heads': var_rat_heads,
                    'mean_std_heads': mean_std_heads
                    }









