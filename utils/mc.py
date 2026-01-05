import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def enable_mc_dropout(model):
    """
    Enable dropout layers during test-time
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_forward(model, x, T=20, MC = True):
    """
    Monte Carlo forward passes with/without dropout

    Args:
        model: trained neural network
        x: input tensor [B, ...]
        MC: whether to use dropout (default True)
        T: number of MC samples

    Returns:
        predicted probs: [T, B, 10] (when MC False, this will be just T repetitions of [B, 10])
    """
    model.eval()               # global eval mode
    if MC:
        enable_mc_dropout(model)   

    probs = []
    for _ in range(T):
        logits = model(x)
        probs.append(F.softmax(logits, dim=-1))

    return torch.stack(probs, dim=0)

