"""Seeding, device selection, optimizer and scheduler factories."""
import os
import random

import numpy as np
import torch
import torch.optim as optim

OPTIMIZERS = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]


def set_seed(seed):
    """Seed all RNGs and request deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    """sgd: SGD with `momentum`; nesterov: SGD with Nesterov momentum (momentum forced > 0).
    Weight decay is L2 regularization for every optimizer."""
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum or 0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")
