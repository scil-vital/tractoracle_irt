import torch
import numpy as np

np_to_torch = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
}

torch_to_np = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

def is_np_type(dtype):
    return dtype in np_to_torch

def is_torch_type(dtype):
    return dtype in torch_to_np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("No GPU or MPS device found, using CPU.")
        return torch.device("cpu")
    
def assert_accelerator():
    assert torch.cuda.is_available() or torch.backends.mps.is_available(), "Hardware acceleration is mandatory, but only no device was found."

def get_device_str():
    return str(get_device())

def gradients_norm(module: torch.nn.Module):
    total_norm = 0
    for p in module.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
