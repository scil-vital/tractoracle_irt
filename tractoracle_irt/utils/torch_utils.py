import torch
import numpy as np

from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

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

global CPU_WARNING_WAS_PRINTED
CPU_WARNING_WAS_PRINTED = False
def get_device():
    global CPU_WARNING_WAS_PRINTED
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        if not CPU_WARNING_WAS_PRINTED:
            CPU_WARNING_WAS_PRINTED = True
            LOGGER.warning("No GPU or MPS device found, using CPU."
                           " Make sure that you have cuda properly installed."
                           " If you're running this with a docker container,"
                           " make sure that you have the nvidia runtime installed"
                           " and that you have specified '--gpus all' when running"
                           " the container.")
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
