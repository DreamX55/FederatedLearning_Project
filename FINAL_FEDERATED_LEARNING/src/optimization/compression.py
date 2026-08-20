import torch
import numpy as np


def quantize_model(model, num_bits=8):
    """
    Simulate post-training quantization by reducing parameter precision.
    Args:
        model: PyTorch model
        num_bits: Number of bits for quantization (default: 8)
    Returns:
        model: Quantized model (in-place)
    """
    scale = 2 ** num_bits - 1
    with torch.no_grad():
        for param in model.parameters():
            min_val = param.min()
            max_val = param.max()
            # Normalize, quantize, and dequantize
            param.copy_((param - min_val) / (max_val - min_val + 1e-8))  # Normalize to [0,1]
            param.mul_(scale).round_().div_(scale)
            param.mul_(max_val - min_val + 1e-8).add_(min_val)
    return model

def prune_model(model, pruning_perc=0.2):
    """
    Prune a percentage of smallest magnitude weights in each layer.
    Args:
        model: PyTorch model
        pruning_perc: Fraction of weights to prune (default: 0.2)
    Returns:
        model: Pruned model (in-place)
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                threshold = np.percentile(abs(tensor), pruning_perc * 100)
                mask = np.abs(tensor) > threshold
                param.data = torch.tensor(tensor * mask, dtype=param.dtype)
    return model
