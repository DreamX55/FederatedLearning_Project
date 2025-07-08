import torch

def add_dp_noise(model, noise_multiplier=0.05, clip_norm=1.0):
    """
    Adds Gaussian noise to model parameters for Differential Privacy.
    Args:
        model: PyTorch model (local client)
        noise_multiplier: Multiplier for Gaussian noise
        clip_norm: Maximum L2 norm for parameter clipping
    Returns:
        model: Noisy model (in-place)
    """
    with torch.no_grad():
        # Clip each parameter tensor
        for param in model.parameters():
            norm = torch.norm(param.data)
            if norm > clip_norm:
                param.data.mul_(clip_norm / (norm + 1e-8))
            # Add Gaussian noise
            noise = torch.randn_like(param) * noise_multiplier * clip_norm
            param.add_(noise)
    return model
