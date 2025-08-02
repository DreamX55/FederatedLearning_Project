import torch

def add_dp_noise_to_delta(delta, noise_multiplier=0.05, clip_norm=1.0):
    """
    Adds Gaussian noise to model update (delta) for Differential Privacy.
    Args:
        delta: dict of parameter tensors (model update: client_model - global_model)
        noise_multiplier: Multiplier for Gaussian noise
        clip_norm: Maximum L2 norm for clipping the delta
    Returns:
        delta: Noisy and clipped delta (dict)
    """
    total_norm = torch.zeros(1, device=next(iter(delta.values())).device)
    for param in delta.values():
        total_norm += torch.sum(param ** 2)
    total_norm = torch.sqrt(total_norm).item()
    scale = min(1.0, clip_norm / (total_norm + 1e-8))
    noisy_delta = {}
    for key, param in delta.items():
        clipped_param = param * scale
        noise = torch.randn_like(clipped_param) * noise_multiplier * clip_norm
        noisy_delta[key] = clipped_param + noise
    return noisy_delta
