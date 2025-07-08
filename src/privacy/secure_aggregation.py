import torch

def mask_update(update, mask):
    """
    Masks a model update with a random mask (for Secure Aggregation).
    Args:
        update: state_dict of model update (from client)
        mask: state_dict of random mask (same shape as update)
    Returns:
        masked_update: state_dict
    """
    masked_update = {}
    for key in update:
        masked_update[key] = update[key] + mask[key]
    return masked_update

def unmask_aggregate(aggregated_update, sum_masks):
    """
    Removes the sum of masks after aggregation (on server).
    Args:
        aggregated_update: state_dict (sum of masked updates)
        sum_masks: state_dict (sum of all masks)
    Returns:
        unmasked_update: state_dict
    """
    unmasked_update = {}
    for key in aggregated_update:
        unmasked_update[key] = aggregated_update[key] - sum_masks[key]
    return unmasked_update

def generate_mask(model):
    """
    Generates a random mask state_dict matching the model's parameters.
    Args:
        model: PyTorch model
    Returns:
        mask: state_dict
    """
    mask = {}
    for key, param in model.state_dict().items():
        mask[key] = torch.randn_like(param)
    return mask
