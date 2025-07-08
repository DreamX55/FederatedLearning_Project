def fedavg(state_dicts):
    """
    Federated Averaging: average model parameters from multiple clients.
    Args:
        state_dicts (list): List of model.state_dict() from clients.
    Returns:
        dict: Averaged state_dict.
    """
    import copy
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict:
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict
