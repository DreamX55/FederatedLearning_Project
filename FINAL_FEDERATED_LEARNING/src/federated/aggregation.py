import torch
import copy

def federated_averaging(client_models):
    """
    Perform federated averaging on client models.
    
    Args:
        client_models: List of client model state dictionaries
        
    Returns:
        Aggregated global model state dictionary
    """
    if not client_models:
        raise ValueError("No client models provided for aggregation")
    
    # Initialize global model with first client's structure
    global_model_state = copy.deepcopy(client_models[0])
    
    # Average all parameters
    for key in global_model_state.keys():
        # Sum all client parameters for this key
        global_model_state[key] = torch.stack([
            client_model[key] for client_model in client_models
        ]).mean(dim=0)
    
    return global_model_state

def weighted_federated_averaging(client_models, client_weights):
    """
    Perform weighted federated averaging on client models.
    
    Args:
        client_models: List of client model state dictionaries
        client_weights: List of weights for each client (e.g., data size)
        
    Returns:
        Weighted aggregated global model state dictionary
    """
    if not client_models or not client_weights:
        raise ValueError("No client models or weights provided")
    
    if len(client_models) != len(client_weights):
        raise ValueError("Number of models and weights must match")
    
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Initialize global model
    global_model_state = copy.deepcopy(client_models[0])
    
    # Weighted average of all parameters
    for key in global_model_state.keys():
        weighted_params = torch.stack([
            client_model[key] * weight 
            for client_model, weight in zip(client_models, normalized_weights)
        ]).sum(dim=0)
        global_model_state[key] = weighted_params
    
    return global_model_state
