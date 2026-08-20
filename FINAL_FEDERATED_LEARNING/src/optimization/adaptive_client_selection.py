import numpy as np

def select_clients(client_metrics, num_clients, strategy='random'):
    """
    Select clients for the current FL round.
    Args:
        client_metrics: dict {client_id: metric_value}, e.g., loss or accuracy
        num_clients: number of clients to select
        strategy: 'random' or 'performance' (select lowest loss/highest accuracy)
    Returns:
        List of selected client IDs
    """
    client_ids = list(client_metrics.keys())
    if strategy == 'random':
        return list(np.random.choice(client_ids, num_clients, replace=False))
    elif strategy == 'performance':
        # Select clients with lowest loss (or highest accuracy)
        sorted_clients = sorted(client_metrics.items(), key=lambda x: x[1])
        return [cid for cid, _ in sorted_clients[:num_clients]]
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
