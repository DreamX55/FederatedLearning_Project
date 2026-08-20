import torch

def benchmark_models(centralized_model, local_models, fl_model, test_loader, device):
    """
    Compare performance of centralized, local, and federated models on test data.
    Args:
        centralized_model: trained centralized model
        local_models: list of locally trained client models
        fl_model: federated global model
        test_loader: DataLoader for test dataset
        device: torch device
    Returns:
        dict with accuracy for each model type
    """
    centralized_model.eval()
    fl_model.eval()
    for lm in local_models:
        lm.eval()

    centralized_preds, centralized_labels = [], []
    fl_preds, fl_labels = [], []
    local_preds, local_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            centralized_output = centralized_model(data)
            centralized_preds.extend(torch.argmax(centralized_output, dim=1).cpu().numpy())
            centralized_labels.extend(labels.cpu().numpy())

            fl_output = fl_model(data)
            fl_preds.extend(torch.argmax(fl_output, dim=1).cpu().numpy())
            fl_labels.extend(labels.cpu().numpy())

            # For local models, average predictions
            local_outputs = []
            for lm in local_models:
                out = lm(data)
                local_outputs.append(out)
            avg_local_output = torch.mean(torch.stack(local_outputs), dim=0)
            local_preds.extend(torch.argmax(avg_local_output, dim=1).cpu().numpy())
            local_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score
    results = {
        'centralized_accuracy': accuracy_score(centralized_labels, centralized_preds),
        'federated_accuracy': accuracy_score(fl_labels, fl_preds),
        'local_accuracy': accuracy_score(local_labels, local_preds)
    }
    return results
