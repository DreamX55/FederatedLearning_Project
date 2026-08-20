import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import torch

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_pred, average='weighted'):
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average)

def compute_recall(y_true, y_pred, average='weighted'):
    """
    Compute recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
        
    Returns:
        Recall score
    """
    return recall_score(y_true, y_pred, average=average)

def compute_precision(y_true, y_pred, average='weighted'):
    """
    Compute precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
        
    Returns:
        Precision score
    """
    return precision_score(y_true, y_pred, average=average)

def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)

def compute_all_metrics(y_true, y_pred):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing all metrics
    """
    return {
        'accuracy': compute_accuracy(y_true, y_pred),
        'f1': compute_f1(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred)
    }

def evaluate_model(model, dataloader, device, num_classes=6):
    """
    Evaluate a PyTorch model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run evaluation on
        num_classes: Number of classes
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return compute_all_metrics(all_labels, all_preds)
