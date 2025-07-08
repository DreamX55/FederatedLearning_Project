import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

def compute_accuracy(y_true, y_pred):
    """Compute accuracy score."""
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_pred, average='macro'):
    """Compute F1 score (macro by default)."""
    return f1_score(y_true, y_pred, average=average)

def compute_recall(y_true, y_pred, average='macro'):
    """Compute recall score (macro by default)."""
    return recall_score(y_true, y_pred, average=average)

# Example usage:
# acc = compute_accuracy(true_labels, pred_labels)
# f1 = compute_f1(true_labels, pred_labels)
# recall = compute_recall(true_labels, pred_labels)
