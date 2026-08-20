"""
Comprehensive Evaluation Metrics Module for Human Activity Recognition (HAR).
Implemented in pure Python standard library for 100% reproducibility without external dependencies.
Supports overall accuracy, macro/weighted F1, per-class precision/recall/F1,
integer & row-normalized confusion matrices, and ambulatory class confusion diagnostics.
"""

from typing import List, Dict, Union, Optional

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

def compute_confusion_matrix_raw(y_true: List[int], y_pred: List[int], num_classes: int = 6) -> List[List[int]]:
    """
    Computes a raw integer confusion matrix of size num_classes x num_classes.
    Rows represent True labels, Columns represent Predicted labels.
    """
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm

def compute_confusion_matrix_normalized(cm_raw: List[List[int]]) -> List[List[float]]:
    """
    Row-normalizes a raw confusion matrix by the class support (true instances).
    Returns proportion of true class samples classified into each predicted class.
    """
    num_classes = len(cm_raw)
    cm_norm = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    for r in range(num_classes):
        row_sum = sum(cm_raw[r])
        if row_sum > 0:
            for c in range(num_classes):
                cm_norm[r][c] = round(cm_raw[r][c] / row_sum, 4)
        else:
            for c in range(num_classes):
                cm_norm[r][c] = 0.0
    return cm_norm

def compute_per_class_metrics(
    cm_raw: List[List[int]],
    class_names: Optional[List[str]] = None,
    zero_division: float = 0.0
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Computes per-class TP, FP, FN, TN, Precision, Recall, F1-score, and Support.
    """
    num_classes = len(cm_raw)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    total_samples = sum(sum(row) for row in cm_raw)
    per_class = {}
    
    for c in range(num_classes):
        tp = cm_raw[c][c]
        fn = sum(cm_raw[c][j] for j in range(num_classes) if j != c)
        fp = sum(cm_raw[i][c] for i in range(num_classes) if i != c)
        tn = total_samples - (tp + fn + fp)
        support = tp + fn
        
        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = zero_division
            
        # Recall = TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = zero_division
            
        # F1 = 2 * P * R / (P + R)
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = zero_division
            
        name = class_names[c]
        per_class[name] = {
            "class_index": c,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "support": int(support),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4)
        }
        
    return per_class

def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    num_classes: int = 6
) -> Dict:
    """
    Computes all standard and expanded classification metrics for HAR.
    """
    if hasattr(y_true, "tolist"):
        y_true = y_true.tolist()
    if hasattr(y_pred, "tolist"):
        y_pred = y_pred.tolist()
        
    if class_names is None:
        class_names = ACTIVITY_NAMES
        
    cm_raw = compute_confusion_matrix_raw(y_true, y_pred, num_classes=num_classes)
    cm_norm = compute_confusion_matrix_normalized(cm_raw)
    per_class = compute_per_class_metrics(cm_raw, class_names=class_names)
    
    total_samples = len(y_true)
    correct_samples = sum(cm_raw[i][i] for i in range(num_classes))
    overall_accuracy = (correct_samples / total_samples) if total_samples > 0 else 0.0
    
    precisions = [per_class[name]["precision"] for name in class_names]
    recalls = [per_class[name]["recall"] for name in class_names]
    f1s = [per_class[name]["f1"] for name in class_names]
    supports = [per_class[name]["support"] for name in class_names]
    
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    
    total_support = sum(supports)
    weighted_f1 = (sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0
    weighted_precision = (sum(p * s for p, s in zip(precisions, supports)) / total_support) if total_support > 0 else 0.0
    weighted_recall = (sum(r * s for r, s in zip(recalls, supports)) / total_support) if total_support > 0 else 0.0
    
    # Ambulatory class confusion diagnostics (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)
    ambulatory_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
    amb_indices = [0, 1, 2]
    amb_cm_raw = [[cm_raw[i][j] for j in amb_indices] for i in amb_indices]
    
    amb_metrics = {
        "classes": ambulatory_classes,
        "raw_submatrix": amb_cm_raw,
        "upstairs_misclassified_as_walking": cm_raw[1][0],
        "upstairs_misclassified_as_downstairs": cm_raw[1][2],
        "downstairs_misclassified_as_walking": cm_raw[2][0],
        "downstairs_misclassified_as_upstairs": cm_raw[2][1],
        "walking_misclassified_as_upstairs": cm_raw[0][1],
        "walking_misclassified_as_downstairs": cm_raw[0][2]
    }
    
    return {
        "total_samples": int(total_samples),
        "overall_accuracy": round(float(overall_accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "per_class": per_class,
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_normalized": cm_norm,
        "ambulatory_diagnostics": amb_metrics
    }
