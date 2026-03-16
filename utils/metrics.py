import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from typing import Tuple


def compute_miou(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    ious = []
    for c in range(num_classes):
        intersection = ((preds == c) & (targets == c)).sum()
        union = ((preds == c) | (targets == c)).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def compute_roc_auc(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


def compute_per_class_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    results = {}
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"class_{c}"] = {"precision": precision, "recall": recall, "f1": f1}
    return results
