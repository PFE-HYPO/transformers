import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------------------------
# Utils: metrics
# -------------------------
def confusion_from_probs(y_true, probs, thresh=0.5):
    y_true = y_true.astype(np.int32)
    y_pred = (probs >= thresh).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")

def metrics_from_conf(tp, tn, fp, fn):
    recall = safe_div(tp, tp + fn)          # sensitivity
    precision = safe_div(tp, tp + fp)
    specificity = safe_div(tn, tn + fp)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (not math.isnan(precision) and not math.isnan(recall) and (precision + recall) != 0) else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "accuracy": acc,
        "f1": f1,
    }
