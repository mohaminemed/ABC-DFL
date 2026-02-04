import numpy as np
from sklearn.cluster import KMeans
import os
import math
import pandas as pd
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import copy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


def compute_asr(y_true, y_pred, poisoned_indices, source_class=1, target_class=0, threshold=0.5):
    """
    y_true, y_pred: 1D lists/arrays/tensors of labels aligned with dataset indices
    poisoned_indices: list of indices that were injected with the trigger
    ASR = fraction of true-source-class poisoned samples predicted as target_class
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(poisoned_indices) == 0:
        return 0.0, 0, 0  # ASR, n_poisoned_src, n_successes

    # Threshold predictions first
    y_pred_labels = (y_pred > threshold).astype(int)

    # Only consider poisoned samples that are true source_class
    poisoned_indices = np.asarray(poisoned_indices, dtype=int)
    src_mask = (y_true[poisoned_indices] == source_class)
    relevant_idx = poisoned_indices[src_mask]

    if relevant_idx.size == 0:
        return 0.0, 0, 0

    successes = (y_pred_labels[relevant_idx] == target_class).sum()
    n_relevant = relevant_idx.size
    asr = float(successes) / float(n_relevant)
    return asr, int(n_relevant), int(successes)



def evaluate_anomaly_detection(y_true, y_pred_clean, y_pred_poisned, backdoored_test_indices=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred_clean > 0.5),
        "Precision": precision_score(y_true, y_pred_clean > 0.5),
        "Recall": recall_score(y_true, y_pred_clean > 0.5),
        "F1-Score": f1_score(y_true, y_pred_clean > 0.5),
        "AUROC": roc_auc_score(y_true, y_pred_clean),
        "ASR": 0,
        "Confusion Matrix": confusion_matrix(y_true, y_pred_clean > 0.5)
    }

    if backdoored_test_indices is not None and np.size(backdoored_test_indices) > 0:

        asr, n_relevant, n_successes = compute_asr(
            y_true, y_pred_poisned, backdoored_test_indices, source_class=1, target_class=0
        )
        print(f"ASR={asr:.3f} ({n_successes}/{n_relevant} poisoned source-class samples predicted as target)")
        metrics["ASR"] = asr

    return metrics




def evaluate_capacity_estimation(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics


def print_metrics(metrics, title):
    print(f"\n{title}")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")



def evaluate_capacity_estimation(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics

def print_metrics(metrics, title):
    print(f"\n{title}")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
        if key == "ASR":
            print(f"Attack Success Rate (ASR): {value:.4f}")