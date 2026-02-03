from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    classes = np.unique(y_true)
    n_classes = classes.size
    average = "binary" if n_classes == 2 else "weighted"
    pos_label = classes[-1] if n_classes == 2 else None

    if n_classes == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            roc_prob = y_prob[:, 1]
        else:
            roc_prob = y_prob
        roc_auc = roc_auc_score(y_true, roc_prob)
    else:
        if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
            raise ValueError(
                "Multiclass ROC AUC requires probability matrix with shape "
                f"(n_samples, n_classes); got {y_prob.shape} for {n_classes} classes."
            )
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "roc_auc": roc_auc,
    }


def classification_curves(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    classes = np.unique(y_true)
    if classes.size != 2:
        raise ValueError("ROC curves are only supported for binary classification.")
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        y_prob = y_prob[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def classification_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
