from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def compute_auc(labels: Sequence[float], probs: Sequence[float]) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def compute_acc(labels: Sequence[float], probs: Sequence[float], threshold: float = 0.5) -> float:
    preds = (np.array(probs) >= threshold).astype(int)
    return float(accuracy_score(labels, preds))


def compute_rmse(labels: Sequence[float], probs: Sequence[float]) -> float:
    mse = mean_squared_error(labels, probs)
    return float(np.sqrt(mse))


__all__ = ["compute_auc", "compute_acc", "compute_rmse"]
