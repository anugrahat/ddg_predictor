"""Evaluation metrics for ddG regression and ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def evaluate(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)

    if len(preds) < 2:
        pear = 0.0
        spear = 0.0
    else:
        pear = float(pearsonr(preds, labels).statistic)
        spear = float(spearmanr(preds, labels).statistic)

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    mae = float(mean_absolute_error(labels, preds))

    sign_true = (labels > 0).astype(np.int32)
    if len(np.unique(sign_true)) < 2:
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(sign_true, preds))

    per_ppi_pear = []
    per_ppi_spear = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 10:
            continue
        p = preds[idx]
        y = labels[idx]
        if len(p) < 2:
            continue
        per_ppi_pear.append(float(pearsonr(p, y).statistic))
        per_ppi_spear.append(float(spearmanr(p, y).statistic))

    return {
        "PearsonR": pear,
        "SpearmanR": spear,
        "RMSE": rmse,
        "MAE": mae,
        "AUROC": auroc,
        "per_ppi_PearsonR": float(np.mean(per_ppi_pear)) if per_ppi_pear else 0.0,
        "per_ppi_SpearmanR": float(np.mean(per_ppi_spear)) if per_ppi_spear else 0.0,
        "n": int(len(preds)),
    }
