from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        out["f1"] = float(f1_score(y_true, y_pred))
    except Exception:
        out["f1"] = float("nan")
    return out


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    thresholds = {
        "default_0_5": 0.5,
    }
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-8, None)
    if len(f1) > 0:
        thresholds["f1_opt"] = float(thr[np.nanargmax(f1)])
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    J = tpr - fpr
    if len(J) > 0:
        thresholds["youden_j"] = float(thr_roc[np.nanargmax(J)])
    mask = rec[:-1] >= 0.8
    if mask.any():
        best_idx = np.nanargmax(prec[:-1][mask])
        thr_cand = thr[mask][best_idx]
        thresholds["prec_at_rec_0_8"] = float(thr_cand)
    return thresholds
