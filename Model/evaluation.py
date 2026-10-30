"""Account-level evaluation; test labels never enter threshold selection."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def checked_arrays(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("Labels and scores must have equal nonzero length")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    return labels.astype(int), scores


def aggregate_accounts(addresses, labels, scores):
    labels, scores = checked_arrays(labels, scores)
    addresses = np.asarray(addresses).reshape(-1)
    if len(addresses) != len(labels):
        raise ValueError("Address/label/score alignment mismatch")
    unique, inverse = np.unique(addresses, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=labels)
    if ((positives != 0) & (positives != counts)).any():
        raise ValueError("Inconsistent labels for windows of the same account")
    return unique, (positives > 0).astype(int), np.bincount(inverse, weights=scores) / counts


def select_f1_threshold(labels, scores):
    labels, scores = checked_arrays(labels, scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation needs both classes for F1 calibration")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    # Highest threshold wins ties, without consulting any test labels.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(labels, scores, threshold, alert_fraction=0.01):
    labels, scores = checked_arrays(labels, scores)
    if not np.isfinite(threshold) or not 0 < alert_fraction <= 1:
        raise ValueError("Expected a finite threshold and alert fraction in (0, 1]")
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    k = max(1, int(np.ceil(len(labels) * alert_fraction)))
    # Stable ties: caller supplies accounts sorted by ID.
    top = np.argsort(-scores, kind="stable")[:k]
    positive_count = int(labels.sum())
    return {
        "accounts": len(labels), "known_phishing": positive_count,
        "threshold": float(threshold), "alert_fraction": alert_fraction, "k": k,
        "average_precision": float(average_precision_score(labels, scores)) if positive_count else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision_at_k": float(labels[top].mean()),
        "recall_at_k": float(labels[top].sum() / positive_count) if positive_count else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positives_per_1000_accounts": float(fp * 1000 / len(labels)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
