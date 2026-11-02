"""Model evaluation helpers."""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    log_loss,
)


def evaluate(model, X_test, y_test, name):
    """Compute all requested metrics + return confusion matrix and curves data."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, prob)
    prec, rec, _ = precision_recall_curve(y_test, prob)

    return {
        "name": name,
        "cm": cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "cross_entropy": float(log_loss(y_test, prob)),
        "roc_fpr": fpr, "roc_tpr": tpr,
        "pr_precision": prec, "pr_recall": rec,
        "pred": pred, "prob": prob,
    }