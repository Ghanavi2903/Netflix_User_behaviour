"""Logistic Regression + Decision Tree (with entropy / information gain)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def entropy(y):
    """Shannon entropy in bits."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def information_gain(feature, y):
    """Information gain of a continuous feature using a median split."""
    parent = entropy(y)
    threshold = np.median(feature)
    left_mask = feature <= threshold
    right_mask = ~left_mask
    n = len(y)
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0.0
    child = (left_mask.sum() / n) * entropy(y[left_mask]) + \
            (right_mask.sum() / n) * entropy(y[right_mask])
    return float(parent - child)


def train_models(X_train, X_test, y_train, y_test):
    """Fit LR and DT, return a dict with models and predictions."""
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)

    dt = DecisionTreeClassifier(
        criterion="entropy", max_depth=6, min_samples_leaf=20,
        random_state=42, class_weight="balanced",
    )
    dt.fit(X_train, y_train)

    return {
        "lr": lr,
        "dt": dt,
        "lr_pred": lr.predict(X_test),
        "lr_prob": lr.predict_proba(X_test)[:, 1],
        "dt_pred": dt.predict(X_test),
        "dt_prob": dt.predict_proba(X_test)[:, 1],
    }