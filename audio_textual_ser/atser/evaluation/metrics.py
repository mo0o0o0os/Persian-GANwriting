"""The paper's evaluation metrics (Comprehensive_Evaluation_Metrics.ipynb and the ``per_class_metrics``
function repeated in every notebook).

Per class (one-vs-rest): Precision, Sensitivity (recall), Specificity, F1, G-Mean, MCC.
Per fold: the macro average of those six, plus UA (balanced accuracy), WA (accuracy) and W-F1.

G-Mean: the author uses ``imblearn.metrics.geometric_mean_score`` on the binary one-vs-rest labels,
which equals sqrt(sensitivity * specificity); we compute it directly so imbalanced-learn is optional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

PER_CLASS = ["Precision", "Sensitivity", "Specificity", "F1", "G-Mean", "MCC"]
FOLD_METRICS = PER_CLASS + ["UA", "WA", "W-F1"]


def per_class_metrics(y_true, y_pred, classes) -> pd.DataFrame:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    sens = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rows = {}
    for c in labels:
        t, p = (y_true == c).astype(int), (y_pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows[classes[c]] = {"Precision": prec[c], "Sensitivity": sens[c], "Specificity": spec, "F1": f1s[c],
                            "G-Mean": float(np.sqrt(sens[c] * spec)), "MCC": matthews_corrcoef(t, p)}
    return pd.DataFrame(rows).T[PER_CLASS]


def fold_metrics(y_true, y_pred, classes) -> dict:
    per_class = per_class_metrics(y_true, y_pred, classes)
    out = per_class.mean(axis=0).to_dict()
    out["UA"] = balanced_accuracy_score(y_true, y_pred)
    out["WA"] = accuracy_score(y_true, y_pred)
    out["W-F1"] = f1_score(y_true, y_pred, average="weighted")
    return out


def normalized_confusion(y_true, y_pred, n_classes) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)), normalize="true")
