from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of items")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    metric_labels = labels
    if metric_labels is None:
        metric_labels = sorted(set(y_true) | set(y_pred))

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=metric_labels,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_precision": dict(zip(metric_labels, precision.tolist())),
        "per_class_recall": dict(zip(metric_labels, recall.tolist())),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=metric_labels,
        ).tolist(),
        "labels": list(metric_labels),
    }