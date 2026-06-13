from __future__ import annotations

import numpy as np


def recall_at_k(y_true: list[set[int]], y_pred: list[list[int]], k: int = 10) -> float:
    scores = []
    for true_items, pred_items in zip(y_true, y_pred):
        if not true_items:
            continue
        hits = len(true_items.intersection(pred_items[:k]))
        scores.append(hits / len(true_items))
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(y_true: list[set[int]], y_pred: list[list[int]], k: int = 10) -> float:
    scores = []
    for true_items, pred_items in zip(y_true, y_pred):
        if not true_items:
            continue
        hits = len(true_items.intersection(pred_items[:k]))
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(y_true: list[set[int]], y_pred: list[list[int]], k: int = 10) -> float:
    aps = []
    for true_items, pred_items in zip(y_true, y_pred):
        if not true_items:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, item in enumerate(pred_items[:k], start=1):
            if item in true_items:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / min(len(true_items), k))
    return float(np.mean(aps)) if aps else 0.0


def hit_rate_at_k(y_true: list[set[int]], y_pred: list[list[int]], k: int = 10) -> float:
    hits = []
    for true_items, pred_items in zip(y_true, y_pred):
        if not true_items:
            continue
        hits.append(int(bool(true_items.intersection(pred_items[:k]))))
    return float(np.mean(hits)) if hits else 0.0


def coverage_at_k(all_preds: list[list[int]], n_items: int, k: int = 10) -> float:
    recommended = set()
    for pred_items in all_preds:
        recommended.update(pred_items[:k])
    return len(recommended) / n_items if n_items else 0.0
