from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

from src.config import (
    ALS_PARAMS,
    ARTIFACTS_DIR,
    MAX_TRAIN_USERS,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    RANDOM_STATE,
    ROOT,
    TOP_K,
)
from src.data import (
    build_matrix,
    build_test_ground_truth,
    load_events,
    popularity_ranking,
    save_encoders,
    select_active_users,
    temporal_split,
)
from src.metrics import coverage_at_k, hit_rate_at_k, map_at_k, precision_at_k, recall_at_k


def recommend_popularity(popular_items: list[int], k: int, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    result = []
    for item in popular_items:
        if item not in exclude:
            result.append(item)
        if len(result) >= k:
            break
    return result


def recommend_als(model: AlternatingLeastSquares, user_idx: int, k: int) -> list[int]:
    item_ids, scores = model.recommend(user_idx, model.user_items[user_idx], N=k, filter_already_liked_items=True)
    return [int(x) for x in item_ids.tolist()]


def evaluate_recommender(
    ground_truth: dict[int, set[int]],
    recommender,
    user_encoder: dict[int, int],
    item_classes: list[int],
    k: int,
    pred_are_item_ids: bool = False,
) -> dict[str, float]:
    y_true: list[set[int]] = []
    y_pred: list[list[int]] = []

    for user_id, true_items in ground_truth.items():
        if user_id not in user_encoder:
            continue
        pred_raw = recommender(user_encoder[user_id], k)
        if pred_are_item_ids:
            pred_items = [int(x) for x in pred_raw]
        else:
            pred_items = [item_classes[int(i)] for i in pred_raw]
        y_true.append(true_items)
        y_pred.append(pred_items)

    return {
        f"recall@{k}": recall_at_k(y_true, y_pred, k),
        f"precision@{k}": precision_at_k(y_true, y_pred, k),
        f"map@{k}": map_at_k(y_true, y_pred, k),
        f"hit_rate@{k}": hit_rate_at_k(y_true, y_pred, k),
        f"coverage@{k}": coverage_at_k(y_pred, len(item_classes), k),
        "eval_users": len(y_true),
    }


def mlflow_name(name: str) -> str:
    return name.replace("@", "_at_")


def save_model(
    model: AlternatingLeastSquares,
    encoders_path: Path,
    popular_items: list[int],
    metrics: dict[str, float],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "recommender.pkl", "wb") as file:
        pickle.dump(
            {
                "model": model,
                "encoders_path": str(encoders_path),
                "popular_items": popular_items,
                "top_k": TOP_K,
            },
            file,
        )
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training() -> dict[str, float]:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading events...", flush=True)
    events = load_events()
    train, test = temporal_split(events)
    train = select_active_users(train, MAX_TRAIN_USERS)
    ground_truth = build_test_ground_truth(test)

    print(f"Train rows: {len(train)}, test users with cart: {len(ground_truth)}", flush=True)

    matrix, user_enc, item_enc, _ = build_matrix(train)
    popular_items = popularity_ranking(train)

    encoders_path = ARTIFACTS_DIR / "encoders.json"
    save_encoders(user_enc, item_enc, encoders_path)
    user_map = {int(cls): int(idx) for idx, cls in enumerate(user_enc.classes_)}
    item_classes = [int(x) for x in item_enc.classes_]

    with mlflow.start_run(run_name="als_addtocart"):
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("max_train_users", MAX_TRAIN_USERS)
        mlflow.log_params({f"als_{k}": v for k, v in ALS_PARAMS.items()})

        baseline_metrics = evaluate_recommender(
            ground_truth,
            lambda _uid, k: recommend_popularity(popular_items, k),
            user_map,
            item_classes,
            TOP_K,
            pred_are_item_ids=True,
        )
        for name, value in baseline_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"baseline_{mlflow_name(name)}", float(value))
        print("Baseline:", baseline_metrics, flush=True)

        model = AlternatingLeastSquares(**ALS_PARAMS)
        model.fit(matrix)

        def als_recommender(user_idx: int, k: int) -> list[int]:
            ids, _ = model.recommend(user_idx, matrix[user_idx], N=k, filter_already_liked_items=True)
            return [int(x) for x in ids.tolist()]

        model_metrics = evaluate_recommender(ground_truth, als_recommender, user_map, item_classes, TOP_K)
        for name, value in model_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(mlflow_name(name), float(value))
        print("ALS:", model_metrics, flush=True)

        model.user_items = matrix
        save_model(model, encoders_path, popular_items, model_metrics)
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")

    return model_metrics


if __name__ == "__main__":
    print(run_training(), flush=True)
