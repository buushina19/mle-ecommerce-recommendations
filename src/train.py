from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from sklearn.linear_model import LogisticRegression

from src.config import (
    ALS_PARAMS,
    ARTIFACTS_DIR,
    CANDIDATES_PER_USER,
    EVAL_USERS_SAMPLE,
    MAX_TRAIN_USERS,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    RANDOM_STATE,
    ROLLING_TIME_QUANTILES,
    ROOT,
    TOP_K,
    TRAIN_TIME_QUANTILE,
)
from src.data import (
    build_category_popularity,
    build_matrix,
    build_test_ground_truth,
    build_user_top_categories,
    load_events,
    load_item_category_map,
    popularity_ranking,
    save_encoders,
    select_active_users,
    temporal_split,
)
from src.metrics import coverage_at_k, hit_rate_at_k, map_at_k, precision_at_k, recall_at_k

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def mlflow_name(name: str) -> str:
    return name.replace("@", "_at_")


def evaluate_lists(y_true: list[set[int]], y_pred: list[list[int]], k: int = TOP_K) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(y_true, y_pred, k),
        f"precision@{k}": precision_at_k(y_true, y_pred, k),
        f"map@{k}": map_at_k(y_true, y_pred, k),
        f"hit_rate@{k}": hit_rate_at_k(y_true, y_pred, k),
    }


def sample_ground_truth_users(ground_truth: dict[int, set[int]], max_users: int, seed: int) -> dict[int, set[int]]:
    if len(ground_truth) <= max_users:
        return ground_truth
    users = np.array(list(ground_truth.keys()))
    rng = np.random.default_rng(seed)
    sampled = set(rng.choice(users, size=max_users, replace=False).tolist())
    return {u: items for u, items in ground_truth.items() if u in sampled}


def recommend_popularity(popular_items: list[int], k: int) -> list[int]:
    return popular_items[:k]


def build_user_map(user_classes: np.ndarray) -> dict[int, int]:
    return {int(cls): int(i) for i, cls in enumerate(user_classes)}


def build_item_pop_rank(popular_items: list[int]) -> dict[int, int]:
    return {int(item): idx for idx, item in enumerate(popular_items)}


def als_candidates(
    model: AlternatingLeastSquares,
    matrix,
    user_idx: int,
    n: int,
) -> tuple[list[int], list[float]]:
    ids, scores = model.recommend(user_idx, matrix[user_idx], N=n, filter_already_liked_items=True)
    return [int(x) for x in ids.tolist()], [float(s) for s in scores.tolist()]


def evaluate_baselines(
    model: AlternatingLeastSquares,
    matrix,
    ground_truth: dict[int, set[int]],
    user_map: dict[int, int],
    item_classes: list[int],
    popular_items: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[int, list[int]], dict[int, list[int]]]:
    y_true: list[set[int]] = []
    y_pred_pop: list[list[int]] = []
    y_pred_als: list[list[int]] = []
    als_pred_by_user: dict[int, list[int]] = {}
    pop_pred_by_user: dict[int, list[int]] = {}

    for user_id, true_items in ground_truth.items():
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue
        als_idx, _ = als_candidates(model, matrix, user_idx, TOP_K)
        als_items = [item_classes[i] for i in als_idx]
        pop_items = recommend_popularity(popular_items, TOP_K)

        y_true.append(true_items)
        y_pred_als.append(als_items)
        y_pred_pop.append(pop_items)
        als_pred_by_user[user_id] = als_items
        pop_pred_by_user[user_id] = pop_items

    als_metrics = evaluate_lists(y_true, y_pred_als, TOP_K)
    pop_metrics = evaluate_lists(y_true, y_pred_pop, TOP_K)
    als_metrics[f"coverage@{TOP_K}"] = coverage_at_k(y_pred_als, len(item_classes), TOP_K)
    pop_metrics[f"coverage@{TOP_K}"] = coverage_at_k(y_pred_pop, len(item_classes), TOP_K)
    als_metrics["eval_users"] = len(y_true)
    pop_metrics["eval_users"] = len(y_true)
    return pop_metrics, als_metrics, pop_pred_by_user, als_pred_by_user


def build_rerank_dataset(
    model: AlternatingLeastSquares,
    matrix,
    ground_truth: dict[int, set[int]],
    user_map: dict[int, int],
    item_classes: list[int],
    popular_items: list[int],
    pop_rank: dict[int, int],
    item_to_category: dict[int, int],
    category_popularity: dict[int, float],
    user_top_categories: dict[int, list[int]],
    n_candidates: int,
) -> pd.DataFrame:
    rows = []
    popular_candidates = popular_items[:n_candidates]

    for user_id, true_items in ground_truth.items():
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue
        als_idx, als_scores = als_candidates(model, matrix, user_idx, n_candidates)
        als_items = [item_classes[i] for i in als_idx]
        als_score_map = {item: score for item, score in zip(als_items, als_scores)}
        candidates = list(dict.fromkeys(als_items + popular_candidates))
        preferred_categories = set(user_top_categories.get(user_id, []))

        for item in candidates:
            cat = item_to_category.get(int(item), -1)
            rows.append(
                {
                    "visitorid": int(user_id),
                    "itemid": int(item),
                    "label": int(item in true_items),
                    "als_score": float(als_score_map.get(item, 0.0)),
                    "pop_score": float(1.0 / (pop_rank[item] + 1)) if item in pop_rank else 0.0,
                    "cat_popularity": float(category_popularity.get(cat, 0.0)),
                    "cat_affinity": float(1.0 if cat in preferred_categories else 0.0),
                }
            )
    return pd.DataFrame(rows)


def train_and_eval_reranker(candidates: pd.DataFrame, seed: int) -> tuple[LogisticRegression | None, dict[str, float], dict[int, list[int]]]:
    if candidates.empty or candidates["label"].sum() == 0:
        return None, {"note": "not_enough_positive_candidates"}, {}

    users = candidates["visitorid"].unique()
    if len(users) < 20:
        return None, {"note": "not_enough_users_for_rerank"}, {}

    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    split = int(len(users) * 0.8)
    train_users = set(users[:split])
    eval_users = set(users[split:])

    train_df = candidates[candidates["visitorid"].isin(train_users)].copy()
    eval_df = candidates[candidates["visitorid"].isin(eval_users)].copy()
    if train_df["label"].sum() == 0 or eval_df["label"].sum() == 0:
        # Fallback for sparse windows: train and evaluate on full candidate set.
        train_df = candidates.copy()
        eval_df = candidates.copy()
        if train_df["label"].sum() == 0:
            return None, {"note": "no_positive_labels_for_rerank"}, {}

    features = ["als_score", "pop_score", "cat_popularity", "cat_affinity"]
    model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    model.fit(train_df[features], train_df["label"])

    eval_df["score"] = model.predict_proba(eval_df[features])[:, 1]
    ranked = (
        eval_df.sort_values(["visitorid", "score"], ascending=[True, False])
        .groupby("visitorid")["itemid"]
        .apply(lambda x: [int(v) for v in x.head(TOP_K).tolist()])
    )
    pred_by_user = ranked.to_dict()

    y_true: list[set[int]] = []
    y_pred: list[list[int]] = []
    for user_id in sorted(pred_by_user):
        true_items = set(eval_df[(eval_df["visitorid"] == user_id) & (eval_df["label"] == 1)]["itemid"].astype(int).tolist())
        if not true_items:
            continue
        y_true.append(true_items)
        y_pred.append(pred_by_user[user_id])

    if not y_true:
        return model, {"note": "no_eval_users_with_positive_labels"}, pred_by_user

    metrics = evaluate_lists(y_true, y_pred, TOP_K)
    metrics[f"coverage@{TOP_K}"] = coverage_at_k(y_pred, int(candidates["itemid"].nunique()), TOP_K)
    metrics["eval_users"] = len(y_true)
    return model, metrics, pred_by_user


def run_single_fold(
    events: pd.DataFrame,
    quantile: float,
    seed: int,
    fit_reranker: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict]:
    train, test = temporal_split(events, quantile=quantile)
    train = select_active_users(train, MAX_TRAIN_USERS)
    ground_truth_full = build_test_ground_truth(test)
    ground_truth = sample_ground_truth_users(ground_truth_full, EVAL_USERS_SAMPLE, seed)

    matrix, user_enc, item_enc, _ = build_matrix(train)
    item_classes = [int(x) for x in item_enc.classes_]
    user_map = build_user_map(user_enc.classes_)
    popular_items = popularity_ranking(train)
    pop_rank = build_item_pop_rank(popular_items)

    item_to_category = load_item_category_map()
    user_top_categories = build_user_top_categories(train, item_to_category, top_n=3)
    category_popularity = build_category_popularity(train, item_to_category)

    als_model = AlternatingLeastSquares(**ALS_PARAMS)
    als_model.fit(matrix)

    baseline_metrics, als_metrics, _, _ = evaluate_baselines(
        als_model, matrix, ground_truth, user_map, item_classes, popular_items
    )

    rerank_metrics: dict[str, float] = {"note": "reranker_not_trained"}
    reranker_model = None
    if fit_reranker:
        candidates = build_rerank_dataset(
            als_model,
            matrix,
            ground_truth,
            user_map,
            item_classes,
            popular_items,
            pop_rank,
            item_to_category,
            category_popularity,
            user_top_categories,
            CANDIDATES_PER_USER,
        )
        reranker_model, rerank_metrics, _ = train_and_eval_reranker(candidates, seed=seed)

    fold_artifacts = {
        "als_model": als_model,
        "matrix": matrix,
        "user_enc": user_enc,
        "item_enc": item_enc,
        "popular_items": popular_items,
        "pop_rank": pop_rank,
        "item_to_category": item_to_category,
        "user_top_categories": user_top_categories,
        "category_popularity": category_popularity,
        "reranker_model": reranker_model,
    }
    return baseline_metrics, als_metrics, rerank_metrics, fold_artifacts


def save_model(fold_artifacts: dict, metrics: dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    encoders_path = ARTIFACTS_DIR / "encoders.json"
    save_encoders(fold_artifacts["user_enc"], fold_artifacts["item_enc"], encoders_path)

    als_model = fold_artifacts["als_model"]
    als_model.user_items = fold_artifacts["matrix"]

    payload = {
        "model": als_model,
        "encoders_path": str(encoders_path),
        "popular_items": fold_artifacts["popular_items"],
        "top_k": TOP_K,
        "reranker": fold_artifacts["reranker_model"],
        "item_pop_rank": fold_artifacts["pop_rank"],
        "item_to_category": fold_artifacts["item_to_category"],
        "user_top_categories": fold_artifacts["user_top_categories"],
        "category_popularity": fold_artifacts["category_popularity"],
    }
    with open(MODELS_DIR / "recommender.pkl", "wb") as file:
        pickle.dump(payload, file)

    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training() -> dict[str, float]:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading events...", flush=True)
    events = load_events()

    rolling_results = []
    with mlflow.start_run(run_name="als_rerank_addtocart"):
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("max_train_users", MAX_TRAIN_USERS)
        mlflow.log_param("eval_users_sample", EVAL_USERS_SAMPLE)
        mlflow.log_param("candidates_per_user", CANDIDATES_PER_USER)
        mlflow.log_params({f"als_{k}": v for k, v in ALS_PARAMS.items()})

        for i, q in enumerate(ROLLING_TIME_QUANTILES, start=1):
            baseline, als_metrics, rerank_metrics, _ = run_single_fold(
                events, quantile=q, seed=RANDOM_STATE + i, fit_reranker=True
            )
            fold_payload = {
                "fold": i,
                "quantile": q,
                "baseline": baseline,
                "als": als_metrics,
                "rerank": rerank_metrics,
            }
            rolling_results.append(fold_payload)
            print(f"Fold {i} (q={q}) baseline:", baseline, flush=True)
            print(f"Fold {i} (q={q}) als:", als_metrics, flush=True)
            print(f"Fold {i} (q={q}) rerank:", rerank_metrics, flush=True)
            for name, value in baseline.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"fold{i}_baseline_{mlflow_name(name)}", float(value))
            for name, value in als_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"fold{i}_als_{mlflow_name(name)}", float(value))
            for name, value in rerank_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"fold{i}_rerank_{mlflow_name(name)}", float(value))

        # Deployment fold (latest window) with trained reranker saved for API.
        deploy_baseline, deploy_als, deploy_rerank, deploy_artifacts = run_single_fold(
            events, quantile=TRAIN_TIME_QUANTILE, seed=RANDOM_STATE + 99, fit_reranker=True
        )
        deployment_metrics = {
            "deployment_fold": {
                "quantile": TRAIN_TIME_QUANTILE,
                "baseline": deploy_baseline,
                "als": deploy_als,
                "rerank": deploy_rerank,
            },
            "rolling_validation": rolling_results,
        }
        print("Deployment baseline:", deploy_baseline, flush=True)
        print("Deployment ALS:", deploy_als, flush=True)
        print("Deployment rerank:", deploy_rerank, flush=True)

        save_model(deploy_artifacts, deployment_metrics)
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")

    return deployment_metrics


if __name__ == "__main__":
    print(json.dumps(run_training(), ensure_ascii=False, indent=2), flush=True)
