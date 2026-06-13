from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
from implicit.als import AlternatingLeastSquares


class RecommenderHandler:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as file:
            payload = pickle.load(file)

        self.model: AlternatingLeastSquares = payload["model"]
        self.popular_items: list[int] = payload["popular_items"]
        self.top_k: int = payload["top_k"]
        self.reranker = payload.get("reranker")
        self.item_pop_rank: dict[int, int] = {int(k): int(v) for k, v in payload.get("item_pop_rank", {}).items()}
        self.item_to_category: dict[int, int] = {int(k): int(v) for k, v in payload.get("item_to_category", {}).items()}
        self.user_top_categories: dict[int, list[int]] = {
            int(k): [int(x) for x in v] for k, v in payload.get("user_top_categories", {}).items()
        }
        self.category_popularity: dict[int, float] = {
            int(k): float(v) for k, v in payload.get("category_popularity", {}).items()
        }

        encoders = json.loads(Path(payload["encoders_path"]).read_text(encoding="utf-8"))
        self.user_map: dict[int, int] = {int(k): int(v) for k, v in encoders["users"].items()}
        self.item_classes: list[int] = [int(x) for x in encoders["item_classes"]]

    def _rerank(self, visitor_id: int, items: list[int], als_scores: list[float], k: int) -> list[int]:
        if self.reranker is None or not items:
            return items[:k]
        preferred_categories = set(self.user_top_categories.get(visitor_id, []))
        features = []
        for item, als_score in zip(items, als_scores):
            cat = self.item_to_category.get(int(item), -1)
            features.append(
                [
                    float(als_score),
                    float(1.0 / (self.item_pop_rank[item] + 1)) if item in self.item_pop_rank else 0.0,
                    float(self.category_popularity.get(cat, 0.0)),
                    float(1.0 if cat in preferred_categories else 0.0),
                ]
            )
        scores = self.reranker.predict_proba(np.array(features))[:, 1]
        ranked = [item for _, item in sorted(zip(scores, items), reverse=True)]
        return ranked[:k]

    def recommend(self, visitor_id: int, k: int | None = None) -> tuple[list[int], str]:
        k = k or self.top_k
        user_idx = self.user_map.get(visitor_id)
        if user_idx is None:
            return self.popular_items[:k], "popularity_cold_start"

        item_idx, scores = self.model.recommend(
            user_idx,
            self.model.user_items[user_idx],
            N=max(k, 50),
            filter_already_liked_items=True,
        )
        items = [self.item_classes[int(i)] for i in item_idx]
        als_scores = [float(x) for x in scores.tolist()]
        ranked = self._rerank(visitor_id, items, als_scores, k)
        return ranked, "als_rerank"
