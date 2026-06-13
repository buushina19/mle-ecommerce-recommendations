from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

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

        encoders = json.loads(Path(payload["encoders_path"]).read_text(encoding="utf-8"))
        self.user_map: dict[int, int] = {int(k): int(v) for k, v in encoders["users"].items()}
        self.item_classes: list[int] = [int(x) for x in encoders["item_classes"]]

    def recommend(self, visitor_id: int, k: int | None = None) -> tuple[list[int], str]:
        k = k or self.top_k
        user_idx = self.user_map.get(visitor_id)
        if user_idx is None:
            return self.popular_items[:k], "popularity_cold_start"

        item_idx, _ = self.model.recommend(
            user_idx,
            self.model.user_items[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        return [self.item_classes[int(i)] for i in item_idx], "als_personal"
