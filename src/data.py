from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from src.config import ARTIFACTS_DIR, DATA_DIR, EVENT_WEIGHTS, EVENTS_PATH, ITEM_PROPERTIES_PATHS, TARGET_EVENT, TRAIN_TIME_QUANTILE


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_PATH)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events["weight"] = events["event"].map(EVENT_WEIGHTS).fillna(0)
    return events


def temporal_split(events: pd.DataFrame, quantile: float = TRAIN_TIME_QUANTILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = events["timestamp"].quantile(quantile)
    train = events[events["timestamp"] <= cutoff].copy()
    test = events[events["timestamp"] > cutoff].copy()
    return train, test


def select_active_users(events: pd.DataFrame, max_users: int) -> pd.DataFrame:
    counts = events.groupby("visitorid").size().sort_values(ascending=False)
    keep = set(counts.head(max_users).index)
    return events[events["visitorid"].isin(keep)].copy()


def build_test_ground_truth(test: pd.DataFrame) -> dict[int, set[int]]:
    positive = test[test["event"] == TARGET_EVENT]
    ground_truth: dict[int, set[int]] = {}
    for visitor_id, group in positive.groupby("visitorid"):
        ground_truth[int(visitor_id)] = set(group["itemid"].astype(int).tolist())
    return ground_truth


def build_matrix(
    train: pd.DataFrame,
) -> tuple[sp.csr_matrix, LabelEncoder, LabelEncoder, np.ndarray]:
    agg = (
        train.groupby(["visitorid", "itemid"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "confidence"})
    )

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    user_ids = user_enc.fit_transform(agg["visitorid"])
    item_ids = item_enc.fit_transform(agg["itemid"])
    confidence = agg["confidence"].astype(float).values

    matrix = sp.csr_matrix(
        (confidence, (user_ids, item_ids)),
        shape=(len(user_enc.classes_), len(item_enc.classes_)),
    )
    return matrix, user_enc, item_enc, confidence


def popularity_ranking(train: pd.DataFrame, top_n: int = 500) -> list[int]:
    cart = train[train["event"] == TARGET_EVENT]
    ranking = cart.groupby("itemid").size().sort_values(ascending=False)
    return [int(x) for x in ranking.head(top_n).index.tolist()]


def load_category_tree() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "category_tree.csv")


def cache_item_categories() -> Path:
    out = ARTIFACTS_DIR / "item_categories.parquet"
    if out.exists():
        return out

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in ITEM_PROPERTIES_PATHS:
        for chunk in pd.read_csv(path, chunksize=500_000):
            cat = chunk[chunk["property"] == "categoryid"].copy()
            if not cat.empty:
                parts.append(cat[["itemid", "value"]].rename(columns={"value": "categoryid"}))
    if not parts:
        pd.DataFrame(columns=["itemid", "categoryid"]).to_parquet(out, index=False)
        return out

    categories = pd.concat(parts, ignore_index=True)
    categories["categoryid"] = pd.to_numeric(categories["categoryid"], errors="coerce")
    categories = categories.dropna().groupby("itemid")["categoryid"].last().reset_index()
    categories.to_parquet(out, index=False)
    return out


def save_encoders(user_enc: LabelEncoder, item_enc: LabelEncoder, path: Path) -> None:
    payload = {
        "users": {int(cls): int(idx) for idx, cls in enumerate(user_enc.classes_)},
        "items": {int(cls): int(idx) for idx, cls in enumerate(item_enc.classes_)},
        "user_classes": [int(x) for x in user_enc.classes_],
        "item_classes": [int(x) for x in item_enc.classes_],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_encoders(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_item_category_map() -> dict[int, int]:
    path = cache_item_categories()
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    return {int(i): int(c) for i, c in zip(df["itemid"].tolist(), df["categoryid"].tolist())}


def build_user_top_categories(
    train: pd.DataFrame,
    item_to_category: dict[int, int],
    top_n: int = 3,
) -> dict[int, list[int]]:
    positive = train[train["event"].isin(["addtocart", "transaction"])].copy()
    if positive.empty:
        return {}
    positive["categoryid"] = positive["itemid"].map(item_to_category)
    positive = positive.dropna(subset=["categoryid"])
    if positive.empty:
        return {}

    grouped = (
        positive.groupby(["visitorid", "categoryid"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["visitorid", "cnt"], ascending=[True, False])
    )
    result: dict[int, list[int]] = {}
    for visitor_id, g in grouped.groupby("visitorid"):
        result[int(visitor_id)] = [int(x) for x in g.head(top_n)["categoryid"].tolist()]
    return result


def build_category_popularity(
    train: pd.DataFrame,
    item_to_category: dict[int, int],
) -> dict[int, float]:
    positive = train[train["event"].isin(["addtocart", "transaction"])].copy()
    if positive.empty:
        return {}
    positive["categoryid"] = positive["itemid"].map(item_to_category)
    positive = positive.dropna(subset=["categoryid"])
    if positive.empty:
        return {}
    counts = positive.groupby("categoryid").size()
    total = counts.sum()
    return {int(cat): float(cnt / total) for cat, cnt in counts.items()}
