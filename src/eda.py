from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS_DIR, DATA_DIR, TARGET_EVENT
from src.data import cache_item_categories, load_category_tree, load_events


def run_eda() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = ARTIFACTS_DIR / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = load_events()
    summary = {
        "rows": int(len(events)),
        "users": int(events["visitorid"].nunique()),
        "items": int(events["itemid"].nunique()),
        "events_by_type": events["event"].value_counts().to_dict(),
        "addtocart_share": float((events["event"] == TARGET_EVENT).mean()),
    }

    sns.set_theme(style="whitegrid")

    events["event"].value_counts().plot(kind="bar", title="Event distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "event_distribution.png", dpi=120)
    plt.close()

    events.set_index("timestamp").resample("D").size().plot(figsize=(10, 4), title="Events per day")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(fig_dir / "events_daily.png", dpi=120)
    plt.close()

    cart = events[events["event"] == TARGET_EVENT]
    cart.groupby("itemid").size().sort_values(ascending=False).head(20).plot(
        kind="barh", figsize=(8, 6), title="Top-20 items by add-to-cart"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "top_cart_items.png", dpi=120)
    plt.close()

    tree = load_category_tree()
    summary["category_pairs"] = int(len(tree))

    cat_path = cache_item_categories()
    if cat_path.exists():
        item_cats = pd.read_parquet(cat_path)
        summary["items_with_category"] = int(len(item_cats))

    (ARTIFACTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
