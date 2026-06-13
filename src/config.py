from pathlib import Path

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

EVENTS_PATH = DATA_DIR / "events.csv"
CATEGORY_TREE_PATH = DATA_DIR / "category_tree.csv"
ITEM_PROPERTIES_PATHS = [
    DATA_DIR / "item_properties_part1.csv",
    DATA_DIR / "item_properties_part2.csv",
]

MLFLOW_EXPERIMENT = "ecommerce_recommendations"
TOP_K = 10

# Веса implicit feedback (фокус — add-to-cart)
EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 5.0,
    "transaction": 5.0,
}

POSITIVE_EVENTS = {"addtocart", "transaction"}
TARGET_EVENT = "addtocart"

# Для VM с ограниченной RAM
MAX_TRAIN_USERS = 80_000
TRAIN_TIME_QUANTILE = 0.85

ALS_PARAMS = {
    "factors": 64,
    "regularization": 0.05,
    "iterations": 15,
    "random_state": RANDOM_STATE,
}
