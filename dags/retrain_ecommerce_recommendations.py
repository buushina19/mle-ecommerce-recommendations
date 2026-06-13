"""Airflow DAG: retrain + validate + champion/challenger gating."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/home/mle-user/mle-ecommerce-recommendations")
DATA_PATH = PROJECT_ROOT / "data" / "events.csv"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
PREV_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics_prev.json"
MODEL_PATH = PROJECT_ROOT / "models" / "recommender.pkl"
CHAMP_MODEL_PATH = PROJECT_ROOT / "models" / "champion_recommender.pkl"


def validate_input_data() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    # Cheap sanity check to fail fast on broken ingestion.
    sample = pd.read_csv(DATA_PATH, nrows=1000)
    required_cols = {"timestamp", "visitorid", "event", "itemid"}
    if not required_cols.issubset(sample.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(sample.columns)}")


def backup_previous_metrics() -> None:
    PREV_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if METRICS_PATH.exists():
        shutil.copy2(METRICS_PATH, PREV_METRICS_PATH)


def choose_promotion() -> str:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("metrics.json was not produced by training")
    current = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    current_recall = (
        current.get("deployment_fold", {})
        .get("rerank", {})
        .get("recall@10")
    )
    if current_recall is None:
        raise ValueError("Current recall@10 missing in metrics.json")

    if not PREV_METRICS_PATH.exists():
        return "promote_challenger"

    prev = json.loads(PREV_METRICS_PATH.read_text(encoding="utf-8"))
    prev_recall = (
        prev.get("deployment_fold", {})
        .get("rerank", {})
        .get("recall@10")
    )
    if prev_recall is None:
        return "promote_challenger"

    # Promote challenger only if recall does not degrade by more than 10%.
    return "promote_challenger" if current_recall >= prev_recall * 0.9 else "keep_champion"


def promote_challenger() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing")
    shutil.copy2(MODEL_PATH, CHAMP_MODEL_PATH)


default_args = {
    "owner": "mle-user",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retrain_ecommerce_recommendations",
    default_args=default_args,
    description="Load events, preprocess, train ALS+rereanker, champion/challenger gate",
    schedule_interval="0 3 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recommendations", "ecommerce"],
) as dag:
    validate_data = PythonOperator(
        task_id="validate_input_data",
        python_callable=validate_input_data,
    )

    backup_metrics = PythonOperator(
        task_id="backup_previous_metrics",
        python_callable=backup_previous_metrics,
    )

    train_model = BashOperator(
        task_id="train_recommender",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "source venv/bin/activate && "
            "export PYTHONPATH=$PWD && "
            "python -m src.train"
        ),
    )

    validate_artifacts = BashOperator(
        task_id="validate_artifacts",
        bash_command=(
            f"test -f {MODEL_PATH} && "
            f"test -f {METRICS_PATH}"
        ),
    )

    choose_branch = BranchPythonOperator(
        task_id="choose_champion_or_keep",
        python_callable=choose_promotion,
    )

    promote = PythonOperator(
        task_id="promote_challenger",
        python_callable=promote_challenger,
    )

    keep = EmptyOperator(task_id="keep_champion")
    finish = EmptyOperator(task_id="finish")

    validate_data >> backup_metrics >> train_model >> validate_artifacts >> choose_branch
    choose_branch >> [promote, keep] >> finish
