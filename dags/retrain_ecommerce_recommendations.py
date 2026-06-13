"""Airflow DAG: scheduled retraining of e-commerce recommender."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/home/mle-user/mle-ecommerce-recommendations"

default_args = {
    "owner": "mle-user",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retrain_ecommerce_recommendations",
    default_args=default_args,
    description="Load events, preprocess, train ALS model, log to MLflow",
    schedule_interval="0 3 * * 0",  # every Sunday at 03:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recommendations", "ecommerce"],
) as dag:
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
            f"test -f {PROJECT_ROOT}/models/recommender.pkl && "
            f"test -f {PROJECT_ROOT}/artifacts/metrics.json"
        ),
    )

    train_model >> validate_artifacts
