# dags/loan_default_training_dag.py
"""
Airflow DAG: Loan Default Training Pipeline

Schedule: weekly (configurable)
Trigger: manual or via drift monitoring DAG

Task flow:
  check_data_availability
       ↓
  ingest_new_data
       ↓
  validate_data           ← Great Expectations
       ↓
  engineer_features
       ↓
  train_model             ← XGBoost + MLflow
       ↓
  evaluate_model
       ↓
  promote_model           ← MLflow Model Registry
       ↓
  deploy_model            ← Restart FastAPI container
       ↓
  notify_completion
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DAG defaults
# ─────────────────────────────────────────────

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

dag = DAG(
    dag_id="loan_default_training",
    default_args=default_args,
    description="Loan default XGBoost training pipeline with automated registry promotion",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["loan-default", "training", "mlops"],
    params={
        "trigger_reason": "scheduled",
        "days_lookback": 365,
        "force_retrain": False,
    },
)

# ─────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────


def check_data_availability(**context) -> bool:
    """Verify sufficient labelled data is available before training."""
    from src.data.ingestion import get_data_stats
    from src.utils.config import get_config

    cfg = get_config()
    stats = get_data_stats()
    logger.info("Data stats: %s", json.dumps(stats, default=str))

    min_rows = cfg["data"]["min_training_rows"]
    total_labelled = (stats.get("default_count", 0) or 0) + (stats.get("repaid_count", 0) or 0)

    context["ti"].xcom_push("data_stats", stats)

    if total_labelled < min_rows:
        logger.warning(
            "Insufficient labelled data: %d rows (need %d). Skipping.",
            total_labelled,
            min_rows,
        )
        return False

    logger.info("Data check passed: %d labelled records available.", total_labelled)
    return True


def ingest_new_data(**context) -> dict:
    """Ingest any new CSV files dropped in the ingestion watch folder."""
    import glob

    from src.data.ingestion import ingest_csv

    watch_dir = os.getenv("DATA_INGESTION_DIR", "/data/incoming")
    csv_files = sorted(glob.glob(f"{watch_dir}/*.csv"))

    if not csv_files:
        logger.info("No new CSV files found in %s", watch_dir)
        context["ti"].xcom_push("ingestion_stats", {"files_processed": 0})
        return {"files_processed": 0}

    total_inserted = 0
    for csv_path in csv_files:
        stats = ingest_csv(csv_path)
        total_inserted += stats["inserted"]
        logger.info("Ingested %s → %d rows", csv_path, stats["inserted"])

    result = {"files_processed": len(csv_files), "total_inserted": total_inserted}
    context["ti"].xcom_push("ingestion_stats", result)
    return result


def validate_data(**context) -> dict:
    """Run Great Expectations validation on training dataset."""
    from src.data.ingestion import get_training_data
    from src.data.validation import validate_dataframe

    params = context["params"]
    days_lookback = params.get("days_lookback", 365)

    df = get_training_data(days_lookback=days_lookback)
    batch_id = f"dag_run_{context['run_id']}"

    validation_result = validate_dataframe(df, batch_id=batch_id, raise_on_failure=True)
    logger.info("Validation result: %s", validation_result)

    context["ti"].xcom_push("validation_result", validation_result)
    context["ti"].xcom_push("training_data_size", len(df))
    return validation_result


def engineer_features_task(**context) -> dict:
    """Engineer features and persist to engineered_features table."""
    import pickle

    from src.data.ingestion import get_training_data
    from src.training.features import engineer_features

    params = context["params"]
    days_lookback = params.get("days_lookback", 365)

    df = get_training_data(days_lookback=days_lookback)
    X, y, pipeline = engineer_features(df, fit=True)

    # Persist pipeline for use during serving
    pipeline_path = "/tmp/feature_pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    result = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "pipeline_path": pipeline_path,
        "default_rate": float(y.mean()),
    }
    context["ti"].xcom_push("feature_engineering_result", result)
    logger.info("Feature engineering: %d rows × %d features", len(X), X.shape[1])
    return result


def train_model_task(**context) -> dict:
    """Train XGBoost model and log to MLflow."""
    from src.data.ingestion import get_training_data
    from src.training.trainer import train

    params = context["params"]
    days_lookback = params.get("days_lookback", 365)

    df = get_training_data(days_lookback=days_lookback)

    run_name = f"weekly_retrain_{context['ds']}"
    tags = {
        "dag_run_id": context["run_id"],
        "trigger_reason": params.get("trigger_reason", "scheduled"),
        "airflow_task": context["task"].task_id,
    }

    result = train(df=df, run_name=run_name, tags=tags)
    context["ti"].xcom_push("training_result", result)
    logger.info(
        "Training complete: run_id=%s version=%s AUC=%.4f",
        result["run_id"],
        result["model_version"],
        result["metrics"]["test_auc_roc"],
    )
    return result


def evaluate_model_task(**context) -> dict:
    """Log evaluation metrics and check promotion readiness."""
    training_result = context["ti"].xcom_pull(task_ids="train_model", key="training_result")

    metrics = training_result.get("metrics", {})
    auc = metrics.get("test_auc_roc", 0)
    f1 = metrics.get("test_f1", 0)

    logger.info("Model evaluation: AUC-ROC=%.4f  F1=%.4f", auc, f1)
    context["ti"].xcom_push("evaluation_passed", auc >= 0.70)
    return {"auc_roc": auc, "f1": f1, "evaluation_passed": auc >= 0.70}


def promote_model_task(**context) -> None:
    """
    The promotion to Production was handled inside trainer.py.
    This task records the pipeline run in the audit table.
    """
    import pandas as pd

    from src.utils.config import get_config
    from src.utils.db import get_engine

    cfg = get_config()
    training_result = context["ti"].xcom_pull(task_ids="train_model", key="training_result")
    metrics = training_result.get("metrics", {})

    engine = get_engine()
    row = {
        "dag_id": context["dag"].dag_id,
        "dag_run_id": context["run_id"],
        "task_id": context["task"].task_id,
        "status": "success",
        "training_rows": training_result.get("n_features"),
        "mlflow_run_id": training_result.get("run_id"),
        "model_version": str(training_result.get("model_version")),
        "auc_roc": metrics.get("test_auc_roc"),
        "promoted_to_prod": True,
        "started_at": datetime.utcnow(),
        "finished_at": datetime.utcnow(),
    }
    df = pd.DataFrame([row])
    schema = cfg["database"]["schema"]
    with engine.begin() as conn:
        df.to_sql("pipeline_runs", conn, schema=schema, if_exists="append", index=False)
    logger.info("Pipeline run recorded for model version %s", row["model_version"])


def deploy_model_task(**context) -> None:
    """Signal FastAPI container to hot-reload the production model."""
    import requests as req

    api_url = os.getenv("API_BASE_URL", "http://api:8000")
    try:
        resp = req.post(f"{api_url}/model/reload", timeout=30)
        resp.raise_for_status()
        logger.info("FastAPI model reload triggered: %s", resp.json())
    except Exception as e:
        logger.warning("Model reload signal failed (non-fatal): %s", e)


def notify_completion(**context) -> None:
    """Log completion summary (hook for Slack / email integration)."""
    training_result = context["ti"].xcom_pull(task_ids="train_model", key="training_result")
    metrics = training_result.get("metrics", {}) if training_result else {}
    logger.info(
        "🎉 Training pipeline complete | version=%s | AUC=%.4f | F1=%.4f | run_id=%s",
        training_result.get("model_version", "?"),
        metrics.get("test_auc_roc", 0),
        metrics.get("test_f1", 0),
        training_result.get("run_id", "?"),
    )


# ─────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────

t_check_data = ShortCircuitOperator(
    task_id="check_data_availability",
    python_callable=check_data_availability,
    provide_context=True,
    dag=dag,
)

t_ingest = PythonOperator(
    task_id="ingest_new_data",
    python_callable=ingest_new_data,
    provide_context=True,
    dag=dag,
)

t_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

t_features = PythonOperator(
    task_id="engineer_features",
    python_callable=engineer_features_task,
    provide_context=True,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    provide_context=True,
    execution_timeout=timedelta(hours=3),
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model_task,
    provide_context=True,
    dag=dag,
)

t_promote = PythonOperator(
    task_id="promote_model",
    python_callable=promote_model_task,
    provide_context=True,
    dag=dag,
)

t_deploy = PythonOperator(
    task_id="deploy_model",
    python_callable=deploy_model_task,
    provide_context=True,
    dag=dag,
)

t_notify = PythonOperator(
    task_id="notify_completion",
    python_callable=notify_completion,
    provide_context=True,
    trigger_rule="all_done",
    dag=dag,
)

# ─────────────────────────────────────────────
# Task dependencies
# ─────────────────────────────────────────────

(
    t_check_data
    >> t_ingest
    >> t_validate
    >> t_features
    >> t_train
    >> t_evaluate
    >> t_promote
    >> t_deploy
    >> t_notify
)
