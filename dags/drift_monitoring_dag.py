# dags/drift_monitoring_dag.py
"""
Airflow DAG: Drift Monitoring & Automatic Retraining Trigger

Schedule: every 6 hours
Checks for data drift, prediction drift, and performance degradation.
Triggers the training DAG automatically when critical thresholds are breached.

Task flow:
  load_reference_data
       ↓
  load_current_data
       ↓
  run_data_drift
       ↓
  run_prediction_drift
       ↓
  run_performance_check
       ↓
  evaluate_drift_signals
       ↓
  [trigger_retraining]  ← conditional
       ↓
  save_drift_report
"""

from __future__ import annotations

import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=1),
}

dag = DAG(
    dag_id="loan_drift_monitoring",
    default_args=default_args,
    description="Evidently AI drift monitoring with auto-retraining trigger",
    schedule_interval="0 */6 * * *",  # every 6 hours
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["loan-default", "monitoring", "drift"],
)


# ─────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────


def load_reference_data_task(**context) -> dict:
    from src.monitoring.drift_detector import load_reference_data

    df = load_reference_data()
    if df.empty:
        raise ValueError("Empty reference dataset — cannot compute drift")
    context["ti"].xcom_push("reference_shape", df.shape)
    # Serialise to parquet in /tmp for downstream tasks
    path = "/tmp/reference_data.parquet"
    df.to_parquet(path, index=False)
    return {"path": path, "rows": len(df)}


def load_current_data_task(**context) -> dict:
    from src.monitoring.drift_detector import load_current_data
    from src.utils.config import get_config

    cfg = get_config()
    df = load_current_data()

    min_samples = cfg["monitoring"]["min_samples_for_drift"]
    if len(df) < min_samples:
        logger.warning(
            "Insufficient current data (%d rows, need %d). Skipping drift check.",
            len(df),
            min_samples,
        )
        context["ti"].xcom_push("skip_drift", True)
        return {"rows": len(df), "skip": True}

    path = "/tmp/current_data.parquet"
    df.to_parquet(path, index=False)
    context["ti"].xcom_push("skip_drift", False)
    return {"path": path, "rows": len(df)}


def run_data_drift_task(**context) -> dict:
    import pandas as pd

    from src.monitoring.drift_detector import run_data_drift_report
    from src.training.trainer import load_production_model

    skip = context["ti"].xcom_pull(task_ids="load_current_data", key="skip_drift")
    if skip:
        return {"skipped": True}

    _, version = load_production_model()
    ref = pd.read_parquet("/tmp/reference_data.parquet")
    cur = pd.read_parquet("/tmp/current_data.parquet")

    result = run_data_drift_report(ref, cur, model_version=version)
    context["ti"].xcom_push("data_drift_result", result)
    logger.info(
        "Data drift score: %.4f (detected=%s)",
        result.get("overall_drift_score", 0),
        result.get("is_drift_detected"),
    )
    return result


def run_prediction_drift_task(**context) -> dict:
    import pandas as pd

    from src.monitoring.drift_detector import run_prediction_drift_report
    from src.training.trainer import load_production_model

    skip = context["ti"].xcom_pull(task_ids="load_current_data", key="skip_drift")
    if skip:
        return {"skipped": True}

    _, version = load_production_model()
    ref = pd.read_parquet("/tmp/reference_data.parquet")
    cur = pd.read_parquet("/tmp/current_data.parquet")

    # Only run if prediction_proba column exists in both
    if "prediction_proba" not in ref.columns or "prediction_proba" not in cur.columns:
        logger.warning("prediction_proba column missing — skipping prediction drift")
        return {"skipped": True, "reason": "missing_column"}

    result = run_prediction_drift_report(ref, cur, model_version=version)
    context["ti"].xcom_push("pred_drift_result", result)
    return result


def run_performance_check_task(**context) -> dict:
    import pandas as pd

    from src.monitoring.drift_detector import run_performance_report
    from src.training.trainer import load_production_model

    skip = context["ti"].xcom_pull(task_ids="load_current_data", key="skip_drift")
    if skip:
        return {"skipped": True}

    _, version = load_production_model()
    ref = pd.read_parquet("/tmp/reference_data.parquet")
    cur = pd.read_parquet("/tmp/current_data.parquet")

    result = run_performance_report(ref, cur, model_version=version)
    context["ti"].xcom_push("perf_result", result)
    return result


def evaluate_drift_signals_task(**context) -> str:
    """
    Branch task: returns 'trigger_retraining' or 'no_retrain_needed'
    based on combined drift signals.
    """
    from src.monitoring.drift_detector import check_drift_and_trigger

    skip = context["ti"].xcom_pull(task_ids="load_current_data", key="skip_drift")
    if skip:
        logger.info("Skipping drift evaluation (insufficient data)")
        return "no_retrain_needed"

    data_drift = context["ti"].xcom_pull(task_ids="run_data_drift", key="data_drift_result") or {}
    pred_drift = (
        context["ti"].xcom_pull(task_ids="run_prediction_drift", key="pred_drift_result") or {}
    )
    perf = context["ti"].xcom_pull(task_ids="run_performance_check", key="perf_result") or {}

    should_retrain, reason = check_drift_and_trigger(
        data_drift=data_drift,
        pred_drift=pred_drift,
        perf_results=perf,
    )

    context["ti"].xcom_push("should_retrain", should_retrain)
    context["ti"].xcom_push("retrain_reason", reason)

    logger.info("Drift evaluation complete: retrain=%s | %s", should_retrain, reason)
    return "trigger_retraining" if should_retrain else "no_retrain_needed"


def save_drift_report_task(**context) -> None:
    from src.monitoring.drift_detector import save_drift_report
    from src.training.trainer import load_production_model

    _, version = load_production_model()

    should_retrain = (
        context["ti"].xcom_pull(task_ids="evaluate_drift_signals", key="should_retrain") or False
    )
    retrain_reason = (
        context["ti"].xcom_pull(task_ids="evaluate_drift_signals", key="retrain_reason") or ""
    )

    data_drift = context["ti"].xcom_pull(task_ids="run_data_drift", key="data_drift_result") or {}
    pred_drift = (
        context["ti"].xcom_pull(task_ids="run_prediction_drift", key="pred_drift_result") or {}
    )
    perf = context["ti"].xcom_pull(task_ids="run_performance_check", key="perf_result") or {}

    combined = {
        "overall_drift_score": data_drift.get("overall_drift_score", 0),
        "is_drift_detected": should_retrain,
        "threshold_used": 0.20,
        "feature_drift": data_drift.get("feature_drift", {}),
        "prediction_drift": pred_drift,
        "performance_metrics": perf,
        "retrain_reason": retrain_reason,
    }

    save_drift_report(
        report_type="combined",
        model_version=version,
        drift_results=combined,
        retrain_triggered=should_retrain,
        dag_run_id=context["run_id"],
    )
    logger.info("Drift report saved (retrain_triggered=%s)", should_retrain)


# ─────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────

t_ref = PythonOperator(
    task_id="load_reference_data",
    python_callable=load_reference_data_task,
    provide_context=True,
    dag=dag,
)

t_cur = PythonOperator(
    task_id="load_current_data",
    python_callable=load_current_data_task,
    provide_context=True,
    dag=dag,
)

t_data_drift = PythonOperator(
    task_id="run_data_drift",
    python_callable=run_data_drift_task,
    provide_context=True,
    dag=dag,
)

t_pred_drift = PythonOperator(
    task_id="run_prediction_drift",
    python_callable=run_prediction_drift_task,
    provide_context=True,
    dag=dag,
)

t_perf = PythonOperator(
    task_id="run_performance_check",
    python_callable=run_performance_check_task,
    provide_context=True,
    dag=dag,
)

t_evaluate = BranchPythonOperator(
    task_id="evaluate_drift_signals",
    python_callable=evaluate_drift_signals_task,
    provide_context=True,
    dag=dag,
)

t_trigger_retrain = TriggerDagRunOperator(
    task_id="trigger_retraining",
    trigger_dag_id="loan_default_training",
    conf={"trigger_reason": "drift_detected", "triggered_by": "drift_monitor"},
    dag=dag,
)

t_no_retrain = DummyOperator(
    task_id="no_retrain_needed",
    dag=dag,
)

t_save_report = PythonOperator(
    task_id="save_drift_report",
    python_callable=save_drift_report_task,
    provide_context=True,
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)

# ─────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────

[t_ref, t_cur] >> [t_data_drift, t_pred_drift, t_perf]
[t_data_drift, t_pred_drift, t_perf] >> t_evaluate
t_evaluate >> [t_trigger_retrain, t_no_retrain]
[t_trigger_retrain, t_no_retrain] >> t_save_report
