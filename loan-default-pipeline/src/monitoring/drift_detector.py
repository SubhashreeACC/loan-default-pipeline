# src/monitoring/drift_detector.py
"""
Evidently AI-based drift detection.
Computes data drift, prediction drift, and model performance degradation.
Persists results to PostgreSQL and generates HTML reports.
Triggers retraining DAG when critical thresholds are breached.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sqlalchemy as sa
from evidently import ColumnMapping
from evidently.metric_presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from src.utils.config import get_config
from src.utils.db import get_engine

logger = logging.getLogger(__name__)
cfg = get_config()
mon_cfg = cfg["monitoring"]
SCHEMA = cfg["database"]["schema"]

# ─────────────────────────────────────────────
# Report output directory
# ─────────────────────────────────────────────
REPORT_DIR = Path(os.getenv("REPORT_DIR", "/tmp/evidently_reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Column mapping for Evidently
# ─────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_term_months",
    "goods_credit_ratio",
    "age_years",
    "employment_years",
    "employed_to_age_ratio",
    "ext_source_mean",
    "ext_source_std",
    "family_income_per_capita",
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
]

CATEGORICAL_FEATURES = [
    "gender_m",
    "contract_type_cash",
    "income_type_working",
    "education_higher",
    "housing_type_house",
    "family_status_married",
    "total_region_mismatches",
    "docs_provided_count",
    "ext_source_missing_cnt",
]


def get_column_mapping(include_target: bool = True) -> ColumnMapping:
    cm = ColumnMapping(
        target="target" if include_target else None,
        prediction="prediction_proba",
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    return cm


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────


def load_reference_data(
    days: int | None = None,
    engine=None,
) -> pd.DataFrame:
    """Load reference window data (training-era distribution)."""
    days = days or mon_cfg["reference_window_days"]
    engine = engine or get_engine()
    query = f"""
        SELECT ef.*, pl.prediction_proba, pl.prediction_label
        FROM {SCHEMA}.engineered_features ef
        LEFT JOIN {SCHEMA}.prediction_log pl ON ef.sk_id_curr = pl.sk_id_curr
        WHERE ef.created_at <= NOW() - INTERVAL '{days} days'
          AND ef.created_at >= NOW() - INTERVAL '{days * 2} days'
        ORDER BY ef.created_at
        LIMIT 50000
    """
    with engine.connect() as conn:
        df = pd.read_sql(sa.text(query), conn)
    logger.info("Loaded %d reference records", len(df))
    return df


def load_current_data(
    days: int | None = None,
    engine=None,
) -> pd.DataFrame:
    """Load current production window data."""
    days = days or mon_cfg["production_window_days"]
    engine = engine or get_engine()
    query = f"""
        SELECT ef.*, pl.prediction_proba, pl.prediction_label, pl.actual_label
        FROM {SCHEMA}.engineered_features ef
        JOIN {SCHEMA}.prediction_log pl ON ef.sk_id_curr = pl.sk_id_curr
        WHERE pl.predicted_at >= NOW() - INTERVAL '{days} days'
        ORDER BY pl.predicted_at
    """
    with engine.connect() as conn:
        df = pd.read_sql(sa.text(query), conn)
    logger.info("Loaded %d current production records", len(df))
    return df


# ─────────────────────────────────────────────
# Drift reports
# ─────────────────────────────────────────────


def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_version: str,
    save_html: bool = True,
) -> dict[str, Any]:
    """
    Run Evidently data drift report comparing reference to current window.
    Returns drift results dict and saves HTML report.
    """
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    col_map = get_column_mapping(include_target=False)

    ref_clean = reference[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    cur_clean = current[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    report.run(
        reference_data=ref_clean,
        current_data=cur_clean,
        column_mapping=ColumnMapping(
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        ),
    )

    result = report.as_dict()
    drift_results = _parse_data_drift(result)
    drift_results["model_version"] = model_version

    if save_html:
        html_path = REPORT_DIR / f"data_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(html_path))
        drift_results["report_html_path"] = str(html_path)
        logger.info("Data drift report saved: %s", html_path)

    return drift_results


def run_prediction_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_version: str,
    save_html: bool = True,
) -> dict[str, Any]:
    """Run prediction drift (target drift) report."""
    report = Report(metrics=[TargetDriftPreset()])

    ref_clean = (
        reference[["prediction_proba"]]
        .rename(columns={"prediction_proba": "prediction_proba"})
        .copy()
    )
    cur_clean = current[["prediction_proba"]].copy()

    report.run(
        reference_data=ref_clean,
        current_data=cur_clean,
        column_mapping=ColumnMapping(prediction="prediction_proba"),
    )

    result = report.as_dict()
    drift_results = _parse_prediction_drift(result)
    drift_results["model_version"] = model_version

    if save_html:
        html_path = REPORT_DIR / f"pred_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(html_path))
        drift_results["report_html_path"] = str(html_path)

    return drift_results


def run_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_version: str,
    save_html: bool = True,
) -> dict[str, Any]:
    """Run model performance degradation report (requires actual labels)."""
    # Only use labelled current data
    cur_labelled = current[current["actual_label"].notna()].copy()
    cur_labelled = cur_labelled.rename(columns={"actual_label": "target"})

    if len(cur_labelled) < mon_cfg["min_samples_for_drift"]:
        logger.warning(
            "Insufficient labelled data (%d rows) for performance report. Need %d.",
            len(cur_labelled),
            mon_cfg["min_samples_for_drift"],
        )
        return {"report_type": "performance", "insufficient_data": True}

    report = Report(metrics=[ClassificationPreset()])
    ref_perf = reference[["target", "prediction_proba", "prediction_label"]].dropna().copy()

    report.run(
        reference_data=ref_perf,
        current_data=cur_labelled[["target", "prediction_proba", "prediction_label"]],
        column_mapping=ColumnMapping(
            target="target",
            prediction="prediction_proba",
        ),
    )

    result = report.as_dict()
    perf_results = _parse_performance(result)
    perf_results["model_version"] = model_version

    if save_html:
        html_path = REPORT_DIR / f"performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(html_path))
        perf_results["report_html_path"] = str(html_path)

    return perf_results


# ─────────────────────────────────────────────
# Drift decision logic
# ─────────────────────────────────────────────


def check_drift_and_trigger(
    data_drift: dict,
    pred_drift: dict,
    perf_results: dict,
    dag_trigger_url: str | None = None,
) -> tuple[bool, str]:
    """
    Evaluate all drift signals and decide whether to trigger retraining.

    Returns (should_retrain, reason_string)
    """
    reasons = []

    # Data drift (PSI)
    psi = data_drift.get("overall_drift_score", 0)
    if psi >= mon_cfg["psi_critical"]:
        reasons.append(f"Data drift PSI={psi:.4f} >= critical {mon_cfg['psi_critical']}")

    # Prediction drift (JS divergence)
    js = pred_drift.get("overall_drift_score", 0)
    if js >= mon_cfg["js_divergence_critical"]:
        reasons.append(
            f"Prediction drift JS={js:.4f} >= critical {mon_cfg['js_divergence_critical']}"
        )

    # Performance degradation
    if not perf_results.get("insufficient_data"):
        current_auc = perf_results.get("current_auc_roc", 1.0)
        ref_auc = perf_results.get("reference_auc_roc", 1.0)
        degradation = ref_auc - current_auc
        if degradation >= mon_cfg["auc_degradation_critical"]:
            reasons.append(
                f"AUC degradation={degradation:.4f} >= critical "
                f"{mon_cfg['auc_degradation_critical']} "
                f"(ref={ref_auc:.4f} → cur={current_auc:.4f})"
            )

    should_retrain = bool(reasons)
    reason_str = "; ".join(reasons) if reasons else "No critical drift detected"
    logger.info("Drift check result: retrain=%s | %s", should_retrain, reason_str)

    if should_retrain and dag_trigger_url:
        _trigger_airflow_dag(dag_trigger_url, reason_str)

    return should_retrain, reason_str


def _trigger_airflow_dag(trigger_url: str, reason: str) -> None:
    """POST to Airflow REST API to trigger the retraining DAG."""
    payload = {
        "conf": {
            "trigger_reason": reason,
            "triggered_by": "drift_monitor",
            "triggered_at": datetime.utcnow().isoformat(),
        }
    }
    try:
        resp = requests.post(
            trigger_url,
            json=payload,
            auth=(
                os.getenv("AIRFLOW_USER", "admin"),
                os.getenv("AIRFLOW_PASSWORD", "admin"),
            ),
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Retraining DAG triggered: HTTP %d", resp.status_code)
    except Exception as e:
        logger.error("Failed to trigger retraining DAG: %s", e)
        raise


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────


def save_drift_report(
    report_type: str,
    model_version: str,
    drift_results: dict,
    retrain_triggered: bool = False,
    dag_run_id: str | None = None,
    engine=None,
) -> str:
    """Persist drift report to PostgreSQL."""
    engine = engine or get_engine()
    report_id = str(uuid.uuid4())
    now = datetime.utcnow()

    row = {
        "report_id": report_id,
        "report_type": report_type,
        "model_version": model_version,
        "reference_start": (now - timedelta(days=mon_cfg["reference_window_days"] * 2)).date(),
        "reference_end": (now - timedelta(days=mon_cfg["reference_window_days"])).date(),
        "current_start": (now - timedelta(days=mon_cfg["production_window_days"])).date(),
        "current_end": now.date(),
        "overall_drift_score": drift_results.get("overall_drift_score"),
        "is_drift_detected": drift_results.get("is_drift_detected", False),
        "threshold_used": drift_results.get("threshold_used", mon_cfg["psi_critical"]),
        "feature_drift_json": json.dumps(drift_results.get("feature_drift", {})),
        "performance_json": json.dumps(drift_results.get("performance_metrics", {})),
        "report_html_path": drift_results.get("report_html_path"),
        "retrain_triggered": retrain_triggered,
        "dag_run_id": dag_run_id,
        "created_at": now,
    }

    df = pd.DataFrame([row])
    with engine.begin() as conn:
        df.to_sql(
            "drift_reports",
            conn,
            schema=SCHEMA,
            if_exists="append",
            index=False,
        )
    logger.info("Drift report %s saved (type=%s)", report_id, report_type)
    return report_id


# ─────────────────────────────────────────────
# Evidently result parsers
# ─────────────────────────────────────────────


def _parse_data_drift(result: dict) -> dict:
    try:
        drift_section = result["metrics"][0]["result"]
        feature_drifts = {}
        share_drifted = drift_section.get("share_of_drifted_columns", 0)
        for col_name, col_data in drift_section.get("drift_by_columns", {}).items():
            feature_drifts[col_name] = {
                "drift_score": col_data.get("drift_score"),
                "drift_detected": col_data.get("drift_detected"),
                "stattest": col_data.get("stattest_name"),
            }
        return {
            "report_type": "data_drift",
            "overall_drift_score": share_drifted,
            "is_drift_detected": share_drifted >= mon_cfg["psi_warning"],
            "threshold_used": mon_cfg["psi_critical"],
            "feature_drift": feature_drifts,
        }
    except (KeyError, IndexError) as e:
        logger.warning("Could not parse data drift result: %s", e)
        return {
            "report_type": "data_drift",
            "overall_drift_score": 0,
            "is_drift_detected": False,
            "feature_drift": {},
        }


def _parse_prediction_drift(result: dict) -> dict:
    try:
        drift_section = result["metrics"][0]["result"]
        score = drift_section.get("drift_score", 0)
        return {
            "report_type": "prediction_drift",
            "overall_drift_score": score,
            "is_drift_detected": score >= mon_cfg["js_divergence_warning"],
            "threshold_used": mon_cfg["js_divergence_critical"],
            "feature_drift": {"prediction_proba": {"drift_score": score}},
        }
    except (KeyError, IndexError) as e:
        logger.warning("Could not parse prediction drift: %s", e)
        return {
            "report_type": "prediction_drift",
            "overall_drift_score": 0,
            "is_drift_detected": False,
        }


def _parse_performance(result: dict) -> dict:
    try:
        metrics = result["metrics"][0]["result"]
        return {
            "report_type": "performance",
            "current_auc_roc": metrics.get("current", {}).get("roc_auc", 0),
            "reference_auc_roc": metrics.get("reference", {}).get("roc_auc", 0),
            "performance_metrics": metrics,
        }
    except (KeyError, IndexError) as e:
        logger.warning("Could not parse performance result: %s", e)
        return {"report_type": "performance", "current_auc_roc": 0, "reference_auc_roc": 0}
