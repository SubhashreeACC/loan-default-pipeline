# src/training/trainer.py
"""
XGBoost training pipeline with full MLflow experiment tracking.
Handles train/val/test splits, hyperparameter logging, artefact storage,
and automated staging → production promotion via the MLflow Model Registry.
"""

from __future__ import annotations

import json
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.training.features import engineer_features
from src.utils.config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()
model_cfg = cfg["model"]
xgb_cfg = model_cfg["xgboost"]
val_cfg = model_cfg["validation"]


# ─────────────────────────────────────────────
# MLflow setup
# ─────────────────────────────────────────────


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return MlflowClient()


# ─────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────


def train(
    df: pd.DataFrame,
    run_name: str | None = None,
    tags: dict | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """
    Full training run: feature engineering → cross-validation → final fit →
    MLflow logging → model registration.

    Parameters
    ----------
    df       : raw DataFrame from ingestion
    run_name : MLflow run name
    tags     : additional MLflow tags
    cv_folds : number of stratified CV folds for validation

    Returns
    -------
    dict containing run_id, model_version, and evaluation metrics
    """
    client = setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # ── 1. Feature engineering ───────────────
        logger.info("Engineering features from %d raw records", len(df))
        x, y, feat_pipeline = engineer_features(df, fit=True)

        if len(x) < cfg["data"]["min_training_rows"]:
            raise ValueError(
                f"Insufficient training data: {len(x)} rows "
                f"(minimum {cfg['data']['min_training_rows']})"
            )

        # ── 2. Train/val/test split ──────────────
        test_size = cfg["data"]["test_split"]
        val_size = cfg["data"]["val_split"] / (1 - test_size)

        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=xgb_cfg["random_state"]
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval,
            y_trainval,
            test_size=val_size,
            stratify=y_trainval,
            random_state=xgb_cfg["random_state"],
        )

        logger.info(
            "Split: train=%d  val=%d  test=%d  default_rate=%.3f",
            len(x_train),
            len(x_val),
            len(x_test),
            y_train.mean(),
        )

        # ── 3. Log parameters ────────────────────
        mlflow.log_params(
            {
                "n_training_rows": len(x_train),
                "n_val_rows": len(x_val),
                "n_test_rows": len(x_test),
                "n_features": x_train.shape[1],
                "train_default_rate": float(y_train.mean()),
                **xgb_cfg,
            }
        )

        mlflow.set_tags(
            {
                "pipeline_version": "1.0",
                "feature_version": "1.0",
                **(tags or {}),
            }
        )

        # ── 4. Cross-validation ──────────────────
        cv_scores = _cross_validate(x_trainval, y_trainval, cv_folds)
        mlflow.log_metrics(
            {
                "cv_auc_mean": cv_scores["auc_mean"],
                "cv_auc_std": cv_scores["auc_std"],
            }
        )
        logger.info(
            "CV AUC-ROC: %.4f ± %.4f",
            cv_scores["auc_mean"],
            cv_scores["auc_std"],
        )

        # ── 5. Final model training ──────────────
        model = _build_xgb_model()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=100,
        )

        # ── 6. Evaluation ────────────────────────
        test_metrics = _evaluate(model, x_test, y_test, prefix="test")
        val_metrics = _evaluate(model, x_val, y_val, prefix="val")
        mlflow.log_metrics({**test_metrics, **val_metrics})

        logger.info(
            "Test AUC-ROC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            test_metrics["test_auc_roc"],
            test_metrics["test_f1"],
            test_metrics["test_precision"],
            test_metrics["test_recall"],
        )

        # ── 7. Artefacts ─────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            _save_artefacts(
                model, feat_pipeline, x_train, y_test, model.predict_proba(x_test)[:, 1], tmp
            )

        # ── 8. Log model to registry ─────────────
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["model_name"],
            input_example=x_train.head(5),
        )

        # ── 9. Promote to staging / production ───
        model_version = _register_and_promote(
            client=client,
            run_id=run_id,
            metrics=test_metrics,
        )

        return {
            "run_id": run_id,
            "model_version": model_version,
            "metrics": {**test_metrics, **cv_scores},
            "n_features": x_train.shape[1],
            "feature_names": list(x_train.columns),
        }


# ─────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────


def _build_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        eval_metric=xgb_cfg["eval_metric"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        n_jobs=xgb_cfg["n_jobs"],
        random_state=xgb_cfg["random_state"],
        enable_categorical=False,
    )


# ─────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────


def _cross_validate(X: pd.DataFrame, y: pd.Series, n_folds: int) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = _build_xgb_model()
        # Use last 20% of fold for early stopping
        n_es = max(1, int(len(train_idx) * 0.2))
        es_idx = train_idx[-n_es:]
        tr_idx = train_idx[:-n_es]
        model.fit(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            eval_set=[(X.iloc[es_idx], y.iloc[es_idx])],
            verbose=False,
        )
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        score = roc_auc_score(y.iloc[val_idx], proba)
        auc_scores.append(score)
        logger.debug("  Fold %d AUC: %.4f", fold + 1, score)
    return {
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "auc_min": float(np.min(auc_scores)),
        "auc_max": float(np.max(auc_scores)),
    }


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────


def _evaluate(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str = "test",
    threshold: float = 0.5,
) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        f"{prefix}_auc_roc": float(roc_auc_score(y, proba)),
        f"{prefix}_avg_precision": float(average_precision_score(y, proba)),
        f"{prefix}_brier_score": float(brier_score_loss(y, proba)),
        f"{prefix}_f1": float(f1_score(y, preds, zero_division=0)),
        f"{prefix}_precision": float(precision_score(y, preds, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, preds, zero_division=0)),
    }


# ─────────────────────────────────────────────
# Artefact persistence
# ─────────────────────────────────────────────


def _save_artefacts(model, feat_pipeline, x_train, y_test, y_proba, tmp_dir: str) -> None:
    tmp = Path(tmp_dir)

    # Feature importance
    fi = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi_path = tmp / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path))

    # Feature pipeline
    pipe_path = tmp / "feature_pipeline.pkl"
    with open(pipe_path, "wb") as f:
        pickle.dump(feat_pipeline, f)
    mlflow.log_artifact(str(pipe_path))

    # Classification report
    preds = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True)
    report_path = tmp / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    mlflow.log_artifact(str(report_path))

    # Feature names
    features_path = tmp / "feature_names.json"
    features_path.write_text(json.dumps(list(x_train.columns)))
    mlflow.log_artifact(str(features_path))


# ─────────────────────────────────────────────
# Registry promotion
# ─────────────────────────────────────────────


def _register_and_promote(
    client: MlflowClient,
    run_id: str,
    metrics: dict,
) -> str:
    """
    Transition new model to Staging always.
    Promote to Production only if it passes validation thresholds
    AND outperforms the current Production model.
    """
    model_name = cfg["mlflow"]["model_name"]

    # Get latest version just registered
    versions = client.search_model_versions(f"name='{model_name}'")
    versions_for_run = [v for v in versions if v.run_id == run_id]
    if not versions_for_run:
        raise RuntimeError(f"No model version found for run {run_id}")
    new_version = versions_for_run[0].version

    # Move to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info("Model version %s → Staging", new_version)

    # Validate against thresholds
    passes_validation = _validate_thresholds(metrics)
    if not passes_validation:
        logger.warning(
            "Model version %s did NOT pass validation thresholds. Remaining in Staging.",
            new_version,
        )
        return new_version

    # Check if it beats current Production
    prod_auc = _get_production_auc(client, model_name)
    new_auc = metrics["test_auc_roc"]

    if prod_auc is not None and new_auc <= prod_auc:
        logger.warning(
            "New model AUC (%.4f) does not exceed Production AUC (%.4f). Staying in Staging.",
            new_auc,
            prod_auc,
        )
        return new_version

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        "Model version %s promoted to Production (AUC=%.4f)",
        new_version,
        new_auc,
    )
    mlflow.set_tag("promoted_to_production", "true")
    return new_version


def _validate_thresholds(metrics: dict) -> bool:
    checks = {
        "auc_roc": (metrics.get("test_auc_roc", 0), val_cfg["min_auc_roc"]),
        "precision": (metrics.get("test_precision", 0), val_cfg["min_precision"]),
        "recall": (metrics.get("test_recall", 0), val_cfg["min_recall"]),
        "f1": (metrics.get("test_f1", 0), val_cfg["min_f1"]),
    }
    brier = metrics.get("test_brier_score", 1.0)

    failures = []
    for name, (actual, minimum) in checks.items():
        if actual < minimum:
            failures.append(f"{name}={actual:.4f} < {minimum}")
    if brier > val_cfg["max_brier_score"]:
        failures.append(f"brier_score={brier:.4f} > {val_cfg['max_brier_score']}")

    if failures:
        logger.warning("Validation failures: %s", ", ".join(failures))
        return False
    return True


def _get_production_auc(client: MlflowClient, model_name: str) -> float | None:
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        return None
    prod_run_id = prod_versions[0].run_id
    prod_run = client.get_run(prod_run_id)
    return prod_run.data.metrics.get("test_auc_roc")


# ─────────────────────────────────────────────
# Load production model for inference
# ─────────────────────────────────────────────


def load_production_model() -> tuple[Any, str]:
    """Load the current Production model from the MLflow registry."""
    setup_mlflow()
    model_uri = f"models:/{cfg['mlflow']['model_name']}/Production"
    try:
        model = mlflow.xgboost.load_model(model_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(cfg["mlflow"]["model_name"], stages=["Production"])
        version = versions[0].version if versions else "unknown"
        logger.info("Loaded Production model version %s", version)
        return model, version
    except Exception as e:
        logger.error("Failed to load production model: %s", e)
        raise
