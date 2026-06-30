# tests/unit/test_trainer.py
"""Unit tests for the training module (mocked MLflow)."""

import numpy as np
import pandas as pd
import pytest

from src.training.trainer import _build_xgb_model, _cross_validate, _evaluate, _validate_thresholds


@pytest.fixture
def binary_classification_data():
    rng = np.random.RandomState(42)
    n = 500
    x = pd.DataFrame({f"feat_{i}": rng.normal(size=n) for i in range(10)})
    # Make target weakly correlated with feat_0 so AUC > 0.5
    logits = x["feat_0"] * 2 + rng.normal(scale=0.5, size=n)
    y = pd.Series((logits > np.median(logits)).astype(int))
    return x, y


class TestBuildModel:
    def test_returns_xgb_classifier(self):
        model = _build_xgb_model()
        assert model.__class__.__name__ == "XGBClassifier"

    def test_uses_configured_hyperparams(self):
        model = _build_xgb_model()
        params = model.get_params()
        assert params["max_depth"] >= 1
        assert params["n_estimators"] >= 1


class TestEvaluateModel:
    def test_evaluate_returns_expected_keys(self, binary_classification_data):
        x, y = binary_classification_data
        model = _build_xgb_model()
        model.set_params(early_stopping_rounds=None)
        model.fit(x, y)
        metrics = _evaluate(model, x, y, prefix="test")
        for key in [
            "test_auc_roc",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_brier_score",
            "test_avg_precision",
        ]:
            assert key in metrics

    def test_auc_within_valid_range(self, binary_classification_data):
        x, y = binary_classification_data
        model = _build_xgb_model()
        model.set_params(early_stopping_rounds=None)
        model.fit(x, y)
        metrics = _evaluate(model, x, y, prefix="test")
        assert 0.0 <= metrics["test_auc_roc"] <= 1.0

    def test_better_than_random_on_separable_data(self, binary_classification_data):
        x, y = binary_classification_data
        model = _build_xgb_model()
        model.set_params(early_stopping_rounds=None)
        model.fit(x, y)
        metrics = _evaluate(model, x, y, prefix="test")
        assert metrics["test_auc_roc"] > 0.6


class TestValidateThresholds:
    def test_passes_when_above_minimums(self):
        metrics = {
            "test_auc_roc": 0.80,
            "test_precision": 0.65,
            "test_recall": 0.60,
            "test_f1": 0.65,
            "test_brier_score": 0.10,
        }
        assert _validate_thresholds(metrics) is True

    def test_fails_when_auc_below_minimum(self):
        metrics = {
            "test_auc_roc": 0.50,
            "test_precision": 0.65,
            "test_recall": 0.60,
            "test_f1": 0.65,
            "test_brier_score": 0.10,
        }
        assert _validate_thresholds(metrics) is False

    def test_fails_when_brier_too_high(self):
        metrics = {
            "test_auc_roc": 0.80,
            "test_precision": 0.65,
            "test_recall": 0.60,
            "test_f1": 0.65,
            "test_brier_score": 0.50,
        }
        assert _validate_thresholds(metrics) is False

    def test_fails_with_missing_metrics_defaults_to_zero(self):
        assert _validate_thresholds({}) is False


class TestCrossValidate:
    def test_returns_expected_stat_keys(self, binary_classification_data):
        x, y = binary_classification_data
        result = _cross_validate(x, y, n_folds=3)
        for key in ["auc_mean", "auc_std", "auc_min", "auc_max"]:
            assert key in result

    def test_auc_mean_in_valid_range(self, binary_classification_data):
        x, y = binary_classification_data
        result = _cross_validate(x, y, n_folds=3)
        assert 0.0 <= result["auc_mean"] <= 1.0
