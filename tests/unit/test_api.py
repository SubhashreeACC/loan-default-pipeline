# tests/unit/test_api.py
"""Unit tests for the FastAPI prediction service (model mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    model = MagicMock()
    # predict_proba returns [P(no default), P(default)]
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    return model


@pytest.fixture
def client(mock_model):
    """TestClient with model state pre-populated, lifespan skipped."""
    with patch("src.serving.api._load_model"):
        from src.serving import api as api_module

        api_module.state.model = mock_model
        api_module.state.version = "test-v1"
        api_module.state.feature_pipeline = None

        with patch(
            "src.serving.api.engineer_features",
            return_value=(pd.DataFrame({"f1": [1.0]}), pd.Series([0]), None),
        ):
            with TestClient(api_module.app) as c:
                yield c


SAMPLE_APPLICATION = {
    "amt_income_total": 135000.0,
    "amt_credit": 406597.5,
    "amt_annuity": 24700.5,
    "code_gender": "M",
    "days_birth": -9461,
    "days_employed": -637,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_model_version(self, client):
        resp = client.get("/health")
        assert resp.json()["model_version"] == "test-v1"

    def test_health_status_ok_when_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_contains_expected_fields(self, client):
        resp = client.get("/model/info")
        body = resp.json()
        assert "model_name" in body
        assert "model_version" in body


class TestPredictEndpoint:
    def test_predict_single_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_APPLICATION)
        assert resp.status_code == 200

    def test_predict_response_shape(self, client):
        resp = client.post("/predict", json=SAMPLE_APPLICATION)
        body = resp.json()
        for key in [
            "prediction_id",
            "default_probability",
            "prediction_label",
            "risk_band",
            "model_version",
            "predicted_at",
        ]:
            assert key in body

    def test_predict_probability_in_valid_range(self, client):
        resp = client.post("/predict", json=SAMPLE_APPLICATION)
        proba = resp.json()["default_probability"]
        assert 0.0 <= proba <= 1.0

    def test_predict_missing_required_field_returns_422(self, client):
        bad_payload = {"amt_income_total": 50000.0}  # missing amt_credit
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_negative_income_rejected(self, client):
        bad_payload = dict(SAMPLE_APPLICATION)
        bad_payload["amt_income_total"] = -100
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        payload = {"applications": [SAMPLE_APPLICATION, SAMPLE_APPLICATION]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_predict_counts_match(self, client):
        payload = {"applications": [SAMPLE_APPLICATION, SAMPLE_APPLICATION]}
        with (
            patch(
                "src.serving.api.engineer_features",
                return_value=(pd.DataFrame({"f1": [1.0, 1.0]}), pd.Series([0, 0]), None),
            ),
            patch.object(
                __import__("src.serving.api", fromlist=["state"]).state.model,
                "predict_proba",
                return_value=np.array([[0.7, 0.3], [0.6, 0.4]]),
            ),
        ):
            resp = client.post("/predict/batch", json=payload)
        body = resp.json()
        assert body["total_applications"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_rejects_too_many_applications(self, client):
        payload = {"applications": [SAMPLE_APPLICATION] * 1001}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 422
