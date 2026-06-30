# tests/integration/test_serving_pipeline.py
"""
Integration test for the FastAPI serving layer against a real
MLflow Production model and PostgreSQL prediction logging.
Run with: pytest tests/integration -m integration
Requires a model already promoted to Production (run test_training_pipeline first,
or rely on conftest.py's session-scoped fixture that trains+promotes one).
"""
import pytest
from fastapi.testclient import TestClient

from src.utils.config import get_config

pytestmark = pytest.mark.integration

cfg = get_config()

SAMPLE_APPLICATION = {
    "sk_id_curr": 999999,
    "amt_income_total": 180000.0,
    "amt_credit": 500000.0,
    "amt_annuity": 28000.0,
    "amt_goods_price": 450000.0,
    "code_gender": "F",
    "days_birth": -12000,
    "days_employed": -2000,
    "name_income_type": "Working",
    "name_education_type": "Higher education",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment",
    "name_contract_type": "Cash loans",
    "ext_source_1": 0.55,
    "ext_source_2": 0.62,
    "ext_source_3": 0.48,
    "cnt_children": 1,
    "cnt_fam_members": 3.0,
    "flag_own_car": "Y",
    "flag_own_realty": "Y",
    "region_rating_client": 2,
}


@pytest.fixture(scope="module")
def live_client(trained_production_model):
    """
    trained_production_model is a session/module fixture (see conftest.py)
    that ensures a Production-stage model exists in MLflow before the
    FastAPI lifespan loads it.
    """
    from src.serving.api import app
    with TestClient(app) as client:
        yield client


class TestLiveHealthCheck:
    def test_health_ok_with_real_model(self, live_client):
        resp = live_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_version"] != "unknown"


class TestLivePrediction:
    def test_single_prediction_end_to_end(self, live_client):
        resp = live_client.post("/predict", json=SAMPLE_APPLICATION)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["prediction_label"] in (0, 1)
        assert body["risk_band"] in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")

    def test_prediction_is_logged_to_database(self, live_client):
        import sqlalchemy as sa
        from src.utils.db import get_engine

        resp = live_client.post("/predict", json=SAMPLE_APPLICATION)
        pred_id = resp.json()["prediction_id"]

        engine = get_engine()
        schema = cfg["database"]["schema"]
        # Background task may take a moment; poll briefly
        import time
        found = False
        for _ in range(10):
            with engine.connect() as conn:
                row = conn.execute(
                    sa.text(
                        f"SELECT 1 FROM {schema}.prediction_log "
                        f"WHERE prediction_id = :pid"
                    ),
                    {"pid": pred_id},
                ).fetchone()
            if row:
                found = True
                break
            time.sleep(0.5)
        assert found, "Prediction was not logged to database"

    def test_batch_prediction_end_to_end(self, live_client):
        payload = {
            "applications": [SAMPLE_APPLICATION, SAMPLE_APPLICATION],
            "threshold": 0.5,
        }
        resp = live_client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_applications"] == 2
        assert len(body["predictions"]) == 2


class TestFeedbackLoop:
    def test_submit_feedback_updates_record(self, live_client):
        import sqlalchemy as sa
        from src.utils.db import get_engine
        import time

        resp = live_client.post("/predict", json=SAMPLE_APPLICATION)
        pred_id = resp.json()["prediction_id"]
        time.sleep(1)  # allow background log task to complete

        fb_resp = live_client.post(f"/feedback/{pred_id}", params={"actual_label": 1})
        assert fb_resp.status_code == 200

        engine = get_engine()
        schema = cfg["database"]["schema"]
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    f"SELECT actual_label FROM {schema}.prediction_log "
                    f"WHERE prediction_id = :pid"
                ),
                {"pid": pred_id},
            ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_invalid_feedback_label_rejected(self, live_client):
        resp = live_client.post(
            "/feedback/00000000-0000-0000-0000-000000000000",
            params={"actual_label": 5},
        )
        assert resp.status_code == 400
