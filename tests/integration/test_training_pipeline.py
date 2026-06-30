# tests/integration/test_training_pipeline.py
"""
Integration test for the full training pipeline: data → features → XGBoost → MLflow.
Requires PostgreSQL + a local/test MLflow tracking server (see conftest.py).
Run with: pytest tests/integration -m integration
"""

import uuid

import numpy as np
import pandas as pd
import pytest

from src.training.trainer import setup_mlflow, train
from src.utils.config import get_config

pytestmark = pytest.mark.integration

cfg = get_config()


@pytest.fixture
def synthetic_training_df():
    """
    Larger synthetic dataset with a learnable signal, sized above
    min_training_rows so the trainer doesn't short-circuit.
    """
    rng = np.random.RandomState(7)
    n = max(cfg["data"]["min_training_rows"], 2000)

    ext_source_2 = rng.uniform(0, 1, n)
    # Lower ext_source_2 → higher default probability
    default_prob = 1 - ext_source_2
    target = (rng.uniform(0, 1, n) < default_prob * 0.3).astype(int)

    df = pd.DataFrame(
        {
            "sk_id_curr": range(800000, 800000 + n),
            "target": target,
            "amt_income_total": rng.uniform(50000, 300000, n),
            "amt_credit": rng.uniform(100000, 2000000, n),
            "amt_annuity": rng.uniform(10000, 80000, n),
            "amt_goods_price": rng.uniform(90000, 1900000, n),
            "days_birth": -rng.randint(7000, 25000, n),
            "days_employed": -rng.randint(100, 10000, n),
            "code_gender": rng.choice(["M", "F"], n),
            "name_income_type": rng.choice(["Working", "Commercial associate", "State servant"], n),
            "name_education_type": rng.choice(
                ["Higher education", "Secondary / secondary special"], n
            ),
            "name_family_status": rng.choice(["Married", "Single / not married"], n),
            "name_housing_type": ["House / apartment"] * n,
            "name_contract_type": rng.choice(["Cash loans", "Revolving loans"], n),
            "ext_source_1": rng.uniform(0, 1, n),
            "ext_source_2": ext_source_2,
            "ext_source_3": rng.uniform(0, 1, n),
            "cnt_children": rng.randint(0, 4, n),
            "cnt_fam_members": rng.randint(1, 6, n).astype(float),
            "flag_own_car": rng.choice(["Y", "N"], n),
            "flag_own_realty": rng.choice(["Y", "N"], n),
            "region_rating_client": rng.randint(1, 3, n),
        }
    )
    return df


class TestTrainingPipeline:
    def test_train_completes_and_returns_metrics(self, synthetic_training_df):
        result = train(
            df=synthetic_training_df,
            run_name=f"integration_test_{uuid.uuid4().hex[:8]}",
            tags={"test": "true"},
            cv_folds=3,
        )
        assert "run_id" in result
        assert "model_version" in result
        assert "metrics" in result
        assert result["metrics"]["test_auc_roc"] > 0.5

    def test_train_raises_on_insufficient_data(self):
        tiny_df = pd.DataFrame(
            {
                "sk_id_curr": [1, 2, 3],
                "target": [0, 1, 0],
                "amt_income_total": [100000.0] * 3,
                "amt_credit": [200000.0] * 3,
                "amt_annuity": [10000.0] * 3,
                "amt_goods_price": [180000.0] * 3,
                "days_birth": [-9000] * 3,
                "days_employed": [-1000] * 3,
            }
        )
        with pytest.raises(ValueError, match="Insufficient training data"):
            train(df=tiny_df, run_name="should_fail")

    def test_model_registered_in_mlflow(self, synthetic_training_df):

        result = train(
            df=synthetic_training_df,
            run_name=f"integration_registry_test_{uuid.uuid4().hex[:8]}",
            cv_folds=3,
        )
        client = setup_mlflow()
        versions = client.search_model_versions(f"name='{cfg['mlflow']['model_name']}'")
        matching = [v for v in versions if v.run_id == result["run_id"]]
        assert len(matching) == 1
        assert matching[0].current_stage in ("Staging", "Production")
