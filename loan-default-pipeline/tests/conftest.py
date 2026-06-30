# tests/conftest.py
"""
Shared pytest fixtures and configuration.

Unit tests (tests/unit/) run with no external dependencies.
Integration tests (tests/integration/) require:
  - POSTGRES (DB_HOST etc. env vars, or docker-compose 'postgres' service)
  - MLflow tracking server (MLFLOW_TRACKING_URI)
Mark integration tests with @pytest.mark.integration; they are skipped by
default unless `--integration` is passed or RUN_INTEGRATION_TESTS=1 is set.
"""

import os
import uuid

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live Postgres/MLflow.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as requiring live infrastructure")


def pytest_collection_modifyitems(config, items):
    run_integration = config.getoption("--integration") or os.getenv("RUN_INTEGRATION_TESTS") == "1"
    if run_integration:
        return
    skip_integration = pytest.mark.skip(
        reason="need --integration flag or RUN_INTEGRATION_TESTS=1 to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ─────────────────────────────────────────────
# Integration fixtures
# ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def trained_production_model():
    """
    Ensures a model exists in MLflow Production stage before serving tests run.
    If one already exists (from a prior training DAG run), reuse it.
    Otherwise trains a quick synthetic model and forces promotion.
    """
    import numpy as np
    import pandas as pd

    from src.training.trainer import setup_mlflow, train
    from src.utils.config import get_config

    cfg = get_config()
    client = setup_mlflow()
    model_name = cfg["mlflow"]["model_name"]

    existing = client.get_latest_versions(model_name, stages=["Production"])
    if existing:
        yield existing[0]
        return

    rng = np.random.RandomState(123)
    n = max(cfg["data"]["min_training_rows"], 2000)
    ext_source_2 = rng.uniform(0, 1, n)
    target = (rng.uniform(0, 1, n) < (1 - ext_source_2) * 0.3).astype(int)

    df = pd.DataFrame(
        {
            "sk_id_curr": range(700000, 700000 + n),
            "target": target,
            "amt_income_total": rng.uniform(50000, 300000, n),
            "amt_credit": rng.uniform(100000, 2000000, n),
            "amt_annuity": rng.uniform(10000, 80000, n),
            "amt_goods_price": rng.uniform(90000, 1900000, n),
            "days_birth": -rng.randint(7000, 25000, n),
            "days_employed": -rng.randint(100, 10000, n),
            "code_gender": rng.choice(["M", "F"], n),
            "name_income_type": rng.choice(["Working", "Commercial associate"], n),
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

    result = train(df=df, run_name=f"conftest_bootstrap_{uuid.uuid4().hex[:8]}", cv_folds=3)

    # Force-promote regardless of beat-production check (test bootstrap only)
    client.transition_model_version_stage(
        name=model_name,
        version=result["model_version"],
        stage="Production",
        archive_existing_versions=True,
    )

    yield client.get_latest_versions(model_name, stages=["Production"])[0]
