# tests/integration/test_data_pipeline.py
"""
Integration tests for the ingestion → validation → feature engineering flow.
Requires a running PostgreSQL instance (see conftest.py for fixture setup).
Run with: pytest tests/integration -m integration
"""
import uuid

import pandas as pd
import pytest
import sqlalchemy as sa

from src.data.ingestion import ingest_dataframe, get_training_data, get_data_stats
from src.training.features import engineer_features
from src.utils.config import get_config

pytestmark = pytest.mark.integration

cfg = get_config()
SCHEMA = cfg["database"]["schema"]


@pytest.fixture
def sample_batch():
    """Synthetic batch resembling Home Credit raw rows."""
    n = 50
    return pd.DataFrame({
        "SK_ID_CURR": range(900000, 900000 + n),
        "TARGET": [i % 10 == 0 for i in range(n)],
        "AMT_INCOME_TOTAL": [150000.0 + i * 1000 for i in range(n)],
        "AMT_CREDIT": [300000.0 + i * 5000 for i in range(n)],
        "AMT_ANNUITY": [20000.0 + i * 100 for i in range(n)],
        "AMT_GOODS_PRICE": [280000.0 + i * 4000 for i in range(n)],
        "DAYS_BIRTH": [-9000 - i * 10 for i in range(n)],
        "DAYS_EMPLOYED": [-1000 - i * 5 for i in range(n)],
        "CODE_GENDER": ["M" if i % 2 == 0 else "F" for i in range(n)],
        "NAME_INCOME_TYPE": ["Working"] * n,
        "NAME_EDUCATION_TYPE": ["Higher education"] * n,
        "NAME_FAMILY_STATUS": ["Married"] * n,
        "NAME_HOUSING_TYPE": ["House / apartment"] * n,
        "NAME_CONTRACT_TYPE": ["Cash loans"] * n,
        "EXT_SOURCE_1": [0.5] * n,
        "EXT_SOURCE_2": [0.6] * n,
        "EXT_SOURCE_3": [0.4] * n,
        "CNT_CHILDREN": [0] * n,
        "CNT_FAM_MEMBERS": [2.0] * n,
        "FLAG_OWN_CAR": ["N"] * n,
        "FLAG_OWN_REALTY": ["Y"] * n,
        "REGION_RATING_CLIENT": [2] * n,
    })


@pytest.fixture(autouse=True)
def cleanup_test_rows():
    """Remove synthetic test rows before and after each test."""
    from src.utils.db import get_engine
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            sa.text(f"DELETE FROM {SCHEMA}.raw_applications WHERE sk_id_curr >= 900000")
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            sa.text(f"DELETE FROM {SCHEMA}.raw_applications WHERE sk_id_curr >= 900000")
        )


class TestIngestionToDatabase:
    def test_ingest_dataframe_inserts_rows(self, sample_batch):
        batch_id = str(uuid.uuid4())[:8]
        result = ingest_dataframe(sample_batch, batch_id=batch_id, source_label="test")
        assert result["inserted"] == len(sample_batch)

    def test_duplicate_ingestion_is_idempotent(self, sample_batch):
        ingest_dataframe(sample_batch, source_label="test")
        second = ingest_dataframe(sample_batch, source_label="test")
        # Second insert should skip all rows due to PK conflict
        assert second["inserted"] == 0

    def test_data_stats_reflect_ingested_rows(self, sample_batch):
        ingest_dataframe(sample_batch, source_label="test")
        stats = get_data_stats()
        assert stats["total_rows"] >= len(sample_batch)


class TestTrainingDataRetrieval:
    def test_get_training_data_returns_labelled_rows(self, sample_batch):
        ingest_dataframe(sample_batch, source_label="test")
        df = get_training_data()
        test_rows = df[df["sk_id_curr"] >= 900000]
        assert len(test_rows) == len(sample_batch)
        assert test_rows["target"].notna().all()


class TestEndToEndFeaturePipeline:
    def test_ingest_then_engineer_features(self, sample_batch):
        ingest_dataframe(sample_batch, source_label="test")
        df = get_training_data()
        test_rows = df[df["sk_id_curr"] >= 900000].reset_index(drop=True)

        X, y, pipeline = engineer_features(test_rows, fit=True)

        assert len(X) == len(test_rows)
        assert len(y) == len(test_rows)
        assert X.select_dtypes(exclude=["number"]).empty

    def test_pipeline_consistent_on_subset(self, sample_batch):
        ingest_dataframe(sample_batch, source_label="test")
        df = get_training_data()
        test_rows = df[df["sk_id_curr"] >= 900000].reset_index(drop=True)

        X_full, y_full, pipeline = engineer_features(test_rows, fit=True)
        X_subset, _, _ = engineer_features(
            test_rows.head(5), pipeline=pipeline, fit=False
        )
        assert set(X_full.columns) == set(X_subset.columns)
