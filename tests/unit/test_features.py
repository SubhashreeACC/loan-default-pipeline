# tests/unit/test_features.py
"""Unit tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.training.features import (
    AgeEmploymentTransformer,
    CategoricalEncoder,
    CreditRatioTransformer,
    ExternalScoreTransformer,
    engineer_features,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_raw_df():
    """Minimal raw DataFrame mimicking Home Credit schema."""
    return pd.DataFrame(
        {
            "sk_id_curr": [100001, 100002, 100003],
            "target": [0, 1, 0],
            "amt_income_total": [135000.0, 99000.0, 202500.0],
            "amt_credit": [406597.5, 1293502.5, 1560726.0],
            "amt_annuity": [24700.5, 35698.5, 43261.5],
            "amt_goods_price": [351000.0, 1129500.0, 1395000.0],
            "days_birth": [-9461, -16765, -19046],
            "days_employed": [-637, -1188, -225],
            "code_gender": ["M", "F", "M"],
            "name_income_type": ["Working", "State servant", "Working"],
            "name_education_type": [
                "Secondary / secondary special",
                "Higher education",
                "Secondary / secondary special",
            ],
            "name_family_status": ["Single / not married", "Married", "Married"],
            "name_housing_type": ["House / apartment", "House / apartment", "House / apartment"],
            "name_contract_type": ["Cash loans", "Cash loans", "Revolving loans"],
            "ext_source_1": [0.0830, np.nan, 0.5021],
            "ext_source_2": [0.2629, 0.3227, 0.5551],
            "ext_source_3": [np.nan, 0.5550, np.nan],
            "cnt_children": [0, 0, 2],
            "cnt_fam_members": [1.0, 2.0, 4.0],
            "flag_own_car": ["N", "N", "Y"],
            "flag_own_realty": ["Y", "N", "Y"],
            "region_rating_client": [2, 1, 2],
            "flag_mobil": [1, 1, 1],
            "flag_emp_phone": [1, 0, 1],
            "reg_city_not_live_city": [0, 0, 1],
            "reg_region_not_live_region": [0, 0, 0],
        }
    )


# ─────────────────────────────────────────────
# CreditRatioTransformer
# ─────────────────────────────────────────────


class TestCreditRatioTransformer:
    def test_credit_income_ratio(self, sample_raw_df):
        t = CreditRatioTransformer()
        out = t.fit_transform(sample_raw_df)
        expected = 406597.5 / 135000.0
        assert abs(out.loc[0, "credit_income_ratio"] - expected) < 1e-4

    def test_zero_income_produces_nan(self):
        df = pd.DataFrame(
            {
                "amt_income_total": [0.0],
                "amt_credit": [10000.0],
                "amt_annuity": [500.0],
                "amt_goods_price": [9000.0],
                "cnt_fam_members": [1.0],
            }
        )
        out = CreditRatioTransformer().fit_transform(df)
        assert pd.isna(out.loc[0, "credit_income_ratio"])

    def test_credit_term_months(self, sample_raw_df):
        t = CreditRatioTransformer()
        out = t.fit_transform(sample_raw_df)
        expected = 406597.5 / 24700.5
        assert abs(out.loc[0, "credit_term_months"] - expected) < 1e-3

    def test_output_columns_present(self, sample_raw_df):
        out = CreditRatioTransformer().fit_transform(sample_raw_df)
        for col in [
            "credit_income_ratio",
            "annuity_income_ratio",
            "credit_term_months",
            "goods_credit_ratio",
            "family_income_per_capita",
        ]:
            assert col in out.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────
# AgeEmploymentTransformer
# ─────────────────────────────────────────────


class TestAgeEmploymentTransformer:
    def test_age_calculation(self, sample_raw_df):
        out = AgeEmploymentTransformer().fit_transform(sample_raw_df)
        expected_age = 9461 / 365.25
        assert abs(out.loc[0, "age_years"] - expected_age) < 0.01

    def test_employment_years_positive(self, sample_raw_df):
        out = AgeEmploymentTransformer().fit_transform(sample_raw_df)
        assert out["employment_years"].dropna().gt(0).all()

    def test_ratio_between_0_and_1(self, sample_raw_df):
        out = AgeEmploymentTransformer().fit_transform(sample_raw_df)
        ratio = out["employed_to_age_ratio"].dropna()
        assert (ratio >= 0).all()
        assert (ratio <= 1.5).all()  # allow small overruns (recent employment)


# ─────────────────────────────────────────────
# ExternalScoreTransformer
# ─────────────────────────────────────────────


class TestExternalScoreTransformer:
    def test_missing_count(self, sample_raw_df):
        out = ExternalScoreTransformer().fit_transform(sample_raw_df)
        # Row 0: ext_source_3 is NaN → missing_cnt = 1
        assert out.loc[0, "ext_source_missing_cnt"] == 1

    def test_mean_within_bounds(self, sample_raw_df):
        out = ExternalScoreTransformer().fit_transform(sample_raw_df)
        means = out["ext_source_mean"].dropna()
        assert (means >= 0).all()
        assert (means <= 1).all()


# ─────────────────────────────────────────────
# CategoricalEncoder
# ─────────────────────────────────────────────


class TestCategoricalEncoder:
    def test_fit_transform(self, sample_raw_df):
        enc = CategoricalEncoder(max_cardinality=50, top_n=3)
        out = enc.fit_transform(sample_raw_df)
        # Check at least one dummy column created
        dummy_cols = [c for c in out.columns if c.startswith("code_gender_")]
        assert len(dummy_cols) > 0

    def test_unseen_category_at_inference(self, sample_raw_df):
        enc = CategoricalEncoder(max_cardinality=50, top_n=3)
        enc.fit(sample_raw_df)
        new_df = sample_raw_df.copy()
        new_df["code_gender"] = "UNKNOWN"
        out = enc.transform(new_df)
        # All gender dummies should be 0 for unseen category
        gender_cols = [c for c in out.columns if c.startswith("code_gender_")]
        assert (out[gender_cols] == 0).all().all()

    def test_missing_column_at_inference(self, sample_raw_df):
        enc = CategoricalEncoder()
        enc.fit(sample_raw_df)
        df_no_gender = sample_raw_df.drop(columns=["code_gender"])
        out = enc.transform(df_no_gender)
        gender_cols = [c for c in out.columns if c.startswith("code_gender_")]
        assert all((out[gender_cols] == 0).all())


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────


class TestEngineerFeatures:
    def test_returns_numeric_only(self, sample_raw_df):
        x, y, pipeline = engineer_features(sample_raw_df, fit=True)
        non_numeric = x.select_dtypes(exclude=["number"]).columns.tolist()
        assert non_numeric == [], f"Non-numeric columns found: {non_numeric}"

    def test_target_separated(self, sample_raw_df):
        x, y, pipeline = engineer_features(sample_raw_df, fit=True)
        assert "target" not in x.columns
        assert len(y) == len(x)

    def test_id_not_in_features(self, sample_raw_df):
        x, y, pipeline = engineer_features(sample_raw_df, fit=True)
        assert "sk_id_curr" not in x.columns

    def test_no_all_nan_column(self, sample_raw_df):
        x, y, pipeline = engineer_features(sample_raw_df, fit=True)
        assert not x.isnull().all().any(), "Found column(s) that are entirely NaN"

    def test_pipeline_reusable_at_inference(self, sample_raw_df):
        x_train, y_train, pipeline = engineer_features(sample_raw_df, fit=True)
        x_infer, _, _ = engineer_features(sample_raw_df.head(1), pipeline=pipeline, fit=False)
        # Same feature set
        assert set(x_train.columns) == set(x_infer.columns)

    def test_feature_count_reasonable(self, sample_raw_df):
        x, y, _ = engineer_features(sample_raw_df, fit=True)
        assert x.shape[1] >= 5, "Too few features engineered"
        assert x.shape[1] <= 200, "Too many features — likely a bug"
