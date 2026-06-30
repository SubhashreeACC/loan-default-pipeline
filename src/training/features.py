# src/training/features.py
"""
Feature engineering pipeline for loan default prediction.
Transforms raw application data into model-ready features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.utils.config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()
feat_cfg = cfg["model"]["features"]

# Columns needed from raw table
RAW_NUMERIC_COLS = [
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
    "amt_goods_price",
    "days_birth",
    "days_employed",
    "region_population_relative",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "cnt_children",
    "cnt_fam_members",
    "own_car_age",
    "region_rating_client",
    "region_rating_client_w_city",
    "hour_appr_process_start",
]

RAW_CATEGORICAL_COLS = [
    "name_contract_type",
    "code_gender",
    "flag_own_car",
    "flag_own_realty",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "organization_type",
    "weekday_appr_process_start",
]

FLAG_COLS = [
    "flag_mobil",
    "flag_emp_phone",
    "flag_work_phone",
    "flag_cont_mobile",
    "flag_phone",
    "flag_email",
    "reg_region_not_live_region",
    "reg_region_not_work_region",
    "live_region_not_work_region",
    "reg_city_not_live_city",
    "reg_city_not_work_city",
    "live_city_not_work_city",
]

TARGET_COL = "target"
ID_COL = "sk_id_curr"


# ─────────────────────────────────────────────
# Custom transformers
# ─────────────────────────────────────────────


class CreditRatioTransformer(BaseEstimator, TransformerMixin):
    """Adds domain-specific ratio features."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        # Credit-to-income ratio
        df["credit_income_ratio"] = np.where(
            df["amt_income_total"] > 0,
            df["amt_credit"] / df["amt_income_total"],
            np.nan,
        )

        # Annuity-to-income ratio (monthly burden)
        df["annuity_income_ratio"] = np.where(
            df["amt_income_total"] > 0,
            df["amt_annuity"] / df["amt_income_total"],
            np.nan,
        )

        # Credit term in months
        df["credit_term_months"] = np.where(
            df["amt_annuity"] > 0,
            df["amt_credit"] / df["amt_annuity"],
            np.nan,
        )

        # Goods-to-credit ratio (LTV proxy)
        df["goods_credit_ratio"] = np.where(
            df["amt_credit"] > 0,
            df["amt_goods_price"] / df["amt_credit"],
            np.nan,
        )

        # Family income per capita
        df["family_income_per_capita"] = np.where(
            df["cnt_fam_members"] > 0,
            df["amt_income_total"] / df["cnt_fam_members"],
            df["amt_income_total"],
        )

        return df


class AgeEmploymentTransformer(BaseEstimator, TransformerMixin):
    """Converts DAYS columns to years and adds employment ratio."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        df["age_years"] = -df["days_birth"] / 365.25
        df["employment_years"] = np.where(
            df["days_employed"].isna(),
            np.nan,
            -df["days_employed"] / 365.25,
        )
        df["employed_to_age_ratio"] = np.where(
            df["age_years"] > 0,
            df["employment_years"] / df["age_years"],
            np.nan,
        )
        return df


class ExternalScoreTransformer(BaseEstimator, TransformerMixin):
    """Aggregates the three external credit scores."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        ext_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
        df["ext_source_mean"] = df[ext_cols].mean(axis=1)
        df["ext_source_std"] = df[ext_cols].std(axis=1).fillna(0)
        df["ext_source_missing_cnt"] = df[ext_cols].isna().sum(axis=1)
        return df


class FlagAggregatorTransformer(BaseEstimator, TransformerMixin):
    """Aggregates binary flag columns."""

    def __init__(self, flag_cols: list[str] = FLAG_COLS):
        self.flag_cols = flag_cols

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        available_flags = [c for c in self.flag_cols if c in df.columns]
        region_flags = [c for c in available_flags if "region" in c or "city" in c]
        if region_flags:
            df["total_region_mismatches"] = df[region_flags].fillna(0).sum(axis=1)

        doc_cols = [c for c in df.columns if c.startswith("flag_document_")]
        if doc_cols:
            df["docs_provided_count"] = df[doc_cols].fillna(0).sum(axis=1)
        else:
            df["docs_provided_count"] = 0

        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes categoricals, handling unseen categories at inference."""

    def __init__(self, max_cardinality: int = 50, top_n: int = 10):
        self.max_cardinality = max_cardinality
        self.top_n = top_n
        self.encoding_map_: dict[str, list[str]] = {}
        self.columns_: list[str] = []

    def fit(self, x: pd.DataFrame, y=None):
        self.encoding_map_ = {}
        self.columns_ = []

        for col in RAW_CATEGORICAL_COLS:
            if col not in x.columns:
                continue
            n_unique = x[col].nunique()
            if n_unique > self.max_cardinality:
                logger.debug("Dropping high-cardinality column: %s (%d unique)", col, n_unique)
                continue
            top_vals = x[col].value_counts().head(self.top_n).index.tolist()
            self.encoding_map_[col] = top_vals
            for val in top_vals:
                self.columns_.append(f"{col}_{val}".lower().replace(" ", "_"))
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        for col, top_vals in self.encoding_map_.items():
            if col not in df.columns:
                # Column absent at inference → fill all dummies with 0
                for val in top_vals:
                    dummy_col = f"{col}_{val}".lower().replace(" ", "_")
                    df[dummy_col] = 0
                continue
            for val in top_vals:
                dummy_col = f"{col}_{val}".lower().replace(" ", "_")
                df[dummy_col] = (df[col] == val).astype(int)
            df = df.drop(columns=[col])
        return df


class LowVarianceFilter(BaseEstimator, TransformerMixin):
    """Drops columns with near-zero variance, learned at fit time."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.to_keep_: list[str] = []

    def fit(self, x: pd.DataFrame, y=None):
        self.to_keep_ = [c for c in x.columns if x[c].var() >= self.threshold]
        if not self.to_keep_:
            self.to_keep_ = list(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.to_keep_ if c in x.columns]
        return x[keep]


class CorrelatedFeatureDropper(BaseEstimator, TransformerMixin):
    """Drops one feature from each highly correlated pair, learned at fit time."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_: list[str] = []

    def fit(self, x: pd.DataFrame, y=None):
        if x.shape[1] < 2:
            self.to_drop_ = []
            return self
        corr = x.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.to_drop_ if c in x.columns]
        return x.drop(columns=drop)


# ─────────────────────────────────────────────
# Feature pipeline
# ─────────────────────────────────────────────


def build_feature_pipeline() -> Pipeline:
    """
    Returns an sklearn Pipeline that transforms raw DataFrames
    into a numeric feature matrix ready for XGBoost.
    """
    return Pipeline(
        [
            ("credit_ratios", CreditRatioTransformer()),
            ("age_employment", AgeEmploymentTransformer()),
            ("ext_scores", ExternalScoreTransformer()),
            ("flag_agg", FlagAggregatorTransformer()),
            (
                "cat_encode",
                CategoricalEncoder(
                    max_cardinality=feat_cfg["max_cardinality"],
                ),
            ),
            (
                "low_var_filter",
                LowVarianceFilter(
                    threshold=feat_cfg.get("variance_threshold", 0.01),
                ),
            ),
            (
                "corr_dropper",
                CorrelatedFeatureDropper(
                    threshold=feat_cfg.get("correlation_threshold", 0.95),
                ),
            ),
        ]
    )


def engineer_features(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Apply feature engineering to a raw DataFrame.

    Parameters
    ----------
    df       : raw DataFrame from ingestion
    pipeline : existing pipeline (None = build new)
    fit      : if True, fit the pipeline on df (training); else transform only

    Returns
    -------
    x_features, y_target, fitted_pipeline
    """
    pipeline = pipeline or build_feature_pipeline()

    # Preserve target & ID before transforming
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else pd.Series(dtype=int)

    # Select feature columns
    drop_cols = [
        c
        for c in [
            TARGET_COL,
            ID_COL,
            "ingested_at",
            "data_batch_id",
            "source_file",
            "feature_version",
            "created_at",
        ]
        if c in df.columns
    ]
    x = df.drop(columns=drop_cols)

    if fit:
        x_transformed = pipeline.fit_transform(x)
    else:
        x_transformed = pipeline.transform(x)

    # Ensure numeric output
    if isinstance(x_transformed, pd.DataFrame):
        x_out = x_transformed.select_dtypes(include=[np.number])
    else:
        x_out = pd.DataFrame(x_transformed)

    logger.info(
        "Feature engineering complete: %d columns → %d features",
        len(df.columns),
        len(x_out.columns),
    )
    return x_out, y, pipeline
