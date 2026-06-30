# src/data/validation.py
"""
Data validation using Great Expectations.
Builds and runs expectation suites against raw and engineered features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import AbstractDataContext

from src.utils.config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()

GE_ROOT = Path(cfg["great_expectations"]["context_root"])
DATASOURCE = cfg["great_expectations"]["datasource_name"]
SUITE_NAME = cfg["great_expectations"]["suite_name"]
CHECKPOINT = cfg["great_expectations"]["checkpoint_name"]


# ─────────────────────────────────────────────
# Context bootstrap
# ─────────────────────────────────────────────


def get_ge_context() -> AbstractDataContext:
    """Return (or create) a Great Expectations FileDataContext."""
    if not (GE_ROOT / "great_expectations.yml").exists():
        logger.info("Initialising new GE context at %s", GE_ROOT)
        context = gx.get_context(mode="file", project_root_dir=str(GE_ROOT))
        _add_pg_datasource(context)
        _create_expectation_suite(context)
        _create_checkpoint(context)
    else:
        context = gx.get_context(mode="file", project_root_dir=str(GE_ROOT))
    return context


def _add_pg_datasource(context: AbstractDataContext) -> None:
    db_cfg = cfg["database"]
    conn_str = (
        f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}"
    )
    context.sources.add_or_update_sql(
        name=DATASOURCE,
        connection_string=conn_str,
    )


# ─────────────────────────────────────────────
# Expectation suite definition
# ─────────────────────────────────────────────


def _create_expectation_suite(context: AbstractDataContext) -> None:
    """Define all expectations for the raw loan applications table."""
    suite = context.add_or_update_expectation_suite(expectation_suite_name=SUITE_NAME)

    # ── Schema expectations ──────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 100, "max_value": None},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "sk_id_curr"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "sk_id_curr"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "sk_id_curr"},
        )
    )

    # ── Target column ────────────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "target", "value_set": [0, 1]},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_mean_to_be_between",
            # Default rate expected 5–20 %
            kwargs={"column": "target", "min_value": 0.03, "max_value": 0.30},
        )
    )

    # ── Income ───────────────────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amt_income_total",
                "min_value": 1_000,
                "max_value": 100_000_000,
                "mostly": 0.99,
            },
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "amt_income_total", "mostly": 0.99},
        )
    )

    # ── Credit amount ────────────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amt_credit",
                "min_value": 0,
                "max_value": 10_000_000,
                "mostly": 0.99,
            },
        )
    )

    # ── Days birth (age) ─────────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "days_birth",
                "min_value": -30_000,  # ~82 years
                "max_value": -6_500,  # ~18 years
                "mostly": 0.99,
            },
        )
    )

    # ── Gender ───────────────────────────────
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "code_gender",
                "value_set": ["M", "F", "XNA"],
                "mostly": 0.99,
            },
        )
    )

    # ── External source scores ────────────────
    for col in ["ext_source_1", "ext_source_2", "ext_source_3"]:
        suite.add_expectation(
            gx.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0.0, "max_value": 1.0, "mostly": 0.95},
            )
        )

    # ── Null rate guardrails ──────────────────
    high_completeness_cols = [
        "amt_income_total",
        "amt_credit",
        "days_birth",
        "code_gender",
        "name_income_type",
        "name_education_type",
    ]
    for col in high_completeness_cols:
        suite.add_expectation(
            gx.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col, "mostly": 0.98},
            )
        )

    context.save_expectation_suite(suite)
    logger.info("Expectation suite '%s' saved.", SUITE_NAME)


def _create_checkpoint(context: AbstractDataContext) -> None:
    context.add_or_update_checkpoint(
        name=CHECKPOINT,
        config_version=1.0,
        class_name="SimpleCheckpoint",
        validations=[
            {
                "batch_request": {
                    "datasource_name": DATASOURCE,
                    "data_connector_name": "default_runtime_data_connector_name",
                    "data_asset_name": "raw_applications",
                },
                "expectation_suite_name": SUITE_NAME,
            }
        ],
    )


# ─────────────────────────────────────────────
# Runtime validation
# ─────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    batch_id: str,
    context: AbstractDataContext | None = None,
    raise_on_failure: bool = True,
) -> dict:
    """
    Validate a DataFrame against the loan features expectation suite.

    Returns a dict with validation results.
    Raises ValidationError if raise_on_failure=True and validation fails.
    """
    context = context or get_ge_context()

    batch_request = RuntimeBatchRequest(
        datasource_name=DATASOURCE,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="runtime_loan_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": batch_id},
    )

    results = context.run_checkpoint(
        checkpoint_name=CHECKPOINT,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": SUITE_NAME,
            }
        ],
    )

    validation_result = results.list_validation_results()[0]
    stats = validation_result.statistics

    summary = {
        "batch_id": batch_id,
        "success": bool(validation_result.success),
        "evaluated_expectations": stats["evaluated_expectations"],
        "successful_expectations": stats["successful_expectations"],
        "unsuccessful_expectations": stats["unsuccessful_expectations"],
        "success_percent": stats["success_percent"],
        "failed_expectations": _extract_failures(validation_result),
    }

    if not summary["success"]:
        msg = (
            f"Data validation FAILED for batch '{batch_id}'. "
            f"{summary['unsuccessful_expectations']} expectations failed: "
            f"{json.dumps(summary['failed_expectations'], indent=2)}"
        )
        logger.error(msg)
        if raise_on_failure:
            raise DataValidationError(msg)
    else:
        logger.info(
            "Validation passed for batch '%s' (%d/%d expectations met).",
            batch_id,
            summary["successful_expectations"],
            summary["evaluated_expectations"],
        )

    return summary


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """Quick column-level schema check (pre-GE fast path)."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
    return missing


def _extract_failures(validation_result) -> list[dict]:
    failures = []
    for result in validation_result.results:
        if not result.success:
            failures.append(
                {
                    "expectation_type": result.expectation_config.expectation_type,
                    "column": result.expectation_config.kwargs.get("column"),
                    "kwargs": result.expectation_config.kwargs,
                }
            )
    return failures


class DataValidationError(Exception):
    """Raised when a Great Expectations validation suite fails."""
