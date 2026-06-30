# src/data/ingestion.py
"""
Data ingestion module — loads raw Home Credit / LendingClub data into PostgreSQL.
Supports initial bulk load and incremental batch loads.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.config import get_config
from src.utils.db import get_engine, upsert_dataframe

logger = logging.getLogger(__name__)
cfg = get_config()


# ─────────────────────────────────────────────
# Column mapping  (Kaggle raw → DB column names)
# ─────────────────────────────────────────────

RAW_COLUMN_MAP: dict[str, str] = {
    "SK_ID_CURR": "sk_id_curr",
    "TARGET": "target",
    "NAME_CONTRACT_TYPE": "name_contract_type",
    "CODE_GENDER": "code_gender",
    "FLAG_OWN_CAR": "flag_own_car",
    "FLAG_OWN_REALTY": "flag_own_realty",
    "CNT_CHILDREN": "cnt_children",
    "AMT_INCOME_TOTAL": "amt_income_total",
    "AMT_CREDIT": "amt_credit",
    "AMT_ANNUITY": "amt_annuity",
    "AMT_GOODS_PRICE": "amt_goods_price",
    "NAME_TYPE_SUITE": "name_type_suite",
    "NAME_INCOME_TYPE": "name_income_type",
    "NAME_EDUCATION_TYPE": "name_education_type",
    "NAME_FAMILY_STATUS": "name_family_status",
    "NAME_HOUSING_TYPE": "name_housing_type",
    "REGION_POPULATION_RELATIVE": "region_population_relative",
    "DAYS_BIRTH": "days_birth",
    "DAYS_EMPLOYED": "days_employed",
    "DAYS_REGISTRATION": "days_registration",
    "DAYS_ID_PUBLISH": "days_id_publish",
    "OWN_CAR_AGE": "own_car_age",
    "FLAG_MOBIL": "flag_mobil",
    "FLAG_EMP_PHONE": "flag_emp_phone",
    "FLAG_WORK_PHONE": "flag_work_phone",
    "FLAG_CONT_MOBILE": "flag_cont_mobile",
    "FLAG_PHONE": "flag_phone",
    "FLAG_EMAIL": "flag_email",
    "OCCUPATION_TYPE": "occupation_type",
    "CNT_FAM_MEMBERS": "cnt_fam_members",
    "REGION_RATING_CLIENT": "region_rating_client",
    "REGION_RATING_CLIENT_W_CITY": "region_rating_client_w_city",
    "WEEKDAY_APPR_PROCESS_START": "weekday_appr_process_start",
    "HOUR_APPR_PROCESS_START": "hour_appr_process_start",
    "REG_REGION_NOT_LIVE_REGION": "reg_region_not_live_region",
    "REG_REGION_NOT_WORK_REGION": "reg_region_not_work_region",
    "LIVE_REGION_NOT_WORK_REGION": "live_region_not_work_region",
    "REG_CITY_NOT_LIVE_CITY": "reg_city_not_live_city",
    "REG_CITY_NOT_WORK_CITY": "reg_city_not_work_city",
    "LIVE_CITY_NOT_WORK_CITY": "live_city_not_work_city",
    "ORGANIZATION_TYPE": "organization_type",
    "EXT_SOURCE_1": "ext_source_1",
    "EXT_SOURCE_2": "ext_source_2",
    "EXT_SOURCE_3": "ext_source_3",
}

SCHEMA = cfg["database"]["schema"]
RAW_TABLE = cfg["data"]["raw_table"]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def ingest_csv(
    file_path: str | Path,
    engine: Optional[Engine] = None,
    batch_size: int = 10_000,
    source_label: Optional[str] = None,
) -> dict:
    """
    Ingest a raw CSV file (Home Credit format) into the PostgreSQL raw table.

    Parameters
    ----------
    file_path : path to CSV file
    engine    : SQLAlchemy engine (created from config if None)
    batch_size: rows per upsert batch
    source_label: tag written to source_file column

    Returns
    -------
    dict with ingestion stats
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    engine = engine or get_engine()
    batch_id = _make_batch_id(file_path)
    source_label = source_label or file_path.name

    logger.info("Starting ingestion: %s (batch_id=%s)", file_path, batch_id)

    total_rows = 0
    inserted = 0
    skipped = 0

    for chunk_num, chunk in enumerate(
        pd.read_csv(file_path, chunksize=batch_size, low_memory=False)
    ):
        df = _preprocess_chunk(chunk, batch_id, source_label)
        rows_in, rows_skip = _upsert_chunk(df, engine)
        inserted += rows_in
        skipped += rows_skip
        total_rows += len(df)

        if chunk_num % 5 == 0:
            logger.info(
                "  chunk %d → %d rows processed (inserted=%d, skipped=%d)",
                chunk_num, total_rows, inserted, skipped,
            )

    logger.info(
        "Ingestion complete: total=%d inserted=%d skipped=%d",
        total_rows, inserted, skipped,
    )
    return {
        "batch_id": batch_id,
        "total_rows": total_rows,
        "inserted": inserted,
        "skipped": skipped,
        "source": source_label,
    }


def ingest_dataframe(
    df: pd.DataFrame,
    engine: Optional[Engine] = None,
    batch_id: Optional[str] = None,
    source_label: str = "dataframe",
) -> dict:
    """Ingest a pre-loaded DataFrame (useful for testing / API ingestion)."""
    engine = engine or get_engine()
    batch_id = batch_id or str(uuid.uuid4())[:8]
    df = _preprocess_chunk(df, batch_id, source_label)
    inserted, skipped = _upsert_chunk(df, engine)
    return {"batch_id": batch_id, "inserted": inserted, "skipped": skipped}


def get_training_data(
    engine: Optional[Engine] = None,
    days_lookback: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve labelled records from the raw table for training.
    Optionally restricts to recent data via days_lookback.
    """
    engine = engine or get_engine()
    where_clause = "WHERE target IS NOT NULL"
    if days_lookback:
        where_clause += (
            f" AND ingested_at >= NOW() - INTERVAL '{days_lookback} days'"
        )

    query = f"""
        SELECT *
        FROM {SCHEMA}.{RAW_TABLE}
        {where_clause}
        ORDER BY ingested_at DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    logger.info("Retrieved %d labelled records for training", len(df))
    return df


def get_data_stats(engine: Optional[Engine] = None) -> dict:
    """Return basic statistics about the raw data table."""
    engine = engine or get_engine()
    query = f"""
        SELECT
            COUNT(*)                                  AS total_rows,
            SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS default_count,
            SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) AS repaid_count,
            SUM(CASE WHEN target IS NULL THEN 1 ELSE 0 END) AS unlabelled_count,
            MIN(ingested_at)                          AS earliest_record,
            MAX(ingested_at)                          AS latest_record,
            COUNT(DISTINCT data_batch_id)             AS batch_count
        FROM {SCHEMA}.{RAW_TABLE}
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).mappings().one()
    return dict(result)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _preprocess_chunk(
    df: pd.DataFrame, batch_id: str, source_label: str
) -> pd.DataFrame:
    """Rename columns, add metadata, sanitise values."""
    # Rename only columns that exist in the mapping
    rename_map = {k: v for k, v in RAW_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only columns that exist in the DB table
    db_cols = list(RAW_COLUMN_MAP.values())
    df = df[[c for c in db_cols if c in df.columns]]

    # Fix DAYS_EMPLOYED anomaly (365243 = unemployed sentinel in Home Credit)
    if "days_employed" in df.columns:
        df["days_employed"] = df["days_employed"].replace(365243, None)

    # Metadata columns
    df["data_batch_id"] = batch_id
    df["source_file"] = source_label
    df["ingested_at"] = datetime.utcnow()

    return df


def _upsert_chunk(df: pd.DataFrame, engine: Engine) -> tuple[int, int]:
    """
    Upsert a chunk into raw_applications.
    Returns (inserted, skipped).
    """
    if df.empty:
        return 0, 0

    if "sk_id_curr" not in df.columns:
        logger.warning("Chunk missing sk_id_curr — skipping")
        return 0, len(df)

    inserted = upsert_dataframe(
        df=df,
        table=RAW_TABLE,
        schema=SCHEMA,
        conflict_column="sk_id_curr",
        engine=engine,
    )
    skipped = len(df) - inserted
    return inserted, skipped


def _make_batch_id(file_path: Path) -> str:
    """Derive a stable batch ID from filename + modification time."""
    stat = file_path.stat()
    key = f"{file_path.name}_{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
