# src/utils/db.py
"""SQLAlchemy engine factory and common DB helpers."""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.config import get_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_config()["database"]
    url = sa.engine.URL.create(
        drivername="postgresql+psycopg2",
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["name"],
        username=cfg["user"],
        password=cfg["password"],
    )
    engine = sa.create_engine(
        url,
        pool_size=int(cfg.get("pool_size", 5)),
        max_overflow=int(cfg.get("max_overflow", 10)),
        pool_pre_ping=True,
    )
    logger.debug("Database engine created: %s", cfg["host"])
    return engine


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    schema: str,
    conflict_column: str,
    engine: Engine | None = None,
) -> int:
    """
    Upsert a DataFrame into a PostgreSQL table using INSERT ON CONFLICT DO NOTHING.
    Returns number of rows actually inserted.
    """
    engine = engine or get_engine()
    if df.empty:
        return 0

    tmp_table = f"_tmp_{table}_{id(df)}"
    cols = ", ".join(f'"{c}"' for c in df.columns)
    placeholders = ", ".join(f":{c}" for c in df.columns)

    insert_sql = text(f"""
        INSERT INTO {schema}.{table} ({cols})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_column}) DO NOTHING
    """)

    inserted = 0
    with engine.begin() as conn:
        records = df.to_dict(orient="records")
        result = conn.execute(insert_sql, records)
        inserted = result.rowcount

    return inserted


def table_exists(table: str, schema: str, engine: Engine | None = None) -> bool:
    engine = engine or get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = :table)"
            ),
            {"schema": schema, "table": table},
        )
        return bool(result.scalar())


def run_migration(sql_file: str, engine: Engine | None = None) -> None:
    """Execute a SQL migration file."""
    engine = engine or get_engine()
    sql = open(sql_file).read()
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("Migration applied: %s", sql_file)
