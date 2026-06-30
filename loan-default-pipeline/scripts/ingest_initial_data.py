#!/usr/bin/env python
# scripts/ingest_initial_data.py
"""
One-off script to bulk-load the Home Credit Default Risk dataset
(application_train.csv from Kaggle) into PostgreSQL.

Usage:
    python scripts/ingest_initial_data.py --file /path/to/application_train.csv
    python scripts/ingest_initial_data.py --source kaggle --dataset home-credit
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.ingestion import ingest_csv, get_data_stats
from src.utils.db import get_engine, run_migration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Bulk-load initial loan default dataset")
    parser.add_argument("--file", type=str, help="Path to local CSV file")
    parser.add_argument(
        "--source", type=str, choices=["local", "kaggle"], default="local",
        help="Data source type",
    )
    parser.add_argument("--dataset", type=str, default="home-credit-default-risk")
    parser.add_argument(
        "--apply-migrations", action="store_true",
        help="Apply DB schema migrations before ingesting",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    if args.apply_migrations:
        logger.info("Applying database migrations…")
        migration_file = Path(__file__).parents[1] / "infra/migrations/001_initial_schema.sql"
        run_migration(str(migration_file))

    if args.source == "kaggle":
        if not args.file:
            logger.error(
                "Kaggle auto-download not bundled in this script for security reasons. "
                "Download 'application_train.csv' from "
                "https://www.kaggle.com/competitions/home-credit-default-risk/data "
                "and pass --file /path/to/application_train.csv"
            )
            sys.exit(1)

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("Starting initial ingestion from %s", file_path)
    stats = ingest_csv(file_path, batch_size=args.batch_size)

    logger.info("=" * 50)
    logger.info("Ingestion Summary")
    logger.info("=" * 50)
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    db_stats = get_data_stats()
    logger.info("-" * 50)
    logger.info("Database Totals")
    logger.info("-" * 50)
    for k, v in db_stats.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
