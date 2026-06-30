#!/usr/bin/env python
# scripts/run_migrations.py
"""
Apply all SQL migration files in infra/migrations/ in order.

Usage:
    python scripts/run_migrations.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.db import run_migration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parents[1] / "infra" / "migrations"


def main():
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        logger.warning("No migration files found in %s", MIGRATIONS_DIR)
        return

    for f in migration_files:
        logger.info("Applying migration: %s", f.name)
        run_migration(str(f))
        logger.info("  ✅ %s applied", f.name)

    logger.info("All %d migrations applied successfully.", len(migration_files))


if __name__ == "__main__":
    main()
