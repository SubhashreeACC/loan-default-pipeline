#!/usr/bin/env python
# scripts/trigger_dag.py
"""
Manually trigger an Airflow DAG run via the REST API.

Usage:
    python scripts/trigger_dag.py --dag loan_default_training
    python scripts/trigger_dag.py --dag loan_drift_monitoring --conf '{"force": true}'
"""

import argparse
import json
import os
import sys

import requests

DEFAULT_AIRFLOW_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
DEFAULT_USER = os.getenv("AIRFLOW_USER", "admin")
DEFAULT_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")


def main():
    parser = argparse.ArgumentParser(description="Trigger an Airflow DAG run")
    parser.add_argument("--dag", required=True, help="DAG ID to trigger")
    parser.add_argument("--conf", type=str, default="{}", help="JSON config payload")
    parser.add_argument("--airflow-url", default=DEFAULT_AIRFLOW_URL)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    args = parser.parse_args()

    try:
        conf = json.loads(args.conf)
    except json.JSONDecodeError:
        print(f"Invalid JSON for --conf: {args.conf}")
        sys.exit(1)

    url = f"{args.airflow_url}/api/v1/dags/{args.dag}/dagRuns"
    payload = {"conf": conf}

    print(f"Triggering DAG '{args.dag}' at {url}")
    resp = requests.post(
        url,
        json=payload,
        auth=(args.user, args.password),
        headers={"Content-Type": "application/json"},
        timeout=15,
    )

    if resp.status_code in (200, 201):
        body = resp.json()
        print(f"✅ DAG run triggered: {body.get('dag_run_id')}")
        print(f"   State: {body.get('state')}")
    else:
        print(f"❌ Failed to trigger DAG: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
