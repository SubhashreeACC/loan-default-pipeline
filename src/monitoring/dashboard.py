# src/monitoring/dashboard.py
"""
Streamlit monitoring dashboard for the Loan Default pipeline.
Shows drift trends, model performance, and prediction statistics.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy as sa
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import get_config
from src.utils.db import get_engine

cfg = get_config()
SCHEMA = cfg["database"]["schema"]

st.set_page_config(
    page_title="Loan Default Monitor",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_drift_history(days: int = 30) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT report_type, overall_drift_score, is_drift_detected,
               retrain_triggered, created_at
        FROM {SCHEMA}.drift_reports
        WHERE created_at >= NOW() - INTERVAL '{days} days'
        ORDER BY created_at
    """
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)


@st.cache_data(ttl=300)
def load_prediction_stats(days: int = 30) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT
            predicted_at::DATE AS date,
            model_version,
            COUNT(*) AS total,
            AVG(prediction_proba) AS avg_proba,
            SUM(prediction_label) AS predicted_defaults,
            100.0 * SUM(prediction_label) / NULLIF(COUNT(*), 0) AS default_rate_pct
        FROM {SCHEMA}.prediction_log
        WHERE predicted_at >= NOW() - INTERVAL '{days} days'
        GROUP BY 1, 2
        ORDER BY 1
    """
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)


@st.cache_data(ttl=300)
def load_pipeline_runs() -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT dag_id, model_version, auc_roc, promoted_to_prod,
               started_at, duration_seconds, status
        FROM {SCHEMA}.pipeline_runs
        ORDER BY started_at DESC
        LIMIT 20
    """
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)


# ─────────────────────────────────────────────
# Dashboard layout
# ─────────────────────────────────────────────

st.title("🏦 Loan Default Prediction — MLOps Monitor")
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Sidebar controls
st.sidebar.header("Controls")
lookback_days = st.sidebar.slider("Lookback (days)", 7, 90, 30)
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ─── KPI row ─────────────────────────────────
drift_df = load_drift_history(lookback_days)
pred_df = load_prediction_stats(lookback_days)
runs_df = load_pipeline_runs()

col1, col2, col3, col4 = st.columns(4)

latest_drift = drift_df["overall_drift_score"].iloc[-1] if not drift_df.empty else 0.0
drift_detected = drift_df["is_drift_detected"].any() if not drift_df.empty else False
retrain_count = int(drift_df["retrain_triggered"].sum()) if not drift_df.empty else 0
total_preds = int(pred_df["total"].sum()) if not pred_df.empty else 0

with col1:
    st.metric("Latest Drift Score", f"{latest_drift:.4f}",
              delta="⚠️ DRIFT" if drift_detected else "✅ OK")
with col2:
    st.metric("Total Predictions", f"{total_preds:,}")
with col3:
    avg_default_rate = pred_df["default_rate_pct"].mean() if not pred_df.empty else 0
    st.metric("Avg Default Rate", f"{avg_default_rate:.1f}%")
with col4:
    st.metric("Retrains Triggered", retrain_count)

st.divider()

# ─── Drift over time ─────────────────────────
st.subheader("📈 Drift Score Over Time")
if not drift_df.empty:
    fig = px.line(
        drift_df,
        x="created_at",
        y="overall_drift_score",
        color="report_type",
        markers=True,
        title="Data & Prediction Drift",
    )
    mon_cfg = cfg["monitoring"]
    fig.add_hline(y=mon_cfg["psi_warning"], line_dash="dot",
                  line_color="orange", annotation_text="Warning")
    fig.add_hline(y=mon_cfg["psi_critical"], line_dash="dash",
                  line_color="red", annotation_text="Critical")
    # Mark retrain triggers
    retrain_events = drift_df[drift_df["retrain_triggered"]]
    if not retrain_events.empty:
        fig.add_scatter(
            x=retrain_events["created_at"],
            y=retrain_events["overall_drift_score"],
            mode="markers",
            marker=dict(symbol="x", size=12, color="red"),
            name="Retrain Triggered",
        )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No drift reports available yet.")

# ─── Prediction trends ───────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Daily Predictions")
    if not pred_df.empty:
        fig = px.bar(pred_df, x="date", y="total", color="model_version",
                     title="Prediction Volume")
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("⚠️ Default Rate Trend")
    if not pred_df.empty:
        fig = px.line(pred_df, x="date", y="default_rate_pct",
                      color="model_version",
                      title="Predicted Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

# ─── Pipeline runs ───────────────────────────
st.subheader("🚀 Recent Pipeline Runs")
if not runs_df.empty:
    st.dataframe(
        runs_df.style.applymap(
            lambda v: "background-color: #d4edda" if v else "background-color: #f8d7da",
            subset=["promoted_to_prod"]
        ),
        use_container_width=True,
    )

    # AUC trend
    prod_runs = runs_df[runs_df["promoted_to_prod"]].sort_values("started_at")
    if not prod_runs.empty:
        fig = px.line(prod_runs, x="started_at", y="auc_roc",
                      markers=True, title="Production Model AUC-ROC Over Time",
                      labels={"auc_roc": "AUC-ROC", "started_at": "Trained At"})
        fig.add_hline(y=cfg["model"]["validation"]["min_auc_roc"],
                      line_dash="dot", line_color="red",
                      annotation_text="Minimum threshold")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pipeline runs recorded yet.")

# ─── Footer ──────────────────────────────────
st.divider()
st.caption(
    "Monitoring powered by Evidently AI | "
    "Orchestrated by Apache Airflow | "
    "Models tracked in MLflow"
)
