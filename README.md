# Loan Default Prediction Pipeline

An end-to-end MLOps pipeline for predicting loan defaults with automated retraining, drift detection, and CI/CD.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Kaggle Home Credit Dataset → PostgreSQL (raw + feature store)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    AIRFLOW ORCHESTRATION                         │
│  Schedule → Ingest → Validate (GE) → Feature Eng → Train →     │
│  Evaluate → Registry Transition → Deploy                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼──────┐  ┌────────▼───────┐  ┌──────▼─────────┐
│  MLflow        │  │  FastAPI       │  │  Evidently AI  │
│  Tracking +    │  │  Serving       │  │  Monitoring    │
│  Model Registry│  │  (Docker)      │  │  Dashboard     │
└────────────────┘  └────────────────┘  └──────┬─────────┘
                                               │ drift alert
                                    ┌──────────▼──────────┐
                                    │  Auto-trigger        │
                                    │  Retraining DAG      │
                                    └─────────────────────┘
```

## Stack

| Component | Technology |
|-----------|------------|
| Orchestration | Apache Airflow 2.8 |
| Data Validation | Great Expectations 0.18 |
| Training | XGBoost + Scikit-learn |
| Experiment Tracking | MLflow 2.10 |
| Model Registry | MLflow Model Registry |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently AI |
| Database | PostgreSQL 15 |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- PostgreSQL client

### 1. Clone and configure
```bash
git clone <repo>
cd loan-default-pipeline
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

### 2. Start infrastructure
```bash
docker-compose up -d postgres mlflow airflow-webserver airflow-scheduler
```

### 3. Ingest initial data
```bash
python scripts/ingest_initial_data.py --source kaggle --dataset home-credit
```

### 4. Trigger first training run
```bash
python scripts/trigger_dag.py --dag loan_default_training
```

### 5. Start serving
```bash
docker-compose up -d api
```

### 6. View dashboards
- Airflow:  http://localhost:8080
- MLflow:   http://localhost:5000
- API docs: http://localhost:8000/docs
- Monitoring: http://localhost:8001

## Project Structure

```
loan-default-pipeline/
├── dags/                    # Airflow DAGs
│   ├── loan_default_training_dag.py
│   └── drift_monitoring_dag.py
├── src/
│   ├── data/               # Data ingestion & validation
│   ├── training/           # Feature engineering & model training
│   ├── serving/            # FastAPI application
│   └── monitoring/         # Evidently drift detection
├── tests/
│   ├── unit/
│   └── integration/
├── docker/                 # Dockerfiles
├── infra/                  # DB migrations, GE config
├── config/                 # YAML configuration
├── scripts/                # Helper scripts
└── .github/workflows/      # CI/CD pipelines
```

## Configuration

All runtime parameters live in `config/config.yaml`:

```yaml
model:
  drift_threshold: 0.15      # PSI threshold triggering retraining
  min_auc_roc: 0.75          # Minimum AUC-ROC for production promotion
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
```

## Monitoring Thresholds

| Metric | Warning | Critical (triggers retrain) |
|--------|---------|------------------------------|
| PSI (feature drift) | 0.10 | 0.20 |
| JS Divergence (pred drift) | 0.05 | 0.10 |
| AUC-ROC degradation | -0.02 | -0.05 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/health` | Health check |
| GET | `/model/info` | Current model metadata |
| POST | `/monitoring/report` | Generate drift report |
