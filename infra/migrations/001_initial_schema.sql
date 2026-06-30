-- infra/migrations/001_initial_schema.sql
-- Complete PostgreSQL schema for the Loan Default Prediction Pipeline

CREATE SCHEMA IF NOT EXISTS loan_data;

-- ─────────────────────────────────────────────
-- RAW DATA LAYER
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_data.raw_applications (
    sk_id_curr          BIGINT PRIMARY KEY,
    target              SMALLINT,              -- 1 = default, 0 = repaid
    name_contract_type  VARCHAR(50),
    code_gender         VARCHAR(10),
    flag_own_car        VARCHAR(5),
    flag_own_realty     VARCHAR(5),
    cnt_children        INTEGER,
    amt_income_total    NUMERIC(18, 2),
    amt_credit          NUMERIC(18, 2),
    amt_annuity         NUMERIC(18, 2),
    amt_goods_price     NUMERIC(18, 2),
    name_type_suite     VARCHAR(100),
    name_income_type    VARCHAR(100),
    name_education_type VARCHAR(100),
    name_family_status  VARCHAR(100),
    name_housing_type   VARCHAR(100),
    region_population_relative NUMERIC(10, 8),
    days_birth          INTEGER,
    days_employed       INTEGER,
    days_registration   NUMERIC(10, 3),
    days_id_publish     INTEGER,
    own_car_age         NUMERIC(10, 3),
    flag_mobil          SMALLINT,
    flag_emp_phone      SMALLINT,
    flag_work_phone     SMALLINT,
    flag_cont_mobile    SMALLINT,
    flag_phone          SMALLINT,
    flag_email          SMALLINT,
    occupation_type     VARCHAR(100),
    cnt_fam_members     NUMERIC(5, 1),
    region_rating_client        SMALLINT,
    region_rating_client_w_city SMALLINT,
    weekday_appr_process_start  VARCHAR(20),
    hour_appr_process_start     SMALLINT,
    reg_region_not_live_region  SMALLINT,
    reg_region_not_work_region  SMALLINT,
    live_region_not_work_region SMALLINT,
    reg_city_not_live_city      SMALLINT,
    reg_city_not_work_city      SMALLINT,
    live_city_not_work_city     SMALLINT,
    organization_type  VARCHAR(100),
    ext_source_1       NUMERIC(10, 8),
    ext_source_2       NUMERIC(10, 8),
    ext_source_3       NUMERIC(10, 8),
    -- Metadata
    ingested_at        TIMESTAMPTZ DEFAULT NOW(),
    data_batch_id      VARCHAR(50),
    source_file        VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_raw_app_target
    ON loan_data.raw_applications(target);
CREATE INDEX IF NOT EXISTS idx_raw_app_ingested
    ON loan_data.raw_applications(ingested_at);
CREATE INDEX IF NOT EXISTS idx_raw_app_batch
    ON loan_data.raw_applications(data_batch_id);


-- ─────────────────────────────────────────────
-- ENGINEERED FEATURES TABLE
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_data.engineered_features (
    sk_id_curr            BIGINT PRIMARY KEY REFERENCES loan_data.raw_applications(sk_id_curr),
    target                SMALLINT,

    -- Credit ratios
    credit_income_ratio   NUMERIC(10, 4),
    annuity_income_ratio  NUMERIC(10, 4),
    credit_term_months    NUMERIC(10, 4),
    goods_credit_ratio    NUMERIC(10, 4),

    -- Age / employment features
    age_years             NUMERIC(6, 2),
    employment_years      NUMERIC(6, 2),
    employed_to_age_ratio NUMERIC(6, 4),

    -- External scores composite
    ext_source_mean       NUMERIC(10, 8),
    ext_source_std        NUMERIC(10, 8),
    ext_source_missing_cnt SMALLINT,

    -- Document flags aggregates
    docs_provided_count   SMALLINT,
    family_income_per_capita NUMERIC(18, 2),

    -- Region mismatch flags
    total_region_mismatches SMALLINT,

    -- One-hot encoded categoricals (top categories)
    gender_m              SMALLINT,
    contract_type_cash    SMALLINT,
    income_type_working   SMALLINT,
    education_higher      SMALLINT,
    housing_type_house    SMALLINT,
    family_status_married SMALLINT,

    feature_version       VARCHAR(20) DEFAULT '1.0',
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eng_features_target
    ON loan_data.engineered_features(target);
CREATE INDEX IF NOT EXISTS idx_eng_features_created
    ON loan_data.engineered_features(created_at);


-- ─────────────────────────────────────────────
-- PREDICTION LOG
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_data.prediction_log (
    prediction_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sk_id_curr          BIGINT,
    model_version       VARCHAR(100) NOT NULL,
    model_run_id        VARCHAR(100),
    prediction_proba    NUMERIC(6, 4) NOT NULL,
    prediction_label    SMALLINT NOT NULL,
    threshold_used      NUMERIC(6, 4) DEFAULT 0.5,
    actual_label        SMALLINT,              -- filled in retrospectively
    request_id          VARCHAR(100),
    api_version         VARCHAR(20),
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    feedback_at         TIMESTAMPTZ,
    metadata            JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pred_log_predicted_at
    ON loan_data.prediction_log(predicted_at);
CREATE INDEX IF NOT EXISTS idx_pred_log_model_version
    ON loan_data.prediction_log(model_version);
CREATE INDEX IF NOT EXISTS idx_pred_log_sk_id
    ON loan_data.prediction_log(sk_id_curr);


-- ─────────────────────────────────────────────
-- DRIFT MONITORING RESULTS
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_data.drift_reports (
    report_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type         VARCHAR(50) NOT NULL,   -- 'data_drift' | 'prediction_drift' | 'performance'
    model_version       VARCHAR(100),
    reference_start     DATE NOT NULL,
    reference_end       DATE NOT NULL,
    current_start       DATE NOT NULL,
    current_end         DATE NOT NULL,
    overall_drift_score NUMERIC(8, 6),
    is_drift_detected   BOOLEAN NOT NULL,
    threshold_used      NUMERIC(8, 6),
    feature_drift_json  JSONB,                  -- per-feature PSI / JS values
    performance_json    JSONB,                  -- AUC, F1, etc.
    report_html_path    TEXT,
    retrain_triggered   BOOLEAN DEFAULT FALSE,
    dag_run_id          VARCHAR(200),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_created_at
    ON loan_data.drift_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_model_version
    ON loan_data.drift_reports(model_version);


-- ─────────────────────────────────────────────
-- PIPELINE RUN AUDIT LOG
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS loan_data.pipeline_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id              VARCHAR(200) NOT NULL,
    dag_run_id          VARCHAR(200),
    task_id             VARCHAR(200),
    status              VARCHAR(50) NOT NULL,   -- 'started' | 'success' | 'failed'
    training_rows       INTEGER,
    mlflow_run_id       VARCHAR(100),
    model_version       VARCHAR(100),
    auc_roc             NUMERIC(8, 6),
    promoted_to_prod    BOOLEAN DEFAULT FALSE,
    error_message       TEXT,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    duration_seconds    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_dag_id
    ON loan_data.pipeline_runs(dag_id, started_at DESC);


-- ─────────────────────────────────────────────
-- HELPER VIEWS
-- ─────────────────────────────────────────────

CREATE OR REPLACE VIEW loan_data.v_recent_predictions AS
SELECT
    p.predicted_at::DATE AS prediction_date,
    p.model_version,
    COUNT(*)             AS total_predictions,
    AVG(p.prediction_proba) AS avg_default_probability,
    SUM(p.prediction_label) AS predicted_defaults,
    SUM(CASE WHEN p.actual_label IS NOT NULL THEN 1 ELSE 0 END) AS labelled_count,
    AVG(CASE WHEN p.actual_label IS NOT NULL
             THEN ABS(p.actual_label - p.prediction_proba) END) AS mae
FROM loan_data.prediction_log p
WHERE p.predicted_at >= NOW() - INTERVAL '30 days'
GROUP BY 1, 2
ORDER BY 1 DESC;


CREATE OR REPLACE VIEW loan_data.v_model_performance_summary AS
SELECT
    pr.model_version,
    pr.auc_roc,
    pr.training_rows,
    pr.promoted_to_prod,
    pr.started_at AS trained_at,
    dr.overall_drift_score AS last_drift_score,
    dr.is_drift_detected,
    dr.created_at AS last_drift_check
FROM loan_data.pipeline_runs pr
LEFT JOIN LATERAL (
    SELECT overall_drift_score, is_drift_detected, created_at
    FROM loan_data.drift_reports
    WHERE model_version = pr.model_version
    ORDER BY created_at DESC
    LIMIT 1
) dr ON TRUE
WHERE pr.status = 'success' AND pr.promoted_to_prod = TRUE
ORDER BY pr.started_at DESC;
