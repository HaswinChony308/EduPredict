"""
config.py — Central Configuration for AAPT System
===================================================
All constants and settings are stored here so they can be
changed in one place without modifying the rest of the codebase.
"""

import os

# ──────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────
# Path to the SQLite database file.  We use a relative path
# so the project stays portable across machines.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "db", "aapt.db")

# ──────────────────────────────────────────────────────────
# MODEL PATHS
# ──────────────────────────────────────────────────────────
# Where trained models, scalers, and encoders are saved/loaded.
MODEL_PATH = os.path.join(BASE_DIR, "models", "aapt_model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
MODEL_BACKUP_PATH = os.path.join(BASE_DIR, "models", "aapt_model_backup.pkl")
MODEL_PREVIOUS_PATH = os.path.join(BASE_DIR, "models", "aapt_model_previous.pkl")

# ──────────────────────────────────────────────────────────
# DATA PATHS
# ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_students.csv")

# ──────────────────────────────────────────────────────────
# DRIFT DETECTION
# ──────────────────────────────────────────────────────────
# Z-score threshold: if |Z| > this value, we flag the student.
# A threshold of 2.0 means the student's engagement is more than
# 2 standard deviations away from their own baseline.
Z_SCORE_THRESHOLD = 2.0

# What fraction of early weeks to use as "baseline" for drift.
# 0.20 = first 20% of recorded engagement history.
BASELINE_PERIOD_FRACTION = 0.20

# ──────────────────────────────────────────────────────────
# RISK CLASSES
# ──────────────────────────────────────────────────────────
# The four performance categories predicted by our model.
# Order matters — index 0 = best, index 3 = worst.
RISK_CLASSES = ["Excelling", "On Track", "Moderate Risk", "High Risk"]

# Color codes for each risk level (used in dashboard UI)
RISK_COLORS = {
    "Excelling":     "#3b82f6",   # Blue
    "On Track":      "#22c55e",   # Green
    "Moderate Risk": "#f59e0b",   # Amber
    "High Risk":     "#ef4444",   # Red
    "Dynamic Risk":  "#a855f7",   # Purple (drift-based override)
}

# ──────────────────────────────────────────────────────────
# EMAIL / SMTP SETTINGS
# ──────────────────────────────────────────────────────────
# To send email alerts, replace these with your Gmail App Password.
# Go to: https://myaccount.google.com/apppasswords to generate one.
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465                       # SSL port for Gmail
SMTP_SENDER = "23ag1a0555@gmail.com"  # ← Replace with your Gmail
SMTP_PASSWORD = "16characterapp"  # ← Replace with App Password

# ──────────────────────────────────────────────────────────
# ADMIN SETTINGS
# ──────────────────────────────────────────────────────────
# Password for the admin retrain page. Change before deployment!
ADMIN_PASSWORD = "admin1729"

# ──────────────────────────────────────────────────────────
# ML HYPERPARAMETERS
# ──────────────────────────────────────────────────────────
# Random Forest parameters
RF_N_ESTIMATORS = 200        # Number of trees in the forest
RF_CRITERION = "entropy"     # Split quality metric (entropy = information gain)
RF_RANDOM_STATE = 42         # Fixed seed for reproducibility

# Gradient Boosting parameters
GBT_N_ESTIMATORS = 100       # Number of boosting rounds
GBT_LEARNING_RATE = 0.05     # How much each tree contributes (smaller = more conservative)
GBT_MAX_DEPTH = 4            # Maximum depth per tree (controls complexity)
GBT_RANDOM_STATE = 42

# Train-test split
TEST_SPLIT_SIZE = 0.20  # 20% of data held out for evaluation

# ──────────────────────────────────────────────────────────
# FEATURE LIST (must match training order)
# ──────────────────────────────────────────────────────────
# These are the features our model expects, in the order they
# were used during training.  Batch upload CSVs must have these columns.
EXPECTED_FEATURES = [
    "gender", "age_band", "region", "highest_education",
    "imd_band", "disability", "num_of_prev_attempts",
    "studied_credits", "weighted_avg_score", "sum_clicks",
    "avg_clicks_per_day", "days_active", "last_active_day",
    "click_trend"
]
