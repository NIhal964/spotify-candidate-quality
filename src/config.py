# src/config.py

from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# =========================
# File names
# =========================
RAW_DATA_FILE = RAW_DATA_DIR / "SpotifyFeatures.csv"
MODEL_FILE = MODEL_DIR / "model.pkl"

# =========================
# Target definition
# =========================
TARGET_COL = "high_skip_risk_proxy"
RISK_PERCENTILE_THRESHOLD = 0.25

# =========================
# Train / validation split
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# Modeling choices
# =========================
BASELINE_MODEL = "logistic_regression"
# Headline metric is Lift at top 5% (locked). Secondary metrics at 10% and 15%.
PRIMARY_METRIC = "lift_at_5pct"
HEADLINE_K = 0.05
SECONDARY_KS = [0.10, 0.15]
CRITICAL_CLASS = 1  # high-risk class
# =========================
# Ensure directories exist
# =========================
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Inference
INFERENCE_THRESHOLD = 0.5  # can be tuned based on recall/precision trade-off
