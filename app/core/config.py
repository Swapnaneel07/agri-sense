"""
AgriSense AI - Core Configuration & Paths
Central repository configuration so changing any file path is a one-line edit.
"""

from pathlib import Path

# Base repository root (agri-sense/)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Directory Drop Zones
APP_DIR = ROOT_DIR / "app"
PAGES_DIR = APP_DIR / "pages"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

# Specific Model Contracts
YIELD_MODEL_PT = MODELS_DIR / "yield_mlp.pt"
YIELD_MODEL_KERAS = MODELS_DIR / "yield_mlp.keras"
YIELD_SCALER = MODELS_DIR / "yield_scaler.joblib"
YIELD_SCHEMA = MODELS_DIR / "yield_schema.json"
YIELD_XGB = MODELS_DIR / "yield_xgb.joblib"

IRRIGATION_MODEL_PT = MODELS_DIR / "irrigation_best.pt"
IRRIGATION_SCALER = MODELS_DIR / "irrigation_scaler.joblib"
IRRIGATION_META = MODELS_DIR / "irrigation_meta.json"

LEAF_MODEL_PTE = MODELS_DIR / "leaf_mobilenetv2.pte"
LEAF_MODEL_PT = MODELS_DIR / "leaf_mobilenetv2.pt"
LEAF_CLASSES = MODELS_DIR / "leaf_classes.json"

GAN_GENERATOR = MODELS_DIR / "generator.pt"
GAN_CLASSES = MODELS_DIR / "CLASSES.txt"
GAN_SAMPLES_DIR = MODELS_DIR / "samples"
GAN_GENERATE_SCRIPT = MODELS_DIR / "generate.py"

# Results Files
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_NOTES = RESULTS_DIR / "notes.md"
PER_CLASS_F1_PNG = RESULTS_DIR / "per_class_f1.png"
CONFUSION_RARE_PNG = RESULTS_DIR / "confusion_rare.png"

# Sample Data
SAMPLE_LEAVES_DIR = SAMPLE_DATA_DIR / "leaves"
SAMPLE_IRRIGATION_CSV = SAMPLE_DATA_DIR / "irrigation" / "sample_window.csv"

