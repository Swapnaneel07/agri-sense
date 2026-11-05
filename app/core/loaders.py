"""
AgriSense AI - Cached Model & Metadata Loaders
Uses Streamlit cache_resource so models are loaded once into memory.
If any model file is absent from models/, returns None gracefully.
"""

import json
from pathlib import Path

try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from . import config


@cache_decorator
def load_yield_schema():
    """Loads feature schema dictionary driving the dynamic yield UI form."""
    if config.YIELD_SCHEMA.exists():
        try:
            with open(config.YIELD_SCHEMA, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback schema
    return {
        "features": [
            {"name": "Area", "label": "Country / Territory", "type": "select", "options": ["India", "Italy", "United States"], "default": "India"},
            {"name": "Item", "label": "Crop Species", "type": "select", "options": ["Maize", "Potatoes", "Wheat", "Rice, paddy"], "default": "Maize"},
            {"name": "Year", "label": "Harvest Year", "type": "int", "min": 1990, "max": 2025, "default": 2013},
            {"name": "average_rain_fall_mm_per_year", "label": "Annual Rainfall (mm)", "type": "float", "min": 50.0, "max": 3500.0, "default": 1120.0},
            {"name": "pesticides_tonnes", "label": "Pesticide Usage (tonnes)", "type": "float", "min": 0.0, "max": 50000.0, "default": 1200.0}
        ],
        "target": "hg/ha_yield",
        "units": "hg/ha"
    }


@cache_decorator
def load_yield_bundle():
    """Loads Keras model and preprocessor bundle from models/."""
    keras_path = config.MODELS_DIR / "yield_mlp.keras"
    preproc_path = config.MODELS_DIR / "yield_preprocessor.joblib"

    bundle = None
    if JOBLIB_AVAILABLE and preproc_path.exists():
        try:
            bundle = joblib.load(preproc_path)
        except Exception:
            bundle = None

    model = None
    if keras_path.exists():
        try:
            import tensorflow as tf
            from tensorflow import keras
            model = keras.models.load_model(keras_path, compile=False)
        except Exception:
            model = None

    return model, bundle


@cache_decorator
def load_yield_model():
    """Backward compatibility wrapper."""
    model, bundle = load_yield_bundle()
    return model, bundle


@cache_decorator
def load_yield_xgb():
    """Loads baseline XGBoost model if present."""
    if JOBLIB_AVAILABLE and config.YIELD_XGB.exists():
        try:
            return joblib.load(config.YIELD_XGB)
        except Exception:
            pass
    return None


@cache_decorator
def load_irrigation_bundle():
    """Loads irrigation model weights, scaler, and metadata."""
    meta = None
    if config.IRRIGATION_META.exists():
        try:
            with open(config.IRRIGATION_META, "r") as f:
                meta = json.load(f)
        except Exception:
            pass

    scaler = None
    if JOBLIB_AVAILABLE and config.IRRIGATION_SCALER.exists():
        try:
            scaler = joblib.load(config.IRRIGATION_SCALER)
        except Exception:
            pass

    model = None
    if TORCH_AVAILABLE and config.IRRIGATION_MODEL_PT.exists():
        try:
            ckpt = torch.load(config.IRRIGATION_MODEL_PT, map_location="cpu", weights_only=True)
            model = ckpt
        except Exception:
            pass

    return {
        "model": model,
        "scaler": scaler,
        "meta": meta,
        "is_ready": (model is not None and scaler is not None)
    }


@cache_decorator
def load_leaf_bundle():
    """Loads leaf classes and model checkpoint."""
    classes = []
    if config.LEAF_CLASSES.exists():
        try:
            with open(config.LEAF_CLASSES, "r") as f:
                data = json.load(f)
                classes = data.get("classes", [])
        except Exception:
            pass

    has_pte = config.LEAF_MODEL_PTE.exists()
    has_pt = config.LEAF_MODEL_PT.exists()

    return {
        "classes": classes,
        "has_pte": has_pte,
        "has_pt": has_pt,
        "model_badge": "ExecuTorch INT8 (3.5 MB)" if has_pte else "MobileNetV2 FP32"
    }


@cache_decorator
def load_gan_bundle():
    """Loads rare classes list and checks if generator is present."""
    classes = []
    if config.GAN_CLASSES.exists():
        try:
            with open(config.GAN_CLASSES, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
        except Exception:
            pass
    if not classes:
        classes = ["Potato___healthy", "Apple___Cedar_apple_rust", "Peach___healthy", "Raspberry___healthy", "Tomato___Tomato_mosaic_virus"]

    has_gen = config.GAN_GENERATOR.exists()
    has_script = config.GAN_GENERATE_SCRIPT.exists()
    return {
        "classes": classes,
        "has_generator": has_gen,
        "has_script": has_script
    }
