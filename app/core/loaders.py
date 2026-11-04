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
            with open(config.YIELD_SCHEMA, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback schema
    return {
        "features": [
            {"name": "nitrogen", "label": "Nitrogen (N) [mg/kg]", "type": "float", "min": 0.0, "max": 140.0, "default": 85.0, "step": 1.0},
            {"name": "phosphorus", "label": "Phosphorus (P) [mg/kg]", "type": "float", "min": 0.0, "max": 120.0, "default": 52.0, "step": 1.0},
            {"name": "potassium", "label": "Potassium (K) [mg/kg]", "type": "float", "min": 0.0, "max": 120.0, "default": 48.0, "step": 1.0},
            {"name": "temperature", "label": "Temperature (°C)", "type": "float", "min": 10.0, "max": 45.0, "default": 26.5, "step": 0.1},
            {"name": "humidity", "label": "Relative Humidity (%)", "type": "float", "min": 10.0, "max": 100.0, "default": 78.0, "step": 0.5},
            {"name": "ph", "label": "Soil pH Level", "type": "float", "min": 3.5, "max": 9.5, "default": 6.5, "step": 0.1},
            {"name": "rainfall", "label": "Annual Rainfall (mm)", "type": "float", "min": 100.0, "max": 2500.0, "default": 1120.0, "step": 10.0}
        ],
        "target": "crop_yield_tons_ha",
        "units": "metric tons / hectare"
    }


@cache_decorator
def load_yield_model():
    """Loads trained MLP model (PyTorch or Keras) and scaler if present."""
    if not JOBLIB_AVAILABLE:
        return None, None
    scaler = None
    if config.YIELD_SCALER.exists():
        try:
            scaler = joblib.load(config.YIELD_SCALER)
        except Exception:
            pass

    if TORCH_AVAILABLE and config.YIELD_MODEL_PT.exists():
        try:
            # Load PyTorch checkpoint
            ckpt = torch.load(config.YIELD_MODEL_PT, map_location="cpu", weights_only=True)
            return ckpt, scaler
        except Exception:
            pass
    return None, scaler


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
            # Load weights
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
        classes = ["Potato___Late_blight", "Tomato___Target_Spot", "Pepper,_bell___Bacterial_spot", "Apple___Cedar_apple_rust"]

    has_gen = config.GAN_GENERATOR.exists()
    has_script = config.GAN_GENERATE_SCRIPT.exists()
    return {
        "classes": classes,
        "has_generator": has_gen,
        "has_script": has_script
    }
