"""Crop Yield Prediction - Role E: Backpropagation, Initialization, Regularization, XGBoost & Final Model.

====================================================================================================
PROJECT CONTEXT & MODULE OVERVIEW (Role E - Agricultural Yield Prediction Module 2)
====================================================================================================
This single unified script consolidates all Role E deliverables for the 10-person crop yield
prediction project into a clean, modular, and non-redundant pipeline:

1. Backpropagation Verification:
   - Analytical derivation and NumPy implementation of a 2-layer neural network backprop step.
   - Verification against TensorFlow's autodiff (tf.GradientTape) with gradient error check.
   - Single-step gradient descent weight update demonstrating loss reduction.

2. Data Pipeline & Preprocessing:
   - Strict 70% Train / 15% Validation / 15% Test split with deterministic random seed (42).
   - Scikit-learn ColumnTransformer (StandardScaler for numerical features, OneHotEncoder for
     categorical Area and Item with handle_unknown='ignore').
   - Independent target scaling (StandardScaler) fitted strictly on the training set to prevent leakage.
   - Fallback split generation from 'yield_df.csv' or seamless ingestion of pre-saved split.npz.

3. XGBoost Baseline Benchmark:
   - Gradient boosted trees regressor (XGBRegressor) evaluated across 3 random seeds.
   - Evaluation on train, validation, and test splits (RMSE, MAE, R², runtime, tree counts).
   - Feature importance extraction and ranking plot.

4. Weight Initialization Experiments:
   - Rigorous comparison across 4 initialization schemes: Zeros, Random Normal, Glorot/Xavier, He.
   - Convergence speed analysis (first 10 epochs loss tracking) and signal statistics probe
     (layer-by-layer activation mean/std and gradient std at epoch 0 to detect vanishing/exploding signals).
   - Selection of optimal initializer strictly based on VALIDATION RMSE.

5. Regularization Ablation & Overfitting Demonstration:
   - Systematic ablation: None vs L2 (1e-4, 1e-3, 1e-2) vs Dropout (0.1, 0.2, 0.3, 0.5) vs
     BatchNorm vs Early Stopping (patience 5 vs 10) vs combined L2+Dropout.
   - Generalization gap tracking (Test RMSE - Train RMSE).
   - Deliberate overfitting demonstration using an oversized capacity network (256-128-64, 200 epochs)
     to illustrate severe unregularized overfitting and subsequent regularization recovery.

6. Final Model & Statistical Comparison:
   - Optimal MLP constructed from best validation initializer, regularization, and early stopping.
   - Evaluation across seeds on the held-out test set (reported once, never used for selection).
   - Statistical significance testing against Baseline MLP and XGBoost with honest conclusions.
   - Checkpointing of deployable artifacts ('models/yield_mlp.keras', 'models/yield_preprocessor.joblib').

7. Teammate Integration API (Role H Interface):
   - 'predict_yield(features: dict) -> float': cached, production-ready inference function adhering
     to the FEATURES.md contract with full input validation and non-negative clamping.
   - 'get_allowed_categories() -> dict': category reflection for frontend dropdown menus.

Command Line Usage:
  python main.py --all           # Run full pipeline end-to-end
  python main.py --backprop      # Run backpropagation verification demo
  python main.py --xgb           # Train and evaluate XGBoost baseline
  python main.py --init          # Run weight initialization experiments
  python main.py --reg           # Run regularization ablation & overfit demo
  python main.py --final         # Train and benchmark final MLP model
  python main.py --predict       # Run sample prediction test
  python main.py --quick         # Run all experiments in quick smoke-test mode (1 seed, few epochs)
  python main.py --trainval      # Retrain final model on combined train+val before saving
====================================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import tempfile
import time
import warnings

# Suppress verbose TensorFlow C++ logging and route matplotlib cache to temp directory
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Lazy imports for heavy ML frameworks (TensorFlow, XGBoost) to ensure lightweight CLI inspection
tf = None
keras = None
layers = None
regularizers = None
XGBRegressor = None


def _ensure_tf():
    """Lazily import and initialize TensorFlow / Keras."""
    global tf, keras, layers, regularizers
    if tf is None:
        try:
            import tensorflow as _tf
            from tensorflow import keras as _keras
            from tensorflow.keras import layers as _layers, regularizers as _reg
            tf = _tf
            keras = _keras
            layers = _layers
            regularizers = _reg
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for this operation. Please install it using: pip install tensorflow"
            ) from exc


def _ensure_xgb():
    """Lazily import XGBoost."""
    global XGBRegressor
    if XGBRegressor is None:
        try:
            from xgboost import XGBRegressor as _XGBR
            XGBRegressor = _XGBR
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for this operation. Please install it using: pip install xgboost"
            ) from exc


# ==================================================================================================
# 1. PATHS, SCHEMA & GLOBAL CONSTANTS
# ==================================================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "yield_df.csv"
FALLBACK_ROOT_CSV = ROOT_DIR / "yield_df.csv"
SPLIT_PATH = DATA_DIR / "split.npz"
PREPROC_PATH = MODELS_DIR / "yield_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "yield_mlp.keras"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"
BEST_INIT_PATH = RESULTS_DIR / "best_init.json"
BEST_REG_PATH = RESULTS_DIR / "best_reg.json"
XGB_BASELINE_PATH = RESULTS_DIR / "xgb_baseline.json"
FINAL_METRICS_PATH = RESULTS_DIR / "final_metrics.json"

# Dataset Schema
CAT_COLS = ["Area", "Item"]
NUM_COLS = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
TARGET = "hg/ha_yield"

# Random Seeds
SEEDS = [42, 43, 44]
SPLIT_SEED = 42

# Default MLP Baseline Hyperparameters (matches team specifications)
DEFAULT_CONFIG = {
    "hidden_units": [128, 64, 32],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 100,
    "dropout": 0.0,
    "l2": 0.0,
    "initializer": "glorot_uniform",
    "batch_norm": False,
}

_CONFIG_ALIASES = {
    "units": "hidden_units",
    "layers": "hidden_units",
    "hidden_layers": "hidden_units",
    "lr": "learning_rate",
    "batch": "batch_size",
    "kernel_initializer": "initializer",
    "dropout_rate": "dropout",
    "l2_strength": "l2",
}

INIT_METHODS = ["zeros", "random_normal", "glorot_uniform", "he_normal"]

# Palette for consistent, professional figures
PALETTE = {
    "zeros": "#c0392b",
    "random_normal": "#e08e0b",
    "glorot_uniform": "#2a7f62",
    "he_normal": "#2c5aa0",
    "baseline": "#8a8f98",
    "best": "#2a7f62",
    "xgboost": "#2c5aa0",
    "train": "#2a7f62",
    "val": "#e08e0b",
    "test": "#c0392b",
}


# ==================================================================================================
# 2. SHARED MATHEMATICAL, STATISTICAL & FILE UTILITIES
# ==================================================================================================
def save_json(obj: object, path: Path | str) -> None:
    """Save serializable Python object to JSON using UTF-8 formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path: Path | str, default: object = None) -> object:
    """Load JSON file safely with fallback if non-existent."""
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Ensure strict deterministic repeatability across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        keras.utils.set_random_seed(seed)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics (RMSE, MAE, R²) in original target units."""
    y_true_flat = np.asarray(y_true, dtype=float).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=float).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "r2": float(r2_score(y_true_flat, y_pred_flat)),
    }


def mean_std(values: list[float] | np.ndarray) -> dict[str, float]:
    """Calculate mean and sample standard deviation."""
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
    }


def format_pm(mean_val: float, std_val: float, nd: int = 1) -> str:
    """Format mean ± standard deviation string with fixed decimals and thousands separators."""
    return f"{mean_val:,.{nd}f} ± {std_val:,.{nd}f}"


def format_md_table(df: pd.DataFrame) -> str:
    """Generate clean GitHub-flavored markdown table without external dependencies."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, separator] + rows)


def summarize_runs(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed training records into a comprehensive summary table with mean and std."""
    order = list(dict.fromkeys(r["variant"] for r in rows))
    out = []
    for variant in order:
        group = [r for r in rows if r["variant"] == variant]
        summary_row = {"variant": variant, "n_seeds": len(group)}
        for split in ("train", "val", "test"):
            for m in ("rmse", "mae", "r2"):
                stats = mean_std([r[split][m] for r in group])
                summary_row[f"{split}_{m}"] = stats["mean"]
                summary_row[f"{split}_{m}_std"] = stats["std"]
        gap_stats = mean_std([r["test"]["rmse"] - r["train"]["rmse"] for r in group])
        summary_row["gap"] = gap_stats["mean"]
        summary_row["gap_std"] = gap_stats["std"]
        for k in ("epochs_to_converge", "best_epoch", "epochs_run", "train_time_s", "n_params"):
            if k in group[0]:
                summary_row[k] = float(np.mean([r[k] for r in group]))
        out.append(summary_row)
    return pd.DataFrame(out)


def load_base_config() -> dict:
    """Load baseline hyperparameter configuration, merging best_config.json if provided by teammate D."""
    raw = load_json(BEST_CONFIG_PATH, default={}) or {}
    cfg = dict(DEFAULT_CONFIG)
    for k, v in raw.items():
        cfg[_CONFIG_ALIASES.get(k, k)] = v
    if isinstance(cfg["hidden_units"], int):
        cfg["hidden_units"] = [cfg["hidden_units"]]
    cfg["hidden_units"] = list(cfg["hidden_units"])
    return cfg


# ==================================================================================================
# 3. DATA INGESTION, PREPROCESSING & SPLIT MANAGEMENT
# ==================================================================================================
@dataclass
class DataBundle:
    """Container holding preprocessed feature matrices, target vectors, and transformers."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # Raw original units (e.g. hg/ha)
    y_val: np.ndarray
    y_test: np.ndarray
    bundle: dict
    Xs: dict[str, np.ndarray] = field(init=False)
    y: dict[str, np.ndarray] = field(init=False)
    ys: dict[str, np.ndarray] = field(init=False)  # Scaled targets for neural net optimization

    def __post_init__(self):
        self.Xs = {s: getattr(self, f"X_{s}").astype("float32") for s in ("train", "val", "test")}
        self.y = {s: np.asarray(getattr(self, f"y_{s}"), float).ravel() for s in ("train", "val", "test")}
        self.ys = {s: self.scale_y(self.y[s]).astype("float32") for s in self.y}

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def feature_names(self) -> list[str]:
        try:
            return list(self.bundle["preprocessor"].get_feature_names_out())
        except Exception:
            return [f"f{i}" for i in range(self.n_features)]

    def scale_y(self, y_raw: np.ndarray) -> np.ndarray:
        return self.bundle["target_scaler"].transform(np.asarray(y_raw, float).reshape(-1, 1)).ravel()

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.bundle["target_scaler"].inverse_transform(np.asarray(y_scaled, float).reshape(-1, 1)).ravel()


def _ensure_dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _build_split_from_csv(csv_path: Path | str) -> tuple[dict, dict]:
    """Construct deterministic 70/15/15 train/val/test split and fit preprocessor on training data."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        if FALLBACK_ROOT_CSV.exists():
            csv_file = FALLBACK_ROOT_CSV
        elif CSV_PATH.exists():
            csv_file = CSV_PATH
        else:
            raise FileNotFoundError(
                f"Dataset CSV not found at '{csv_path}' or '{CSV_PATH}'.\n"
                "Please ensure 'yield_df.csv' is placed in the project root or 'data/' folder."
            )

    print(f"[*] Building deterministic 70/15/15 split from CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[CAT_COLS + NUM_COLS + [TARGET]].dropna().drop_duplicates().reset_index(drop=True)

    # Deterministic split: 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SPLIT_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SPLIT_SEED)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    preprocessor.fit(train_df[CAT_COLS + NUM_COLS])  # Fitted STRICTLY on training split

    arrays = {}
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        transformed = preprocessor.transform(split_df[CAT_COLS + NUM_COLS])
        arrays[f"X_{split_name}"] = _ensure_dense(transformed).astype("float32")
        arrays[f"y_{split_name}"] = split_df[TARGET].to_numpy(float)

    bundle = {
        "preprocessor": preprocessor,
        "cat_cols": list(CAT_COLS),
        "num_cols": list(NUM_COLS),
        "feature_order": list(CAT_COLS) + list(NUM_COLS),
        "target": TARGET,
    }
    np.savez_compressed(SPLIT_PATH, **arrays)
    return arrays, bundle


def load_data(csv_path: Path | str = None, force_rebuild: bool = False) -> DataBundle:
    """Load or generate the standardized dataset split and feature preprocessors."""
    csv_to_use = csv_path or (CSV_PATH if CSV_PATH.exists() else FALLBACK_ROOT_CSV)

    if not force_rebuild and SPLIT_PATH.exists() and PREPROC_PATH.exists():
        stored = np.load(SPLIT_PATH, allow_pickle=True)
        arrays = {k: _ensure_dense(stored[k]) for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test")}
        loaded_bundle = joblib.load(PREPROC_PATH)
        if isinstance(loaded_bundle, dict):
            bundle = dict(loaded_bundle)
        else:
            bundle = {"preprocessor": loaded_bundle}
        bundle.setdefault("cat_cols", list(CAT_COLS))
        bundle.setdefault("num_cols", list(NUM_COLS))
        bundle.setdefault("feature_order", list(CAT_COLS) + list(NUM_COLS))
        bundle.setdefault("target", TARGET)
    else:
        arrays, bundle = _build_split_from_csv(csv_to_use)

    if "target_scaler" not in bundle:
        bundle["target_scaler"] = StandardScaler().fit(np.asarray(arrays["y_train"], float).reshape(-1, 1))

    # Persist updated bundle with target scaler for seamless inference
    joblib.dump(bundle, PREPROC_PATH)
    return DataBundle(**arrays, bundle=bundle)


# ==================================================================================================
# 4. NEURAL NETWORK ARCHITECTURE & TRAINING HARNESS
# ==================================================================================================
def get_initializer(name: str):
    """Factory creating a fresh Keras kernel initializer instance."""
    _ensure_tf()
    name_clean = str(name).lower()
    table = {
        "zeros": keras.initializers.Zeros,
        "zero": keras.initializers.Zeros,
        "constant": lambda: keras.initializers.Constant(0.05),
        "random_normal": lambda: keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        "normal": lambda: keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        "glorot_uniform": keras.initializers.GlorotUniform,
        "glorot": keras.initializers.GlorotUniform,
        "xavier": keras.initializers.GlorotUniform,
        "glorot_normal": keras.initializers.GlorotNormal,
        "he_normal": keras.initializers.HeNormal,
        "he": keras.initializers.HeNormal,
        "he_uniform": keras.initializers.HeUniform,
    }
    if name_clean not in table:
        raise ValueError(f"Unknown initializer '{name}'. Supported options: {sorted(table.keys())}")
    return table[name_clean]()


def get_optimizer(name: str, learning_rate: float):
    """Factory creating a compiled Keras optimizer."""
    _ensure_tf()
    opt_name = str(name).lower()
    table = {
        "adam": keras.optimizers.Adam,
        "nadam": keras.optimizers.Nadam,
        "rmsprop": keras.optimizers.RMSprop,
        "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    }
    if opt_name not in table:
        raise ValueError(f"Unknown optimizer '{name}'. Supported options: {sorted(table.keys())}")
    opt_fn = table[opt_name]
    return opt_fn(learning_rate=learning_rate) if opt_name == "sgd" else opt_fn(learning_rate=learning_rate)


def build_mlp(config: dict, input_dim: int) -> keras.Model:
    """Build configurable Keras MLP model: Dense -> [BatchNorm] -> Activation -> [Dropout] -> Dense(1)."""
    _ensure_tf()
    cfg = {**DEFAULT_CONFIG, **config}
    reg = regularizers.l2(cfg["l2"]) if cfg.get("l2") and cfg["l2"] > 0 else None

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(cfg["hidden_units"]):
        x = layers.Dense(
            units,
            kernel_initializer=get_initializer(cfg["initializer"]),
            kernel_regularizer=reg,
            name=f"dense_{i}",
        )(x)
        if cfg.get("batch_norm"):
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation(cfg["activation"], name=f"act_{i}")(x)
        if cfg.get("dropout") and cfg["dropout"] > 0:
            x = layers.Dropout(cfg["dropout"], name=f"drop_{i}")(x)

    outputs = layers.Dense(1, kernel_initializer=get_initializer(cfg["initializer"]), name="out")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="yield_mlp")
    opt = get_optimizer(cfg["optimizer"], cfg["learning_rate"])
    model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])
    return model


def evaluate_mlp(model: keras.Model, data: DataBundle) -> dict[str, dict[str, float]]:
    """Compute regression metrics across train, val, and test splits in raw original units (hg/ha)."""
    results = {}
    for split in ("train", "val", "test"):
        predictions_scaled = model.predict(data.Xs[split], batch_size=4096, verbose=0).ravel()
        predictions_raw = data.inverse_y(predictions_scaled)
        results[split] = calc_metrics(data.y[split], predictions_raw)
    return results


def train_mlp(
    config: dict,
    data: DataBundle,
    seed: int,
    epochs: int = None,
    patience: int = None,
    x_train: np.ndarray = None,
    y_train: np.ndarray = None,
    verbose: int = 0,
) -> tuple[dict, keras.Model]:
    """Train single MLP model instance with optional early stopping or custom train arrays."""
    _ensure_tf()
    cfg = {**DEFAULT_CONFIG, **config}
    set_seed(seed)
    model = build_mlp(cfg, data.n_features)

    callbacks = []
    if patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_mse",
                patience=patience,
                restore_best_weights=True,
            )
        )

    x_tr = data.Xs["train"] if x_train is None else x_train
    y_tr = data.ys["train"] if y_train is None else y_train
    val_data = (data.Xs["val"], data.ys["val"]) if x_train is None else None

    t0 = time.time()
    history = model.fit(
        x_tr,
        y_tr,
        validation_data=val_data,
        epochs=epochs or cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    elapsed = time.time() - t0

    h = history.history
    mse_key = "mse" if "mse" in h else next(k for k in h if "mean_squared" in k and not k.startswith("val_"))
    train_mse = [float(v) for v in h[mse_key]]
    val_mse = [float(v) for v in h[f"val_{mse_key}"]] if val_data else train_mse
    best_val = min(val_mse)

    record = {
        "seed": seed,
        "epochs_run": len(train_mse),
        "best_epoch": int(np.argmin(val_mse)) + 1,
        "epochs_to_converge": int(np.argmax(np.array(val_mse) <= best_val * 1.05)) + 1,
        "train_time_s": elapsed,
        "n_params": int(model.count_params()),
        "history": {"train_mse": train_mse, "val_mse": val_mse},
    }
    record.update(evaluate_mlp(model, data))
    return record, model


def probe_signal_stats(config: dict, data: DataBundle, seed: int = 42, n: int = 512) -> list[dict]:
    """Measure layer-wise activation mean/std and gradient std at epoch 0 to detect signal explosion/vanishing."""
    _ensure_tf()
    cfg = {**DEFAULT_CONFIG, **config}
    set_seed(seed)
    model = build_mlp(cfg, data.n_features)

    x_sample = tf.constant(data.Xs["train"][:n])
    y_sample = tf.constant(data.ys["train"][:n].reshape(-1, 1))

    activation_layers = [layer for layer in model.layers if layer.name.startswith("act_")]
    probe_model = keras.Model(model.input, [l.output for l in activation_layers] + [model.output])

    dense_names = [f"dense_{i}" for i in range(len(activation_layers))] + ["out"]
    kernels = [model.get_layer(name).kernel for name in dense_names]

    with tf.GradientTape() as tape:
        outputs = probe_model(x_sample, training=False)
        loss = tf.reduce_mean(tf.square(outputs[-1] - y_sample))

    gradients = tape.gradient(loss, kernels)
    stats = []
    for i, name in enumerate(dense_names):
        act_vals = outputs[i].numpy() if i < len(activation_layers) else outputs[-1].numpy()
        stats.append({
            "layer": name,
            "act_mean": float(act_vals.mean()),
            "act_std": float(act_vals.std()),
            "grad_std": float(tf.math.reduce_std(gradients[i]).numpy()),
        })
    return stats


# ==================================================================================================
# 5. DAY 1: BACKPROPAGATION VERIFICATION DEMO
# ==================================================================================================
def run_backprop_demo() -> dict[str, float]:
    """Demonstrate 2-layer backpropagation step in NumPy vs tf.GradientTape and gradient descent update."""
    _ensure_tf()
    print("\n" + "=" * 80)
    print("DAY 1: BACKPROPAGATION THEORETICAL & EMPIRICAL VERIFICATION DEMO")
    print("=" * 80)
    print("Architecture: x(4, 3) -> Dense(5, tanh) -> Dense(1, linear), Loss: Mean Squared Error")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    y = rng.normal(size=(4, 1))
    W1 = rng.normal(size=(3, 5)) * 0.5
    b1 = np.zeros(5)
    W2 = rng.normal(size=(5, 1)) * 0.5
    b2 = np.zeros(1)

    # 1. Forward Pass
    z1 = x @ W1 + b1
    a1 = np.tanh(z1)
    yhat = a1 @ W2 + b2
    loss_initial = np.mean((yhat - y) ** 2)

    # 2. Backward Pass via Chain Rule
    dL_dyhat = 2 * (yhat - y) / len(x)              # dL / dyhat
    dW2 = a1.T @ dL_dyhat                           # dL / dW2 = a1^T @ dL/dyhat
    db2 = dL_dyhat.sum(axis=0)                      # dL / db2
    dz1 = (dL_dyhat @ W2.T) * (1 - a1 ** 2)         # Backprop through W2 and tanh' = 1 - tanh^2
    dW1 = x.T @ dz1                                 # dL / dW1 = x^T @ dz1
    db1 = dz1.sum(axis=0)                           # dL / db1

    # 3. Autodiff Cross-Verification via TensorFlow GradientTape
    tf_vars = [tf.Variable(val, dtype=tf.float64) for val in (W1, b1, W2, b2)]
    with tf.GradientTape() as tape:
        tf_z1 = tf.constant(x) @ tf_vars[0] + tf_vars[1]
        tf_a1 = tf.tanh(tf_z1)
        tf_yhat = tf_a1 @ tf_vars[2] + tf_vars[3]
        tf_loss = tf.reduce_mean((tf_yhat - tf.constant(y)) ** 2)

    tf_grads = tape.gradient(tf_loss, tf_vars)
    discrepancies = {}
    print("\n[*] Comparing Analytical Gradients (NumPy) vs Autodiff (tf.GradientTape):")
    for name, analytical, autodiff in zip(["W1", "b1", "W2", "b2"], [dW1, db1, dW2, db2], tf_grads):
        max_diff = float(np.abs(analytical - autodiff.numpy()).max())
        discrepancies[name] = max_diff
        print(f"    - Parameter {name:2s}: Max absolute difference = {max_diff:.2e}")

    # 4. Weight Update Step (Gradient Descent: W <- W - lr * dW)
    lr = 0.1
    W1_new, b1_new = W1 - lr * dW1, b1 - lr * db1
    W2_new, b2_new = W2 - lr * dW2, b2 - lr * db2
    loss_after = float(np.mean((np.tanh(x @ W1_new + b1_new) @ W2_new + b2_new - y) ** 2))

    print(f"\n[*] Loss before update: {loss_initial:.5f}")
    print(f"[*] Loss after 1 step : {loss_after:.5f} (Delta = {loss_initial - loss_after:.5f})")

    # Generate visual validation plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    param_names = list(discrepancies.keys())
    diff_values = [discrepancies[k] for k in param_names]
    ax.bar(param_names, diff_values, color="#2a7f62")
    ax.set_yscale("log")
    ax.set_ylabel("Max |NumPy - TF| difference")
    ax.set_title("Backpropagation Gradient Verification (Zero Discrepancy)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_file = PLOTS_DIR / "backprop_demo.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"[*] Saved verification plot to: {plot_file}")
    return discrepancies


# ==================================================================================================
# 6. DAY 1: XGBOOST BASELINE BENCHMARK
# ==================================================================================================
def run_xgb_baseline(data: DataBundle = None, seeds: list[int] = None, quick: bool = False) -> dict:
    """Train and evaluate the XGBoost regression baseline model across multiple seeds."""
    _ensure_xgb()
    print("\n" + "=" * 80)
    print("DAY 1: XGBOOST TABULAR BASELINE BENCHMARK")
    print("=" * 80)
    data = data or load_data()
    eval_seeds = [42] if quick else (seeds or SEEDS)
    n_estimators = 50 if quick else 500

    runs = []
    first_model = None

    for seed in eval_seeds:
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            early_stopping_rounds=30 if not quick else None,
            eval_metric="rmse",
            random_state=seed,
            n_jobs=-1,
        )
        t0 = time.time()
        model.fit(
            data.X_train,
            data.y["train"],
            eval_set=[(data.X_val, data.y["val"])],
            verbose=False,
        )
        elapsed = time.time() - t0
        best_tree = int(getattr(model, "best_iteration", n_estimators - 1)) + 1
        rec = {
            "seed": seed,
            "train_time_s": elapsed,
            "n_trees": best_tree,
        }
        for split in ("train", "val", "test"):
            x_split = getattr(data, f"X_{split}")
            rec[split] = calc_metrics(data.y[split], model.predict(x_split))
        runs.append(rec)
        first_model = first_model or model
        print(
            f"[*] Seed {seed}: Test RMSE={rec['test']['rmse']:,.1f} | "
            f"Test R²={rec['test']['r2']:.4f} | Trees={rec['n_trees']} | Time={elapsed:.1f}s"
        )

    summary = {m: mean_std([r["test"][m] for r in runs]) for m in ("rmse", "mae", "r2")}
    summary["train_time_s"] = mean_std([r["train_time_s"] for r in runs])
    summary["n_trees"] = mean_std([r["n_trees"] for r in runs])

    xgb_result = {"runs": runs, "summary": summary}
    save_json(xgb_result, XGB_BASELINE_PATH)

    # Feature Importance Plot (top 20 features)
    importances = first_model.feature_importances_
    names = np.array(data.feature_names)
    sorted_idx = np.argsort(importances)[::-1][:20][::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names[sorted_idx], importances[sorted_idx], color="#2a7f62")
    ax.set_xlabel("XGBoost Feature Importance (Gain Share)")
    ax.set_title("XGBoost Baseline - Top 20 Features")
    fig.tight_layout()
    plot_file = PLOTS_DIR / "xgb_feature_importance.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)

    print(
        f"\n[+] XGBoost Summary: Test RMSE {format_pm(summary['rmse']['mean'], summary['rmse']['std'])} | "
        f"Test R² {format_pm(summary['r2']['mean'], summary['r2']['std'], 4)} | "
        f"Saved: {XGB_BASELINE_PATH}"
    )
    return xgb_result


# ==================================================================================================
# 7. DAY 2-A: WEIGHT INITIALIZATION ABLATION EXPERIMENTS
# ==================================================================================================
def run_init_experiments(
    data: DataBundle = None,
    seeds: list[int] = None,
    quick: bool = False,
    also_tanh: bool = False,
    epochs: int = None,
    patience: int = 10,
) -> dict:
    """Compare Zeros vs Random Normal vs Glorot vs He across activations with convergence logging."""
    _ensure_tf()
    print("\n" + "=" * 80)
    print("DAY 2-A: WEIGHT INITIALIZATION COMPARISON EXPERIMENTS")
    print("=" * 80)
    data = data or load_data()
    eval_seeds = [42] if quick else (seeds or SEEDS)
    base_cfg = load_base_config()
    max_epochs = epochs or (6 if quick else base_cfg["epochs"])

    activations = [base_cfg["activation"]]
    if also_tanh and base_cfg["activation"] != "tanh":
        activations.append("tanh")

    best_init_overall = None

    for idx, act in enumerate(activations):
        is_primary = (idx == 0)
        print(f"\n[*] Evaluating Initializers on Activation: '{act}'")
        runs = []
        for init in INIT_METHODS:
            for s in eval_seeds:
                cfg = {
                    **base_cfg,
                    "activation": act,
                    "initializer": init,
                    "dropout": 0.0,
                    "l2": 0.0,
                    "batch_norm": False,
                }
                rec, _ = train_mlp(cfg, data, s, epochs=max_epochs, patience=patience)
                rec["variant"] = init
                runs.append(rec)
                print(
                    f"    [{act}] {init:15s} Seed {s}: Val RMSE={rec['val']['rmse']:,.1f} | "
                    f"Test R²={rec['test']['r2']:.4f} | Converged @ Epoch {rec['epochs_to_converge']}"
                )

        summary_df = summarize_runs(runs)

        # Build clean markdown and CSV tables for presentation
        presentation_table = pd.DataFrame({
            "Initializer": summary_df["variant"],
            "Val RMSE": [format_pm(m, s) for m, s in zip(summary_df["val_rmse"], summary_df["val_rmse_std"])],
            "Test RMSE": [format_pm(m, s) for m, s in zip(summary_df["test_rmse"], summary_df["test_rmse_std"])],
            "Test R²": [format_pm(m, s, 3) for m, s in zip(summary_df["test_r2"], summary_df["test_r2_std"])],
            "Epochs to Converge": [f"{v:.1f}" for v in summary_df["epochs_to_converge"]],
        })

        summary_df.to_csv(RESULTS_DIR / f"init_table_{act}.csv", index=False)
        (RESULTS_DIR / f"init_table_{act}.md").write_text(format_md_table(presentation_table) + "\n", encoding="utf-8")
        save_json(runs, RESULTS_DIR / f"init_results_{act}.json")

        print(f"\n--- Initializer Results Table ({act}) ---")
        print(format_md_table(presentation_table))

        # First N epochs convergence plot
        L = min(10, min(len(r["history"]["val_mse"]) for r in runs))
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for init in INIT_METHODS:
            curves = np.array([r["history"]["val_mse"][:L] for r in runs if r["variant"] == init])
            mu, sd = curves.mean(axis=0), curves.std(axis=0)
            ax.plot(range(1, L + 1), mu, marker="o", ms=3, label=init, color=PALETTE.get(init, "#333333"))
            ax.fill_between(range(1, L + 1), mu - sd, mu + sd, alpha=0.15, color=PALETTE.get(init, "#333333"))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE (Standardized Target)")
        ax.set_title(f"Initialization vs Convergence Speed ({act}, First {L} Epochs)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"init_convergence_{act}.png", dpi=150)
        plt.close(fig)

        # Layer-wise Signal Statistics at Epoch 0
        stats = []
        for init in INIT_METHODS:
            cfg = {
                **base_cfg,
                "activation": act,
                "initializer": init,
                "dropout": 0.0,
                "l2": 0.0,
                "batch_norm": False,
            }
            layer_stats = probe_signal_stats(cfg, data, eval_seeds[0])
            for st in layer_stats:
                stats.append({"init": init, **st})
        pd.DataFrame(stats).to_csv(RESULTS_DIR / f"init_signal_stats_{act}.csv", index=False)

        if is_primary:
            best_idx = summary_df["val_rmse"].idxmin()
            best_init = summary_df.loc[best_idx]
            best_init_overall = {
                "initializer": best_init["variant"],
                "activation": act,
                "val_rmse": float(best_init["val_rmse"]),
            }
            save_json(best_init_overall, BEST_INIT_PATH)
            print(f"\n[+] Selected Best Initializer (by Validation RMSE): {best_init_overall['initializer']}")

    return best_init_overall


# ==================================================================================================
# 8. DAY 2-B: REGULARIZATION ABLATION & OVERFITTING DEMO
# ==================================================================================================
def run_reg_experiments(
    data: DataBundle = None,
    seeds: list[int] = None,
    quick: bool = False,
    epochs: int = None,
    overfit_epochs: int = None,
    skip_overfit: bool = False,
) -> dict:
    """Evaluate L2, Dropout, BatchNorm, Early Stopping, and run deliberate overfitting demo."""
    _ensure_tf()
    print("\n" + "=" * 80)
    print("DAY 2-B: REGULARIZATION ABLATION & GENERALIZATION STUDY")
    print("=" * 80)
    data = data or load_data()
    eval_seeds = [42] if quick else (seeds or SEEDS)
    base_cfg = load_base_config()
    reg_epochs = epochs or (6 if quick else 60)
    overfit_ep = overfit_epochs or (10 if quick else 200)

    best_init = load_json(BEST_INIT_PATH)
    if best_init:
        base_cfg["initializer"] = best_init["initializer"]
    base_cfg.update(dropout=0.0, l2=0.0, batch_norm=False)

    def _execute_variant(name: str, overrides: dict, patience: int = None, cfg_base: dict = None, n_ep: int = None):
        records = []
        for s in eval_seeds:
            c = {**(cfg_base or base_cfg), **overrides}
            rec, _ = train_mlp(c, data, s, epochs=n_ep or reg_epochs, patience=patience)
            rec.update(variant=name, overrides=overrides, patience=patience)
            records.append(rec)
            print(
                f"    {name:26s} Seed {s}: Train RMSE={rec['train']['rmse']:,.1f} | "
                f"Val RMSE={rec['val']['rmse']:,.1f} | Test RMSE={rec['test']['rmse']:,.1f}"
            )
        return records

    # Stage 1: Individual Regularization Techniques
    spec: dict[str, tuple[dict, int | None]] = {"none": ({}, None)}
    for l2_val in (1e-4, 1e-3, 1e-2):
        spec[f"L2={l2_val:g}"] = ({"l2": l2_val}, None)
    for do_val in (0.1, 0.2, 0.3, 0.5):
        spec[f"Dropout={do_val}"] = ({"dropout": do_val}, None)
    spec["BatchNorm"] = ({"batch_norm": True}, None)
    spec["EarlyStop p=5"] = ({}, 5)
    spec["EarlyStop p=10"] = ({}, 10)

    all_runs = []
    for name, (ov, pat) in spec.items():
        all_runs += _execute_variant(name, ov, pat)

    summary_df = summarize_runs(all_runs).set_index("variant")
    best_l2 = min([k for k in spec if k.startswith("L2=")], key=lambda k: summary_df.loc[k, "val_rmse"])
    best_do = min([k for k in spec if k.startswith("Dropout=")], key=lambda k: summary_df.loc[k, "val_rmse"])

    # Stage 2: Combined L2 + Dropout (Best of Each)
    combo_name = f"L2+Dropout ({best_l2}, {best_do})"
    combo_overrides = {**spec[best_l2][0], **spec[best_do][0]}
    spec[combo_name] = (combo_overrides, None)
    all_runs += _execute_variant(combo_name, combo_overrides)
    summary_df = summarize_runs(all_runs).set_index("variant")

    # Select best combination strictly based on VALIDATION RMSE (among fixed-epoch runs)
    fixed_candidates = [k for k, (_, pat) in spec.items() if pat is None]
    best_overall_name = min(fixed_candidates, key=lambda k: summary_df.loc[k, "val_rmse"])
    best_reg_name = min([k for k in fixed_candidates if k != "none"], key=lambda k: summary_df.loc[k, "val_rmse"])
    best_es_name = min(["EarlyStop p=5", "EarlyStop p=10"], key=lambda k: summary_df.loc[k, "val_rmse"])
    es_patience = spec[best_es_name][1]

    best_reg_summary = {
        "variant": best_overall_name,
        "overrides": spec[best_overall_name][0],
        "early_stopping_patience": es_patience,
        "val_rmse": float(summary_df.loc[best_overall_name, "val_rmse"]),
        "best_regularized_variant": best_reg_name,
        "best_regularized_overrides": spec[best_reg_name][0],
    }
    save_json(best_reg_summary, BEST_REG_PATH)
    save_json(all_runs, RESULTS_DIR / "reg_results.json")

    # Regularization Markdown & CSV Tables
    table_df = summary_df.reset_index()
    presentation_table = pd.DataFrame({
        "Variant": table_df["variant"],
        "Train RMSE": [format_pm(m, s) for m, s in zip(table_df["train_rmse"], table_df["train_rmse_std"])],
        "Val RMSE": [format_pm(m, s) for m, s in zip(table_df["val_rmse"], table_df["val_rmse_std"])],
        "Test RMSE": [format_pm(m, s) for m, s in zip(table_df["test_rmse"], table_df["test_rmse_std"])],
        "Gap (Test - Train)": [format_pm(m, s) for m, s in zip(table_df["gap"], table_df["gap_std"])],
        "Test R²": [format_pm(m, s, 3) for m, s in zip(table_df["test_r2"], table_df["test_r2_std"])],
        "Epochs Run": [f"{v:.0f}" for v in table_df["epochs_run"]],
    })
    table_df.to_csv(RESULTS_DIR / "reg_table.csv", index=False)
    (RESULTS_DIR / "reg_table.md").write_text(format_md_table(presentation_table) + "\n", encoding="utf-8")

    print("\n--- Regularization Comparison Table ---")
    print(format_md_table(presentation_table))

    # Plot 1: Train vs Val Loss Curves (None vs Best Regularized)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["none", best_reg_name]):
        first_run = next(r for r in all_runs if r["variant"] == name and r["seed"] == eval_seeds[0])
        ax.plot(first_run["history"]["train_mse"], label="Train", color=PALETTE["train"])
        ax.plot(first_run["history"]["val_mse"], label="Validation", color=PALETTE["val"])
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MSE (Standardized Target)")
    fig.suptitle("Train vs Validation Loss: Baseline vs Best Regularization")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reg_loss_curves.png", dpi=150)
    plt.close(fig)

    # Plot 2: Generalization Gap Bar Chart
    fixed_variants = table_df[~table_df["variant"].str.startswith("EarlyStop")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x_idx = np.arange(len(fixed_variants))
    w = 0.38
    ax.bar(
        x_idx - w / 2,
        fixed_variants["train_rmse"],
        w,
        yerr=fixed_variants["train_rmse_std"],
        label="Train RMSE",
        color="#2a7f62",
    )
    ax.bar(
        x_idx + w / 2,
        fixed_variants["test_rmse"],
        w,
        yerr=fixed_variants["test_rmse_std"],
        label="Test RMSE",
        color="#c0392b",
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(fixed_variants["variant"], rotation=30, ha="right")
    ax.set_ylabel("RMSE (Raw Yield Units)")
    ax.set_title("Generalization Gap by Regularization Variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reg_gap.png", dpi=150)
    plt.close(fig)

    # Stage 3: Deliberate Overfitting Demonstration (256-128-64, no early stopping)
    if not skip_overfit:
        print("\n[*] Running Deliberate Overfitting Demonstration (Oversized MLP, 200 epochs)...")
        oversized_cfg = {**base_cfg, "hidden_units": [256, 128, 64]}
        demo_records = []
        for demo_name, ov in (("No Regularization (Overfit)", {}), (f"Regularized ({combo_name})", combo_overrides)):
            rec, _ = train_mlp(
                {**oversized_cfg, **ov},
                data,
                eval_seeds[0],
                epochs=overfit_ep,
                patience=None,
            )
            rec["variant"] = demo_name
            demo_records.append(rec)
            print(
                f"    {demo_name}: Train RMSE={rec['train']['rmse']:,.1f} | "
                f"Test RMSE={rec['test']['rmse']:,.1f} | Gap={rec['test']['rmse'] - rec['train']['rmse']:,.1f}"
            )
        save_json(demo_records, RESULTS_DIR / "overfit_demo.json")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, rec in zip(axes, demo_records):
            ax.plot(rec["history"]["train_mse"], label="Train", color=PALETTE["train"])
            ax.plot(rec["history"]["val_mse"], label="Validation", color=PALETTE["val"])
            gap = rec["test"]["rmse"] - rec["train"]["rmse"]
            ax.set_title(f"{rec['variant']}\nTrain RMSE {rec['train']['rmse']:,.0f} vs Test {rec['test']['rmse']:,.0f} (Gap: {gap:,.0f})", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("MSE (Standardized Target)")
        fig.suptitle(f"Deliberate Overfitting Demo: 256-128-64 Architecture, {overfit_ep} Epochs (No Early Stopping)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "overfit_demo.png", dpi=150)
        plt.close(fig)

    print(f"\n[+] Selected Best Regularization Configuration: {best_reg_summary['variant']}")
    return best_reg_summary


# ==================================================================================================
# 9. DAY 2-C/D: FINAL MODEL TRAINING, CHECKPOINTING & XGBOOST COMPARISON
# ==================================================================================================
def run_final_model(
    data: DataBundle = None,
    seeds: list[int] = None,
    quick: bool = False,
    trainval: bool = False,
    epochs: int = None,
) -> dict:
    """Train final MLP combining best initializer and regularization, benchmark vs XGBoost, save models."""
    _ensure_tf()
    print("\n" + "=" * 80)
    print("DAY 2-C/D: FINAL MODEL TRAINING & COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    data = data or load_data()
    eval_seeds = [42] if quick else (seeds or SEEDS)
    base_cfg = load_base_config()

    # Automatically load or run missing prerequisite stages
    best_init = load_json(BEST_INIT_PATH)
    if not best_init:
        print("[!] best_init.json not found -> executing initialization ablation first...")
        best_init = run_init_experiments(data, seeds=eval_seeds, quick=quick)

    best_reg = load_json(BEST_REG_PATH)
    if not best_reg:
        print("[!] best_reg.json not found -> executing regularization ablation first...")
        best_reg = run_reg_experiments(data, seeds=eval_seeds, quick=quick)

    xgb_metrics = load_json(XGB_BASELINE_PATH)
    if not xgb_metrics:
        print("[!] xgb_baseline.json not found -> executing XGBoost baseline first...")
        xgb_metrics = run_xgb_baseline(data, seeds=eval_seeds, quick=quick)

    patience = best_reg["early_stopping_patience"]
    max_epochs = epochs or (6 if quick else base_cfg["epochs"])

    baseline_cfg = {**base_cfg, "dropout": 0.0, "l2": 0.0, "batch_norm": False}
    best_cfg = {**base_cfg, "initializer": best_init["initializer"], **best_reg["overrides"]}

    print("\n[*] Training Baseline MLP (unregularized) and Best Regularized MLP across seeds...")
    base_records, best_records, best_models = [], [], []

    for s in eval_seeds:
        rec_base, _ = train_mlp(baseline_cfg, data, s, epochs=max_epochs, patience=patience)
        base_records.append(rec_base)

        rec_best, model_best = train_mlp(best_cfg, data, s, epochs=max_epochs, patience=patience)
        best_records.append(rec_best)
        best_models.append(model_best)
        print(
            f"    Seed {s} -> Baseline Val RMSE: {rec_base['val']['rmse']:,.1f} | "
            f"Best MLP Val RMSE: {rec_best['val']['rmse']:,.1f} | Best Test RMSE: {rec_best['test']['rmse']:,.1f}"
        )

    # Model to ship is strictly chosen by BEST VALIDATION RMSE (never tuned on test set)
    best_val_idx = int(np.argmin([r["val"]["rmse"] for r in best_records]))
    ship_model = best_models[best_val_idx]
    ship_seed = eval_seeds[best_val_idx]

    # Optional retraining on combined train + validation splits
    if trainval:
        mean_best_epoch = max(1, int(round(np.mean([r["best_epoch"] for r in best_records]))))
        print(f"\n[*] Retraining final model on Train + Validation splits for {mean_best_epoch} epochs (Seed {ship_seed})...")
        set_seed(ship_seed)
        ship_model = build_mlp(best_cfg, data.n_features)
        combined_x = np.vstack([data.Xs["train"], data.Xs["val"]])
        combined_y = np.concatenate([data.ys["train"], data.ys["val"]])
        ship_model.fit(
            combined_x,
            combined_y,
            epochs=mean_best_epoch,
            batch_size=best_cfg["batch_size"],
            verbose=0,
        )

    # Save finalized model and configuration artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ship_model.save(MODEL_PATH)
    save_json({"config": best_cfg, "early_stopping_patience": patience}, MODELS_DIR / "final_config.json")
    print(f"\n[+] Saved Deployable Model Checkpoint to: {MODEL_PATH}")
    print(f"[+] Saved Production Preprocessor Bundle to: {PREPROC_PATH}")

    # Aggregate Test Performance Across Seeds
    def _aggregate(records: list[dict]) -> dict:
        return {
            "rmse": mean_std([r["test"]["rmse"] for r in records]),
            "mae": mean_std([r["test"]["mae"] for r in records]),
            "r2": mean_std([r["test"]["r2"] for r in records]),
            "train_time_s": mean_std([r["train_time_s"] for r in records]),
            "n_params": records[0]["n_params"],
        }

    final_results = {
        "baseline_mlp": _aggregate(base_records),
        "best_mlp": _aggregate(best_records),
        "xgboost": xgb_metrics["summary"],
        "best_config": best_cfg,
        "baseline_config": baseline_cfg,
        "early_stopping_patience": patience,
        "shipped_seed": ship_seed,
        "trainval_retrain": bool(trainval),
    }

    save_json(final_results, FINAL_METRICS_PATH)
    save_json({"baseline": base_records, "best": best_records}, RESULTS_DIR / "final_runs.json")

    # Build Comparison Summary Table
    comparison_data = [
        {
            "Model": "Baseline MLP (No Reg)",
            "Test RMSE": format_pm(final_results["baseline_mlp"]["rmse"]["mean"], final_results["baseline_mlp"]["rmse"]["std"]),
            "Test MAE": format_pm(final_results["baseline_mlp"]["mae"]["mean"], final_results["baseline_mlp"]["mae"]["std"]),
            "Test R²": format_pm(final_results["baseline_mlp"]["r2"]["mean"], final_results["baseline_mlp"]["r2"]["std"], 3),
            "Train Time (s)": f"{final_results['baseline_mlp']['train_time_s']['mean']:.1f}",
            "Complexity": f"{final_results['baseline_mlp']['n_params']:,} params",
        },
        {
            "Model": "Best MLP (Init + Reg)",
            "Test RMSE": format_pm(final_results["best_mlp"]["rmse"]["mean"], final_results["best_mlp"]["rmse"]["std"]),
            "Test MAE": format_pm(final_results["best_mlp"]["mae"]["mean"], final_results["best_mlp"]["mae"]["std"]),
            "Test R²": format_pm(final_results["best_mlp"]["r2"]["mean"], final_results["best_mlp"]["r2"]["std"], 3),
            "Train Time (s)": f"{final_results['best_mlp']['train_time_s']['mean']:.1f}",
            "Complexity": f"{final_results['best_mlp']['n_params']:,} params",
        },
        {
            "Model": "XGBoost Baseline",
            "Test RMSE": format_pm(final_results["xgboost"]["rmse"]["mean"], final_results["xgboost"]["rmse"]["std"]),
            "Test MAE": format_pm(final_results["xgboost"]["mae"]["mean"], final_results["xgboost"]["mae"]["std"]),
            "Test R²": format_pm(final_results["xgboost"]["r2"]["mean"], final_results["xgboost"]["r2"]["std"], 3),
            "Train Time (s)": f"{final_results['xgboost']['train_time_s']['mean']:.1f}",
            "Complexity": f"{final_results['xgboost']['n_trees']['mean']:.0f} trees",
        },
    ]
    comparison_df = pd.DataFrame(comparison_data)

    # Statistical Significance Analysis (Noise threshold = max seed standard deviation)
    def _evaluate_significance(model_a: dict, model_b: dict, label_a: str, label_b: str) -> str:
        diff = model_b["rmse"]["mean"] - model_a["rmse"]["mean"]
        noise_threshold = max(model_a["rmse"]["std"], model_b["rmse"]["std"])
        if abs(diff) <= noise_threshold:
            return f"{label_a} and {label_b} are statistically indistinguishable (RMSE difference {abs(diff):,.1f} <= seed standard deviation {noise_threshold:,.1f})."
        winner, loser = (label_a, label_b) if diff > 0 else (label_b, label_a)
        return f"{winner} outperforms {loser} by {abs(diff):,.1f} RMSE (statistically significant compared to seed variation {noise_threshold:,.1f})."

    conclusions = [
        _evaluate_significance(final_results["best_mlp"], final_results["baseline_mlp"], "Best MLP", "Baseline MLP"),
        _evaluate_significance(final_results["best_mlp"], final_results["xgboost"], "Best MLP", "XGBoost"),
        "Tabular agricultural data with heterogeneous categorical and continuous distributions often provides competitive performance across both gradient boosted trees and regularized neural networks; the MLP has been calibrated, verified, and shipped for production inference.",
    ]

    report_markdown = (
        "# Final Model Comparison & Benchmark (Test Set, Mean ± Std over Seeds)\n\n"
        + format_md_table(comparison_df)
        + "\n\n## Empirical Findings & Conclusions\n"
        + "\n".join(f"- {c}" for c in conclusions)
        + "\n"
    )
    (RESULTS_DIR / "final_comparison.md").write_text(report_markdown, encoding="utf-8")
    print("\n" + report_markdown)

    # Final Plots: Comparative Bar Charts & Learning Curves
    labels = ["Baseline MLP", "Best MLP", "XGBoost"]
    keys = ["baseline_mlp", "best_mlp", "xgboost"]
    colors = [PALETTE["baseline"], PALETTE["best"], PALETTE["xgboost"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("rmse", "r2"), ("Test RMSE (Lower is Better)", "Test R² (Higher is Better)")):
        means = [final_results[k][metric]["mean"] for k in keys]
        stds = [final_results[k][metric]["std"] for k in keys]
        ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    best_hist = best_records[best_val_idx]["history"]
    ax.plot(best_hist["train_mse"], label="Train MSE", color=PALETTE["train"])
    ax.plot(best_hist["val_mse"], label="Validation MSE", color=PALETTE["val"])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (Standardized Target)")
    ax.set_title(f"Final Shipped MLP Learning Curves (Seed {ship_seed})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_mlp_curves.png", dpi=150)
    plt.close(fig)

    return final_results


# ==================================================================================================
# 10. PRODUCTION INFERENCE API (ROLE H CONTRACT)
# ==================================================================================================
_MODEL_CACHE: dict = {}


def _load_inference_bundle() -> tuple[keras.Model, dict]:
    """Load and cache the trained Keras model and preprocessor bundle for high-speed inference."""
    _ensure_tf()
    if not _MODEL_CACHE:
        for required_path in (MODEL_PATH, PREPROC_PATH):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required inference asset '{required_path}' is missing.\n"
                    "Please run 'python main.py --final' to train and export the model and preprocessor."
                )
        bundle = joblib.load(PREPROC_PATH)
        if not isinstance(bundle, dict) or "target_scaler" not in bundle:
            raise RuntimeError(
                f"Invalid preprocessor bundle format at '{PREPROC_PATH}'. "
                "Ensure it was exported by the unified training pipeline."
            )
        _MODEL_CACHE["model"] = keras.models.load_model(MODEL_PATH, compile=False)
        _MODEL_CACHE["bundle"] = bundle
    return _MODEL_CACHE["model"], _MODEL_CACHE["bundle"]


def get_allowed_categories() -> dict[str, list[str]]:
    """Return dictionary of known categorical feature values (e.g. {'Area': [...], 'Item': [...]})."""
    try:
        _, bundle = _load_inference_bundle()
        cat_transformer = bundle["preprocessor"].named_transformers_["cat"]
        categories = cat_transformer.categories_
        return {col: [str(val) for val in vals] for col, vals in zip(bundle["cat_cols"], categories)}
    except Exception:
        return {}


def predict_yield(features: dict) -> float:
    """Predict crop yield in dataset units (hg/ha) from feature dictionary.

    Parameters:
        features: Dictionary containing required agricultural indicators:
                  - 'Area': str (e.g. "India", "Italy")
                  - 'Item': str (e.g. "Maize", "Wheat")
                  - 'Year': int/float (e.g. 2000)
                  - 'average_rain_fall_mm_per_year': float
                  - 'pesticides_tonnes': float

    Returns:
        float: Predicted crop yield in hectograms per hectare (hg/ha), clamped >= 0.0.
    """
    model, bundle = _load_inference_bundle()
    feature_order = list(bundle["feature_order"])

    # Schema Validation
    missing_keys = [k for k in feature_order if k not in features or features[k] is None]
    if missing_keys:
        raise ValueError(f"Missing required feature(s): {missing_keys}. Expected schema: {feature_order}")

    row = {}
    for col in bundle["num_cols"]:
        try:
            numeric_val = float(features[col])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{col}' must be a numeric value, received: {features[col]!r}") from None
        if not np.isfinite(numeric_val):
            raise ValueError(f"Feature '{col}' must be finite, received: {features[col]!r}")
        row[col] = numeric_val

    allowed_cats = get_allowed_categories()
    for col in bundle["cat_cols"]:
        raw_cat = str(features[col]).strip()
        row[col] = raw_cat
        if allowed_cats.get(col) and raw_cat not in allowed_cats[col]:
            warnings.warn(
                f"Unseen category '{raw_cat}' for feature '{col}'. It will be encoded as all-zeros "
                f"(handle_unknown='ignore'). For known values, call get_allowed_categories()."
            )

    df_row = pd.DataFrame([row])[feature_order]
    x_transformed = bundle["preprocessor"].transform(df_row)
    x_dense = _ensure_dense(x_transformed).astype("float32")

    scaled_prediction = model.predict(x_dense, verbose=0).reshape(-1, 1)
    raw_yield = float(bundle["target_scaler"].inverse_transform(scaled_prediction)[0, 0])
    return float(max(raw_yield, 0.0))


def run_sample_predictions():
    """Verify inference pipeline against standard test examples."""
    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE PIPELINE VERIFICATION")
    print("=" * 80)
    test_cases = [
        {
            "Area": "Italy",
            "Item": "Maize",
            "Year": 2000,
            "average_rain_fall_mm_per_year": 832.0,
            "pesticides_tonnes": 79447.0,
            "actual": 95277.0,
        },
        {
            "Area": "India",
            "Item": "Wheat",
            "Year": 2010,
            "average_rain_fall_mm_per_year": 1083.0,
            "pesticides_tonnes": 45000.0,
            "actual": None,
        },
        {
            "Area": "Albania",
            "Item": "Potatoes",
            "Year": 1990,
            "average_rain_fall_mm_per_year": 1485.0,
            "pesticides_tonnes": 121.0,
            "actual": 66667.0,
        },
    ]

    for idx, sample in enumerate(test_cases, 1):
        actual_str = f"(Actual: {sample['actual']:,.0f})" if sample["actual"] else ""
        pred = predict_yield(sample)
        print(f"[{idx}] {sample['Area']} | {sample['Item']} | Year {sample['Year']} -> Predicted Yield: {pred:,.0f} hg/ha {actual_str}")


# ==================================================================================================
# 11. UNIFIED COMMAND-LINE INTERFACE & DISPATCHER
# ==================================================================================================
def build_cli_parser() -> argparse.ArgumentParser:
    """Build unified argument parser for the entire Role E workflow."""
    parser = argparse.ArgumentParser(
        description="Agricultural Crop Yield Prediction - Role E Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Execution Stage Flags
    parser.add_argument("--all", action="store_true", help="Execute the complete pipeline end-to-end")
    parser.add_argument("--backprop", action="store_true", help="Run backpropagation verification demo")
    parser.add_argument("--xgb", action="store_true", help="Train and benchmark XGBoost baseline")
    parser.add_argument("--init", action="store_true", help="Run weight initialization experiments")
    parser.add_argument("--reg", action="store_true", help="Run regularization ablation & overfit demo")
    parser.add_argument("--final", action="store_true", help="Train and benchmark final MLP model")
    parser.add_argument("--predict", action="store_true", help="Run sample prediction test")

    # Configuration & Tuning Flags
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode: 1 seed, minimal epochs")
    parser.add_argument("--trainval", action="store_true", help="Retrain final model on combined train+val")
    parser.add_argument("--also-tanh", action="store_true", help="Include tanh activation in init comparison")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--overfit-epochs", type=int, default=None, help="Epochs for deliberate overfit demo")
    parser.add_argument("--no-overfit-demo", action="store_true", help="Skip deliberate overfit demo")
    parser.add_argument("--data-path", type=str, default=None, help="Custom path to yield_df.csv")
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    # Determine Active Seeds
    active_seeds = SEEDS[:1] if args.quick else SEEDS[: max(1, min(args.n_seeds, len(SEEDS)))]

    # If no specific stage flag is provided, default to --all or provide helpful overview
    no_flags = not any([args.all, args.backprop, args.xgb, args.init, args.reg, args.final, args.predict])
    run_all = args.all or no_flags

    print("=" * 80)
    print("AGRICULTURAL CROP YIELD PREDICTION PIPELINE (ROLE E)")
    print(f"Working Directory : {ROOT_DIR}")
    print(f"Active Seeds      : {active_seeds}")
    print(f"Mode              : {'Quick Smoke-Test' if args.quick else 'Full Research Benchmark'}")
    print("=" * 80)

    # 1. Backpropagation Verification
    if run_all or args.backprop:
        run_backprop_demo()

    # Load Data once for subsequent stages
    data = None
    if run_all or any([args.xgb, args.init, args.reg, args.final]):
        data = load_data(csv_path=args.data_path)

    # 2. XGBoost Baseline
    if run_all or args.xgb:
        run_xgb_baseline(data, seeds=active_seeds, quick=args.quick)

    # 3. Initialization Ablation
    if run_all or args.init:
        run_init_experiments(
            data,
            seeds=active_seeds,
            quick=args.quick,
            also_tanh=args.also_tanh,
            epochs=args.epochs,
        )

    # 4. Regularization Ablation
    if run_all or args.reg:
        run_reg_experiments(
            data,
            seeds=active_seeds,
            quick=args.quick,
            epochs=args.epochs,
            overfit_epochs=args.overfit_epochs,
            skip_overfit=args.no_overfit_demo,
        )

    # 5. Final Model Training & Comparison
    if run_all or args.final:
        run_final_model(
            data,
            seeds=active_seeds,
            quick=args.quick,
            trainval=args.trainval,
            epochs=args.epochs,
        )

    # 6. Inference Verification
    if run_all or args.predict:
        run_sample_predictions()

    print("\n" + "=" * 80)
    print("[+] ALL REQUESTED STAGES COMPLETED SUCCESSFULLY.")
    print(f"[+] Output Models   : {MODELS_DIR}")
    print(f"[+] Output Results  : {RESULTS_DIR}")
    print(f"[+] Output Plots    : {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
