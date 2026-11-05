import matplotlib
matplotlib.use("Agg")

import os
import sys
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "yield_module")
DATA_PATH = os.path.join(ROOT, "data", "raw", "yield_df.csv")

for d in ["data/raw", "data/processed", "models", "src", "results/eda", "results/plots", "results/grid"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

def find_dataset():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(BASE_DIR, "dataset", "yield_df.csv"),
        os.path.join(BASE_DIR, "dataset", "yield_df.csv.csv"),
        os.path.join(BASE_DIR, "yield_df.csv"),
        os.path.join(home, "Downloads", "yield_df.csv"),
        os.path.join(home, "Downloads", "yield_df.csv.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            shutil.copy(c, DATA_PATH)
            print(f"Using dataset: {c}")
            return DATA_PATH
    raise FileNotFoundError("yield_df.csv not found in AgriSense\\dataset or Downloads.")

DATA_PATH = find_dataset()

SEED = 42
SEEDS = [42, 43, 44]
HIDDEN = (128, 64, 32)
TRAIN_CFG = {"epochs": 100, "batch_size": 128, "patience": 10}

MODEL_SRC = r'''
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def make_optimizer(name, lr):
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    raise ValueError("unknown optimizer: " + name)

def build_mlp(config, input_dim):
    l2 = config.get("l2", 0.0)
    reg = regularizers.l2(l2) if l2 else None
    init = config.get("initializer", "glorot_uniform")
    act = config.get("activation", "relu")
    dropout = config.get("dropout", 0.0)
    model = keras.Sequential([keras.Input(shape=(input_dim,))])
    for units in config.get("hidden", (128, 64, 32)):
        if act == "leaky_relu":
            model.add(layers.Dense(units, kernel_initializer=init, kernel_regularizer=reg))
            model.add(layers.LeakyReLU())
        else:
            model.add(layers.Dense(units, activation=act, kernel_initializer=init, kernel_regularizer=reg))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=make_optimizer(config.get("optimizer", "adam"), config.get("lr", 1e-3)),
        loss="mse",
        metrics=["mae"],
    )
    return model
'''

DATA_PREP_SRC = r'''
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

TARGET = "hg/ha_yield"
CAT = ["Area", "Item"]
NUM = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
FEATURES = CAT + NUM
LOG_COLS = ["pesticides_tonnes"]
WINSOR = ["average_rain_fall_mm_per_year", "pesticides_tonnes"]

def load_clean(path):
    raw = pd.read_csv(path)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    df = raw.drop_duplicates().dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return raw, df

def apply_logs(X):
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X

def clip_with(X, bounds):
    X = X.copy()
    for c, (lo, hi) in bounds.items():
        X[c] = X[c].clip(lo, hi)
    return X

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_eda(raw, df, out):
    os.makedirs(out, exist_ok=True)
    rep = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "duplicate_rows_removed": int(raw.duplicated().sum()),
        "missing_values_in_features_or_target": int(raw[FEATURES + [TARGET]].isna().sum().sum()),
        "areas": int(df["Area"].nunique()),
        "crops": int(df["Item"].nunique()),
        "years": [int(df["Year"].min()), int(df["Year"].max())],
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "outliers_outside_1_99_percentile": {},
    }
    for c in WINSOR + [TARGET]:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        rep["outliers_outside_1_99_percentile"][c] = {
            "below_p1": int((df[c] < lo).sum()),
            "above_p99": int((df[c] > hi).sum()),
        }
    with open(os.path.join(out, "eda_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    df[NUM + [TARGET]].describe().T.to_csv(os.path.join(out, "numeric_summary.csv"))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=40, ax=ax)
    ax.set_xscale("log")
    ax.set_title("Yield distribution (log scale)")
    _save(fig, os.path.join(out, "target_hist.png"))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[NUM + [TARGET]].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Spearman correlation")
    _save(fig, os.path.join(out, "correlation.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Item", y=TARGET, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Yield by crop (log scale)")
    plt.xticks(rotation=30, ha="right")
    _save(fig, os.path.join(out, "yield_by_crop.png"))

    med = df.groupby("Area")[TARGET].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(med))))
    med.plot.barh(ax=ax)
    ax.set_title("Median yield by country (hg/ha)")
    _save(fig, os.path.join(out, "yield_by_country.png"))
    return rep

def make_split(n, seed, path):
    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed)
    np.savez(path, train=tr, val=va, test=te)
    return tr, va, te

def prepare(df, seed, processed_dir, models_dir):
    tr, va, te = make_split(len(df), seed, os.path.join(processed_dir, "split.npz"))
    X = apply_logs(df[FEATURES])
    y_raw = df[TARGET].values.astype(float)
    y = np.log1p(y_raw)
    bounds = {c: [float(X.iloc[tr][c].quantile(0.01)), float(X.iloc[tr][c].quantile(0.99))] for c in WINSOR}
    Xw = clip_with(X, bounds)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM),
    ]).fit(Xw.iloc[tr])
    y_scaler = StandardScaler().fit(y[tr].reshape(-1, 1))
    joblib.dump({
        "preprocessor": pre, "y_scaler": y_scaler,
        "cat": CAT, "num": NUM, "log_cols": LOG_COLS,
        "winsor_bounds": bounds, "target_transform": "log1p",
    }, os.path.join(models_dir, "yield_preprocessor.joblib"))
    Xs = pre.transform(Xw).astype(np.float32)
    ys = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
    return {
        "Xtr": Xs[tr], "ytr": ys[tr],
        "Xva": Xs[va], "yva": ys[va], "yva_true": y_raw[va],
        "Xte": Xs[te], "yte": ys[te], "yte_true": y_raw[te],
        "y_scaler": y_scaler, "splits": (tr, va, te), "y_raw": y_raw,
        "bounds": bounds, "pre": pre,
    }
'''

TRAIN_SRC = r'''
import time
import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import build_mlp

def to_units(scaled, data):
    return np.expm1(data["y_scaler"].inverse_transform(scaled)).ravel()

def metrics(true, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "MAE": float(mean_absolute_error(true, pred)),
        "R2": float(r2_score(true, pred)),
    }

def fit_model(config, seed, data, epochs, batch_size, patience):
    keras.utils.set_random_seed(seed)
    model = build_mlp(config, data["Xtr"].shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(data["Xtr"], data["ytr"], validation_data=(data["Xva"], data["yva"]),
                     epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return model, hist.history, time.time() - t0

def evaluate(model, data):
    vp = to_units(model.predict(data["Xva"], verbose=0), data)
    tp = to_units(model.predict(data["Xte"], verbose=0), data)
    return metrics(data["yva_true"], vp), metrics(data["yte_true"], tp)

def run_config(config, seed, data, train_cfg):
    model, hist, elapsed = fit_model(config, seed, data, train_cfg["epochs"],
                                     train_cfg["batch_size"], train_cfg["patience"])
    vm, tm = evaluate(model, data)
    row = {
        "activation": config["activation"], "optimizer": config["optimizer"],
        "lr": config["lr"], "seed": seed,
        "val_RMSE": vm["RMSE"], "val_R2": vm["R2"],
        "test_RMSE": tm["RMSE"], "test_MAE": tm["MAE"], "test_R2": tm["R2"],
        "epochs_run": len(hist["loss"]), "train_time_s": round(elapsed, 2),
        "params": model.count_params(),
    }
    keras.backend.clear_session()
    return row, hist
'''

RUN_GRID_SRC = r'''
import itertools
import pandas as pd
from train import run_config

ACTIVATIONS = ["relu", "leaky_relu", "elu", "tanh", "sigmoid"]
OPTIMIZER_LRS = {
    "sgd": [1e-2, 1e-3, 1e-4],
    "sgd_momentum": [1e-2, 1e-3, 1e-4],
    "rmsprop": [1e-2, 1e-3, 1e-4],
    "adam": [1e-2, 1e-3, 1e-4],
}
PAIRS = [(o, lr) for o, lrs in OPTIMIZER_LRS.items() for lr in lrs]

def base_config(activation, optimizer, lr, hidden):
    return {"hidden": hidden, "activation": activation, "optimizer": optimizer, "lr": lr,
            "dropout": 0.0, "l2": 0.0, "initializer": "glorot_uniform"}

def run_full_grid(data, seeds, hidden, train_cfg):
    combos = list(itertools.product(ACTIVATIONS, PAIRS))
    total = len(combos) * len(seeds)
    n = 0
    records, histories = [], []
    for act, (opt, lr) in combos:
        cfg = base_config(act, opt, lr, hidden)
        for s in seeds:
            n += 1
            row, hist = run_config(cfg, s, data, train_cfg)
            print(f"[{n}/{total}] {act} | {opt} lr={lr:g} | seed {s} | "
                  f"test RMSE {row['test_RMSE']:.0f} | R2 {row['test_R2']:.3f} | epochs {row['epochs_run']}",
                  flush=True)
            records.append(row)
            histories.append({"activation": act, "optimizer": opt, "lr": lr, "seed": s, "hist": hist})
    return pd.DataFrame(records), histories
'''

PREDICT_SRC = r'''
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))
_cache = {}

def _load():
    if not _cache:
        _cache["art"] = joblib.load(os.path.join(BASE, "models", "yield_preprocessor.joblib"))
        _cache["model"] = keras.models.load_model(os.path.join(BASE, "models", "yield_mlp.keras"))
    return _cache["art"], _cache["model"]

def predict_yield(features):
    art, model = _load()
    row = pd.DataFrame([{c: features[c] for c in art["cat"] + art["num"]}])
    for c in art["log_cols"]:
        row[c] = np.log1p(row[c])
    for c, (lo, hi) in art["winsor_bounds"].items():
        row[c] = row[c].clip(lo, hi)
    x = art["preprocessor"].transform(row).astype(np.float32)
    scaled = model.predict(x, verbose=0)
    return float(np.expm1(art["y_scaler"].inverse_transform(scaled))[0, 0])
'''

files = {
    "src/model.py": MODEL_SRC,
    "src/data_prep.py": DATA_PREP_SRC,
    "src/train.py": TRAIN_SRC,
    "src/run_grid.py": RUN_GRID_SRC,
    "predict.py": PREDICT_SRC,
}
for path, text in files.items():
    with open(os.path.join(ROOT, path), "w") as f:
        f.write(text.lstrip("\n"))

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
for m in ["model", "train", "run_grid", "data_prep", "predict"]:
    sys.modules.pop(m, None)

from data_prep import load_clean, run_eda, prepare, apply_logs, clip_with, TARGET, FEATURES
from train import fit_model, evaluate, to_units
from run_grid import run_full_grid, ACTIVATIONS, base_config
from predict import predict_yield

raw, df = load_clean(DATA_PATH)
eda = run_eda(raw, df, os.path.join(ROOT, "results", "eda"))
print(json.dumps(eda, indent=2))

data = prepare(df, SEED, os.path.join(ROOT, "data", "processed"), os.path.join(ROOT, "models"))

res, hist_records = run_full_grid(data, SEEDS, HIDDEN, TRAIN_CFG)
res.to_csv(os.path.join(ROOT, "results", "grid", "grid_results_all_seeds.csv"), index=False)

curve_rows = []
for r in hist_records:
    for ep, (l, vl) in enumerate(zip(r["hist"]["loss"], r["hist"]["val_loss"]), start=1):
        curve_rows.append({"activation": r["activation"], "optimizer": r["optimizer"], "lr": r["lr"],
                           "seed": r["seed"], "epoch": ep, "loss": l, "val_loss": vl})
pd.DataFrame(curve_rows).to_csv(os.path.join(ROOT, "results", "grid", "loss_histories.csv"), index=False)

AGG = dict(
    val_RMSE_mean=("val_RMSE", "mean"), val_RMSE_std=("val_RMSE", "std"),
    test_RMSE_mean=("test_RMSE", "mean"), test_RMSE_std=("test_RMSE", "std"),
    test_R2_mean=("test_R2", "mean"), test_R2_std=("test_R2", "std"),
    epochs_mean=("epochs_run", "mean"), time_mean_s=("train_time_s", "mean"),
)
full = res.groupby(["activation", "optimizer", "lr"]).agg(**AGG).reset_index().sort_values("val_RMSE_mean")
full.to_csv(os.path.join(ROOT, "results", "grid", "summary_full_grid.csv"), index=False)

base = full[(full.activation == "relu") & (full.optimizer == "adam") & (full.lr == 1e-3)].iloc[0]
assert base["test_R2_mean"] > 0, "baseline (ReLU, Adam 1e-3) does not beat predicting the mean"

act_tbl = full[(full.optimizer == "adam") & (full.lr == 1e-3)].set_index("activation").loc[ACTIVATIONS].reset_index()
act_tbl.to_csv(os.path.join(ROOT, "results", "grid", "activation_comparison.csv"), index=False)

best = full.iloc[0]
best_act, best_opt, best_lr = best["activation"], best["optimizer"], float(best["lr"])
opt_tbl = full[full.activation == best_act].reset_index(drop=True)
opt_tbl.to_csv(os.path.join(ROOT, "results", "grid", "optimizer_comparison.csv"), index=False)

print("\nActivation comparison (Adam lr=1e-3):")
print(act_tbl[["activation", "val_RMSE_mean", "test_RMSE_mean", "test_R2_mean", "epochs_mean"]].round(3).to_string(index=False))
print(f"\nOptimizer comparison (activation {best_act}):")
print(opt_tbl[["optimizer", "lr", "val_RMSE_mean", "test_RMSE_mean", "test_R2_mean", "epochs_mean"]].round(3).to_string(index=False))
print(f"\nBest: {best_act} | {best_opt} lr={best_lr:g}")

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def curves(pred, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in hist_records:
        if r["seed"] == SEED and pred(r):
            ax.plot(r["hist"]["val_loss"], label=f'{r["activation"]} / {r["optimizer"]} lr={r["lr"]:g}')
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (log-scaled target)")
    ax.set_title(f"{title} (seed {SEED})")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(ROOT, "results", "plots", fname))

def bars(labels, values, errors, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(labels), list(values), yerr=None if errors is None else list(errors), capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    _save(fig, os.path.join(ROOT, "results", "plots", fname))

curves(lambda r: r["optimizer"] == "adam" and r["lr"] == 1e-3,
       "Activation comparison, Adam lr=1e-3", "activation_val_loss.png")
curves(lambda r: r["activation"] == best_act,
       f"Optimizer comparison, activation {best_act}", "optimizer_val_loss.png")
bars(act_tbl["activation"], act_tbl["test_RMSE_mean"], act_tbl["test_RMSE_std"],
     "test RMSE (hg/ha), mean ± std", "Activation test RMSE", "activation_test_rmse.png")
opt_labels = [f'{o} lr={lr:g}' for o, lr in zip(opt_tbl["optimizer"], opt_tbl["lr"])]
bars(opt_labels, opt_tbl["test_RMSE_mean"], opt_tbl["test_RMSE_std"],
     "test RMSE (hg/ha), mean ± std", "Optimizer test RMSE", "optimizer_test_rmse.png")
bars(opt_labels, opt_tbl["epochs_mean"], None,
     "epochs to early stop", "Optimizer convergence", "optimizer_epochs.png")

final_cfg = base_config(best_act, best_opt, best_lr, HIDDEN)
final_model, final_hist, _ = fit_model(final_cfg, SEED, data, TRAIN_CFG["epochs"],
                                       TRAIN_CFG["batch_size"], TRAIN_CFG["patience"])
final_model.save(os.path.join(ROOT, "models", "yield_mlp.keras"))
vm, tm = evaluate(final_model, data)

best_json = {
    "activation": best_act, "optimizer": best_opt, "lr": best_lr,
    "hidden": list(HIDDEN), "epochs_max": TRAIN_CFG["epochs"],
    "batch_size": TRAIN_CFG["batch_size"], "early_stopping_patience": TRAIN_CFG["patience"],
    "target_transform": "log1p", "seeds_for_selection": SEEDS, "final_model_seed": SEED,
    "selection_val_RMSE_mean": round(float(best["val_RMSE_mean"]), 2),
    "test_RMSE_mean": round(float(best["test_RMSE_mean"]), 2),
    "test_RMSE_std": round(float(best["test_RMSE_std"]), 2),
    "test_R2_mean": round(float(best["test_R2_mean"]), 4),
    "final_test_RMSE": round(tm["RMSE"], 2),
    "final_test_R2": round(tm["R2"], 4),
}
with open(os.path.join(ROOT, "results", "grid", "best_config.json"), "w") as f:
    json.dump(best_json, f, indent=2)
with open(os.path.join(ROOT, "best_config.json"), "w") as f:
    json.dump(best_json, f, indent=2)

feature_rows = [
    ("Area", "categorical", f"country name, {eda['areas']} values", "one-hot"),
    ("Item", "categorical", f"crop, {eda['crops']} values", "one-hot"),
    ("Year", "integer", f"calendar year, {eda['years'][0]} to {eda['years'][1]}", "standardized"),
    ("average_rain_fall_mm_per_year", "float", "mm per year, clipped to train 1st/99th percentile", "standardized"),
    ("pesticides_tonnes", "float", "tonnes, log(1+x), then clipped to train 1st/99th percentile", "standardized"),
]
with open(os.path.join(ROOT, "FEATURES.md"), "w") as f:
    f.write("# Yield model features\n\n")
    f.write("Input to predict_yield(features) is a dict with these keys.\n\n")
    f.write("| Name | Type | Meaning / unit | Preprocessing |\n|---|---|---|---|\n")
    for name, typ, meaning, prep in feature_rows:
        f.write(f"| {name} | {typ} | {meaning} | {prep} |\n")
    f.write("\nTarget: hg/ha_yield (hectograms per hectare, 10,000 hg/ha = 1 tonne/ha).\n")
    f.write("The model learns log(1+yield). predict_yield returns predictions in hg/ha.\n")
    f.write("Scalers, encoders and clipping bounds are fit on the training split only.\n")

with open(os.path.join(ROOT, "handoff_E.txt"), "w") as f:
    f.write("Handoff from D to E (Module 2)\n\n")
    f.write(f"Split: data/processed/split.npz (train/val/test row indices, seed {SEED}).\n")
    f.write("Preprocessor: models/yield_preprocessor.joblib (log transforms, clipping bounds, one-hot, scaler, target log1p).\n")
    f.write("Model builder: src/model.py, build_mlp(config, input_dim). Config keys: hidden, activation, optimizer, lr, dropout, l2, initializer.\n")
    f.write(f"Training: src/train.py, fit_model and evaluate. Use the same epochs, batch size and patience ({TRAIN_CFG['epochs']}, {TRAIN_CFG['batch_size']}, {TRAIN_CFG['patience']}).\n")
    f.write(f"Baseline (ReLU, Adam 1e-3): test R2 mean {base['test_R2_mean']:.3f}, test RMSE mean {base['test_RMSE_mean']:.0f} hg/ha over {len(SEEDS)} seeds.\n")
    f.write(f"Best config: {best_act}, {best_opt}, lr={best_lr:g} (see best_config.json).\n")

with open(os.path.join(ROOT, "handoff_I.md"), "w") as f:
    f.write("# Module 1 results for slides\n\n")
    f.write(f"Setup: {eda['clean_rows']:,} rows, {eda['areas']} countries, {eda['crops']} crops, {eda['years'][0]} to {eda['years'][1]}. "
            f"70/15/15 split, seed {SEED}. MLP {HIDDEN}, batch {TRAIN_CFG['batch_size']}, up to {TRAIN_CFG['epochs']} epochs, "
            f"early stopping patience {TRAIN_CFG['patience']}, {len(SEEDS)} seeds per config. "
            "Yield and pesticides are log-transformed for training; metrics are reported in hg/ha.\n\n")
    f.write("Activation table: results/grid/activation_comparison.csv (Adam lr=1e-3).\n\n")
    f.write("Optimizer table: results/grid/optimizer_comparison.csv (best activation, all optimizer and LR pairs).\n\n")
    f.write(f"Activation: best is {best_act}. Sigmoid and tanh are expected to train slower because of vanishing gradients.\n\n")
    f.write(f"Optimizer: best is {best_opt} with lr={best_lr:g}, test RMSE {best['test_RMSE_mean']:.0f} ± {best['test_RMSE_std']:.0f} hg/ha.\n\n")
    f.write("Plots: results/plots/.\n")

with open(os.path.join(ROOT, "README.md"), "w") as f:
    f.write("# AgriSense Module 1: Yield prediction (MLP)\n\n")
    f.write("Data: data/raw/yield_df.csv (FAO-based crop yield dataset). Do not edit the raw file.\n\n")
    f.write("Reproduce: run main.py. Seeds are fixed.\n\n")
    f.write("Files: src/model.py, src/data_prep.py, src/train.py, src/run_grid.py, predict.py, FEATURES.md, results/, models/.\n")

s_idx, v_idx, t_idx = data["splits"]
assert not (set(s_idx) & set(v_idx)) and not (set(s_idx) & set(t_idx)) and not (set(v_idx) & set(t_idx)), "split overlap"
assert len(set(s_idx) | set(v_idx) | set(t_idx)) == len(df), "split does not cover all rows"
assert not df[FEATURES + [TARGET]].isna().any().any(), "missing values remain"
assert not df.duplicated().any(), "duplicate rows remain"
assert np.isfinite(data["Xtr"]).all(), "non-finite training features"
assert abs(data["y_scaler"].mean_[0] - np.log1p(data["y_raw"][s_idx]).mean()) < 1e-6, "target scaler not fit on train only"
assert tm["R2"] > 0, "final model does not beat predicting the mean"

row = df.iloc[int(t_idx[0])]
feats = {c: row[c] for c in FEATURES}
p = predict_yield(feats)
single = clip_with(apply_logs(pd.DataFrame([feats])), data["bounds"])
x = data["pre"].transform(single).astype(np.float32)
model_pred = float(to_units(final_model.predict(x, verbose=0), data)[0])
assert np.isfinite(p), "prediction is not finite"
assert abs(p - model_pred) <= 1e-3 * max(1.0, abs(model_pred)), "predict_yield disagrees with training pipeline"

try:
    predict_yield({"Area": "France"})
    raise AssertionError("missing features should raise an error")
except KeyError:
    pass
assert np.isfinite(predict_yield(dict(feats, Area="Atlantis"))), "unseen country crashed prediction"

print("\nAll tests passed.")
print(f"Final model ({best_act}, {best_opt}, lr={best_lr:g}, seed {SEED}): test RMSE {tm['RMSE']:.0f} hg/ha, R2 {tm['R2']:.3f}")
print(f"Sample: {feats['Area']} / {feats['Item']} / {feats['Year']} | actual {row[TARGET]:.0f} | predicted {p:.0f} hg/ha")
print(f"Outputs saved to {ROOT}")