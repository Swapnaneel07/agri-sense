"""Train and evaluate the LSTM or GRU soil-moisture forecaster."""

import argparse
import json
import os
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "windows")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecurrentRegressor(nn.Module):
    def __init__(self, model_type, input_size):
        super().__init__()
        recurrent_layer = {"lstm": nn.LSTM, "gru": nn.GRU}[model_type]
        self.recurrent = recurrent_layer(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, inputs):
        recurrent_output = self.recurrent(inputs)
        hidden = recurrent_output[1][0] if isinstance(self.recurrent, nn.LSTM) else recurrent_output[1]
        return self.output(self.dropout(hidden[-1]))


def load_dataset(split):
    data = np.load(os.path.join(DATA_DIR, f"arnesano_{split}.npz"))
    inputs = torch.tensor(data["X"], dtype=torch.float32)
    targets = torch.tensor(data["y"], dtype=torch.float32)
    return TensorDataset(inputs, targets)


def save_plots(model_type, history, actual, predictions):
    plot_dir = os.path.join(OUTPUT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation Loss")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch: {best_epoch}")
    plt.title(f"{model_type.upper()} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_type}_training_loss.png"), dpi=300)
    plt.close()

    actual = actual.reshape(-1)
    predictions = predictions.reshape(-1)
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predictions, alpha=0.35)
    minimum = min(actual.min(), predictions.min())
    maximum = max(actual.max(), predictions.max())
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Prediction",
    )
    plt.title(f"{model_type.upper()}: Actual vs Predicted Soil Moisture")
    plt.xlabel("Actual Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_type}_actual_vs_predicted.png"), dpi=300)
    plt.close()


def train_and_evaluate(model_type):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_dir = os.path.join(OUTPUT_ROOT, model_type)
    os.makedirs(model_dir, exist_ok=True)

    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")
    test_dataset = load_dataset("test")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    input_size = train_dataset.tensors[0].shape[-1]

    model = RecurrentRegressor(model_type, input_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"Training {model_type.upper()} on {DEVICE}")
    print(f"Train/validation/test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                val_loss += criterion(model(inputs), targets).item() * inputs.size(0)
        val_loss /= len(val_dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    training_time = time.time() - start_time
    with open(os.path.join(model_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    config = {
        "model": model_type.upper(),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "training_time_seconds": training_time,
    }
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best.pt"), map_location=DEVICE, weights_only=True)
    )
    model.eval()
    scaled_predictions = []
    scaled_actual = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(DEVICE)).cpu().numpy()
            scaled_predictions.append(outputs)
            scaled_actual.append(targets.numpy())
    inference_time = time.perf_counter() - inference_start
    scaled_predictions = np.concatenate(scaled_predictions).reshape(-1, 1)
    scaled_actual = np.concatenate(scaled_actual).reshape(-1, 1)

    scalers = joblib.load(os.path.join(BASE_DIR, "data", "scaler.joblib"))
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    predictions = y_scaler.inverse_transform(scaled_predictions)
    actual = y_scaler.inverse_transform(scaled_actual)

    with open(os.path.join(DATA_DIR, "arnesano_meta.json"), encoding="utf-8") as file:
        metadata = json.load(file)
    target_index = metadata["features"].index(metadata["target"])
    test_inputs = test_dataset.tensors[0]
    last_moisture_scaled = test_inputs[:, -1, target_index].numpy()
    baseline = (
        last_moisture_scaled * x_scaler.scale_[target_index]
        + x_scaler.mean_[target_index]
    ).reshape(-1, 1)

    metrics = {
        "model": model_type.upper(),
        "dataset": "Arnesano",
        "target": metadata["target"],
        "horizon_steps": metadata["horizons"][0],
        "test_samples": int(len(test_dataset)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "MAE": float(mean_absolute_error(actual, predictions)),
        "R2": float(r2_score(actual, predictions)),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_sample_seconds": float(inference_time / len(test_dataset)),
        "persistence_RMSE": float(np.sqrt(mean_squared_error(actual, baseline))),
        "persistence_MAE": float(mean_absolute_error(actual, baseline)),
    }
    np.save(os.path.join(model_dir, "preds_test.npy"), predictions)
    np.save(os.path.join(model_dir, "actual_test.npy"), actual)
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    save_plots(model_type, history, actual, predictions)

    print(
        f"{model_type.upper()} complete: best epoch {best_epoch}, "
        f"RMSE {metrics['RMSE']:.5f}, MAE {metrics['MAE']:.5f}, "
        f"R2 {metrics['R2']:.5f}"
    )
    print(f"Artifacts saved to: {model_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM or GRU soil-moisture model.")
    parser.add_argument("--model", choices=("lstm", "gru"), required=True)
    arguments = parser.parse_args()
    train_and_evaluate(arguments.model)
