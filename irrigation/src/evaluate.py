import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from bilstm_model import BiLSTM


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "windows")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "bilstm")
OUTPUT_DIR = MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.joblib")

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

print("Loading test dataset...")

test_data = np.load(
    os.path.join(DATA_DIR, "arnesano_test.npz")
)

X_test = torch.tensor(
    test_data["X"], dtype=torch.float32
)

y_test = test_data["y"]

test_loader = DataLoader(
    TensorDataset(
        X_test,
        torch.tensor(y_test, dtype=torch.float32)
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test input shape:", X_test.shape)
print("Test target shape:", y_test.shape)


# ==========================================
# 3. LOAD SCALERS
# ==========================================

scalers = joblib.load(SCALER_PATH)

x_scaler = scalers["x_scaler"]
y_scaler = scalers["y_scaler"]

with open(
    os.path.join(DATA_DIR, "arnesano_meta.json"), "r"
) as f:
    metadata = json.load(f)

features = metadata["features"]
target_index = features.index("soil_moisture")

print("Features:", len(features))
print("Target:", metadata["target"])


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

model = BiLSTM(
    input_size=len(features),
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    output_size=1
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("\nTrained BiLSTM loaded successfully!")


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

predictions = []
actual_values = []

start_time = time.perf_counter()

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        output = model(X)

        predictions.extend(
            output.cpu().numpy()
        )

        actual_values.extend(
            y.numpy()
        )

total_inference_time = time.perf_counter() - start_time

predictions = np.asarray(predictions).reshape(-1, 1)
actual_values = np.asarray(actual_values).reshape(-1, 1)


# ==========================================
# 6. INVERSE TRANSFORM
# ==========================================

predictions_original = y_scaler.inverse_transform(
    predictions
)

actual_original = y_scaler.inverse_transform(
    actual_values
)


# ==========================================
# 7. CALCULATE MODEL METRICS
# ==========================================

rmse = np.sqrt(
    mean_squared_error(actual_original, predictions_original)
)

mae = mean_absolute_error(
    actual_original, predictions_original
)

r2 = r2_score(
    actual_original, predictions_original
)

inference_time_per_sample = (
    total_inference_time / len(X_test)
)


# ==========================================
# 8. PERSISTENCE BASELINE
# ==========================================

# Predict the next soil moisture value
# using the last observed soil moisture value.

last_moisture_scaled = X_test[:, -1, target_index].numpy()

x_mean = x_scaler.mean_[target_index]
x_std = x_scaler.scale_[target_index]

last_moisture_original = (
    last_moisture_scaled * x_std + x_mean
).reshape(-1, 1)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_original,
        last_moisture_original
    )
)

baseline_mae = mean_absolute_error(
    actual_original,
    last_moisture_original
)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

np.save(
    os.path.join(OUTPUT_DIR, "preds_test.npy"),
    predictions_original
)

np.save(
    os.path.join(OUTPUT_DIR, "actual_test.npy"),
    actual_original
)


# ==========================================
# 10. SAVE METRICS
# ==========================================

metrics = {
    "model": "BiLSTM",
    "dataset": "Arnesano",
    "target": "soil_moisture",
    "horizon_steps": 1,
    "test_samples": int(len(X_test)),
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R2": float(r2),
    "inference_time_seconds": float(total_inference_time),
    "inference_time_per_sample_seconds": float(
        inference_time_per_sample
    ),
    "persistence_RMSE": float(baseline_rmse),
    "persistence_MAE": float(baseline_mae)
}

with open(
    os.path.join(OUTPUT_DIR, "metrics.json"), "w"
) as f:

    json.dump(metrics, f, indent=4)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("BiLSTM EVALUATION RESULTS")
print("====================================")

print("RMSE:", round(rmse, 5))
print("MAE:", round(mae, 5))
print("R2 Score:", round(r2, 5))

print("\nPersistence Baseline:")
print("Baseline RMSE:", round(baseline_rmse, 5))
print("Baseline MAE:", round(baseline_mae, 5))

print("\nInference time:", round(total_inference_time, 5), "seconds")
print(
    "Inference time per sample:",
    round(inference_time_per_sample * 1000, 5),
    "ms"
)

print("\nSaved:")
print("preds_test.npy")
print("actual_test.npy")
print("metrics.json")

print("\nBiLSTM evaluation completed successfully!")