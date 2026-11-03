import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "bilstm"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load training history
with open(OUTPUT_DIR / "history.json", "r") as file:
    history = json.load(file)

train_loss = history["train_loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_loss) + 1)

# 1. Training and validation loss plot
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="s", label="Validation Loss")

best_epoch = int(np.argmin(val_loss)) + 1

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

plt.title("BiLSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "bilstm_training_loss.png", dpi=300)
plt.close()

# 2. Actual vs predicted plot
actual = np.load(OUTPUT_DIR / "actual_test.npy").reshape(-1)
predicted = np.load(OUTPUT_DIR / "preds_test.npy").reshape(-1)

plt.figure(figsize=(8, 8))

plt.scatter(actual, predicted, alpha=0.35)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("BiLSTM: Actual vs Predicted Soil Moisture")
plt.xlabel("Actual Soil Moisture")
plt.ylabel("Predicted Soil Moisture")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "bilstm_actual_vs_predicted.png", dpi=300)
plt.close()

print("BiLSTM plots generated successfully!")
print("Best epoch:", best_epoch)
print("Plots saved in:", PLOT_DIR)