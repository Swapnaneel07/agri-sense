import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

models = ["lstm", "gru", "bilstm"]
results = []

for model_name in models:

    metrics_path = OUTPUT_DIR / model_name / "metrics.json"

    if not metrics_path.exists():
        print(f"{model_name.upper()} results not found. Skipping.")
        continue

    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    results.append({
        "Model": model_name.upper(),
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "Inference Time (s)": metrics["inference_time_seconds"]
    })

if not results:
    print("No model results found.")
    raise SystemExit

df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(df.to_string(index=False))

# Save comparison table
df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# RMSE comparison plot
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["RMSE"])
plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison_rmse.png", dpi=300)
plt.close()

# MAE comparison plot
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["MAE"])
plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison_mae.png", dpi=300)
plt.close()

print("\nComparison file and plots saved successfully.")