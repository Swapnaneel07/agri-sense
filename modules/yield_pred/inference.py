"""
AgriSense AI - Modules 1 & 2: Crop Yield Prediction Inference Engine
Handles user inputs (N, P, K, temperature, humidity, pH, rainfall) and generates:
- Yield prediction in metric tons / hectare
- Recommended crop match
- Agronomic soil health advisory
"""

from pathlib import Path
import numpy as np

try:
    import torch
    import joblib
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models import CropYieldMLP

BASE_DIR = Path(__file__).resolve().parents[2]


def predict_crop_yield(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predicts yield based on 7 soil & climate features.
    """
    model_path = BASE_DIR / "saved_models" / "yield_pred" / "mlp_best.pt"
    scaler_path = BASE_DIR / "saved_models" / "yield_pred" / "scaler.pkl"

    raw_features = np.array([[n, p, k, temperature, humidity, ph, rainfall]], dtype=np.float32)

    # Check if teammate weights exist
    if TORCH_AVAILABLE and model_path.exists():
        try:
            device = torch.device("cpu")
            model = CropYieldMLP(input_dim=7, hidden_dims=[64, 32, 16], output_dim=1)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()

            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                scaled_x = scaler.transform(raw_features)
            else:
                scaled_x = raw_features

            with torch.no_grad():
                tensor = torch.tensor(scaled_x, dtype=torch.float32)
                pred_yield = float(model(tensor).item())
            engine_status = "Trained PyTorch MLP"
        except Exception as e:
            pred_yield = _agronomic_heuristic_yield(n, p, k, temperature, humidity, ph, rainfall)
            engine_status = f"Fallback Simulation (Error loading weights: {e})"
    else:
        pred_yield = _agronomic_heuristic_yield(n, p, k, temperature, humidity, ph, rainfall)
        engine_status = "Integrated Simulation (Awaiting Role D weights)"

    # Identify optimal crop recommendation based on NPK & climatic profile
    crop_rec = _recommend_crop(n, p, k, rainfall, ph)

    return {
        "predicted_yield_tons_ha": round(max(0.5, float(pred_yield)), 2),
        "recommended_crop": crop_rec["crop"],
        "expected_range": f"{round(max(0.4, pred_yield * 0.9), 1)} - {round(pred_yield * 1.1, 1)} tons/ha",
        "soil_health_grade": crop_rec["soil_grade"],
        "advisory": crop_rec["advisory"],
        "engine_status": engine_status
    }


def _agronomic_heuristic_yield(n, p, k, temp, hum, ph, rain):
    # Normalized academic response curve
    opt_ph = 1.0 - min(abs(ph - 6.5) / 3.0, 0.8)
    opt_npk = min((n / 100.0 + p / 50.0 + k / 50.0) / 3.0, 1.5)
    opt_climate = min((rain / 1200.0) * (hum / 80.0), 1.3)
    base_yield = 3.2 * opt_ph * (0.4 + 0.6 * opt_npk) * (0.5 + 0.5 * opt_climate)
    return round(float(base_yield), 2)


def _recommend_crop(n, p, k, rain, ph):
    if rain > 1000 and n > 70:
        return {
            "crop": "Rice / Paddy",
            "soil_grade": "A (High Hydration)",
            "advisory": "High nitrogen and abundant rainfall favor wetland rice cultivation."
        }
    elif n > 80 and p > 40:
        return {
            "crop": "Maize (Corn)",
            "soil_grade": "A+ (Nutrient Rich)",
            "advisory": "High nitrogen and phosphorus reserves provide exceptional yield potential for maize."
        }
    elif k > 50 and 6.0 <= ph <= 7.2:
        return {
            "crop": "Wheat / Barley",
            "soil_grade": "A (Balanced Potassium)",
            "advisory": "Optimal pH with robust potassium supports grain development and drought resilience."
        }
    elif ph < 6.0:
        return {
            "crop": "Tea / Coffee / Potatoes",
            "soil_grade": "B (Acidic Soil)",
            "advisory": "Soil pH is slightly acidic. Ideal for acid-tolerant crops or consider lime amendment."
        }
    else:
        return {
            "crop": "Pulses / Legumes (Chickpea / Lentil)",
            "soil_grade": "B+ (Nitrogen-Fixing Suitable)",
            "advisory": "Legume crops will help fix atmospheric nitrogen and enrich depleted soil beds."
        }

