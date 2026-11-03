import sys
import json
import joblib
import torch
import numpy as np

from pathlib import Path

# Import BiLSTM model
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bilstm_model import BiLSTM


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "outputs" / "bilstm" / "best.pt"
SCALER_PATH = BASE_DIR / "data" / "scaler.joblib"
META_PATH = BASE_DIR / "windows" / "arnesano_meta.json"

DEVICE = torch.device("cpu")


class AgriSensePredictor:

    def __init__(self):

        with open(META_PATH, "r") as file:
            self.metadata = json.load(file)

        scalers = joblib.load(SCALER_PATH)

        self.x_scaler = scalers["x_scaler"]
        self.y_scaler = scalers["y_scaler"]

        self.features = self.metadata["features"]
        self.window_size = self.metadata["window_size"]

        self.model = BiLSTM(
            input_size=len(self.features),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, readings):

        readings = np.asarray(readings, dtype=np.float32)

        if readings.shape != (self.window_size, len(self.features)):
            raise ValueError(
                f"Expected shape "
                f"({self.window_size}, {len(self.features)}), "
                f"received {readings.shape}"
            )

        if not np.isfinite(readings).all():
            raise ValueError("Input contains missing or invalid values.")

        # Scale input
        scaled = self.x_scaler.transform(readings)

        # Convert to tensor
        tensor = torch.tensor(
            scaled,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()

        # Convert prediction to original soil moisture units
        moisture = self.y_scaler.inverse_transform(
            prediction.reshape(-1, 1)
        )[0, 0]

        return {
            "predicted_soil_moisture": round(float(moisture), 2),
            "forecast_horizon": "10 minutes",
            "advice": self.get_advice(moisture)
        }

    @staticmethod
    def get_advice(moisture):

        if moisture < 20:
            return "Very low moisture. Irrigation may be required."

        elif moisture < 35:
            return "Low moisture. Monitor soil and consider irrigation."

        elif moisture < 70:
            return "Moderate moisture. Continue monitoring."

        else:
            return "High moisture. Irrigation may not be necessary."


if __name__ == "__main__":

    predictor = AgriSensePredictor()

    # Load one real test window
    test_data = np.load(
        BASE_DIR / "windows" / "arnesano_test.npz"
    )

    sample = test_data["X"][0]

    result = predictor.predict(
        predictor.x_scaler.inverse_transform(
            sample
        )
    )

    print("\nAgriSense AI - Soil Moisture Forecast")
    print("-------------------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")