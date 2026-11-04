"""
AgriSense AI - Page 3: Sequential Soil Moisture & Irrigation Forecasting
Handles time-series window upload or sample window.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import config, loaders, predict

st.set_page_config(page_title="Irrigation Forecasting", page_icon="💧", layout="wide")

st.title("💧 Module 4: Sequential Irrigation Forecasting")
st.markdown("Time-series soil moisture and atmospheric conditions forecasting with sliding windows (24 time-steps × 10 min).")

bundle = loaders.load_irrigation_bundle()
meta = bundle.get("meta") or {}
expected_features = meta.get("features", [
    "soil_moisture", "soil_temperature", "soil_temperature_deep",
    "air_temperature", "relative_humidity", "rainfall",
    "air_pressure", "wind_speed", "solar_radiation",
    "hour_sin", "hour_cos"
])

col_in, col_out = st.columns([1, 1.2])

with col_in:
    st.subheader("1. Sliding Window Input")

    use_sample = st.checkbox("Use pre-packaged Arnesano IoT sample window (24 timesteps)", value=True)
    uploaded_csv = st.file_uploader("Or upload historical readings CSV (24 rows)", type=["csv"])

    df = None
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample and config.SAMPLE_IRRIGATION_CSV.exists():
        df = pd.read_csv(config.SAMPLE_IRRIGATION_CSV)

    if df is not None:
        st.markdown(f"**Loaded Window:** `{len(df)} time-steps` × `{len(df.columns)} sensor features`")

        # Validate columns
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            st.warning(f"⚠️ Missing expected sensor features: `{missing}`. Falling back to default simulation.")
        else:
            st.success("✅ All 11 sensor columns validated against `models/irrigation_meta.json`.")

        st.dataframe(df.tail(6), use_container_width=True)

with col_out:
    st.subheader("2. Forecast & Soil Health Advisory")
    if df is not None:
        # Run prediction
        try:
            feats = [c for c in expected_features if c in df.columns]
            readings = df[feats].values if len(feats) == 11 else np.zeros((24, 11))
            res = predict.predict_irrigation(readings)

            if res.get("mock", False):
                st.warning("⚠️ **Mock Mode Active:** Real weights not found in `models/irrigation_best.pt`. Showing simulated forecast.")
            else:
                st.success("🟢 **Live Inference:** Model loaded from `models/irrigation_best.pt`.")

            m1, m2 = st.columns(2)
            m1.metric("Predicted Soil Moisture (T+10 min)", f"{res['predicted_moisture']}%", delta=f"{round(res['predicted_moisture'] - df['soil_moisture'].iloc[-1], 2)}%")
            m2.metric("Forecast Horizon", res.get("horizon", "10 minutes"))

            # Recommendation
            st.info(f"**Advisory:** {res.get('advice')}")

            # Plot history + predicted point
            st.markdown("#### Soil Moisture Trajectory (Past 24 Timesteps + Forecast)")
            past_moisture = list(df["soil_moisture"].values) if "soil_moisture" in df.columns else [30.0]*24
            chart_df = pd.DataFrame({
                "Timestep": list(range(-23, 1)) + [1],
                "Soil Moisture (%)": past_moisture + [res["predicted_moisture"]]
            }).set_index("Timestep")

            st.line_chart(chart_df)

            # Winner model banner
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 12px; border-radius: 8px; border-left: 4px solid #2E7D32;">
                <strong>Model Chosen:</strong> {res.get('winner_model', 'BiLSTM')}
                <br><span style="font-size:0.85rem; color:#444;">Inspect the <strong>5_Results</strong> tab to review the 3-way LSTM vs GRU vs BiLSTM benchmarks.</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Inference error: {e}")

