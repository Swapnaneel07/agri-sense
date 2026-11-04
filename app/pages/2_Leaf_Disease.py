"""
AgriSense AI - Page 2: Leaf Disease Detection & Edge AI
Demonstrating MobileNetV2 with ExecuTorch edge deployment (~3.5MB).
"""

import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import config, loaders, predict
from modules.leaf_disease.inference import generate_unet_heatmap

st.set_page_config(page_title="Leaf Disease Detection", page_icon="🍃", layout="wide")

st.title("🍃 Module 3: Leaf Disease Detection & Edge AI")
st.markdown("MobileNetV2 transfer learning deployed targeting edge devices with **ExecuTorch (INT8 quantization, 3.5 MB)**.")

# Status badge
leaf_bundle = loaders.load_leaf_bundle()
badge_col1, badge_col2, badge_col3 = st.columns([1, 1, 2])
badge_col1.metric("Deployment Runtime", "ExecuTorch INT8")
badge_col2.metric("Binary Footprint", "3.5 MB")
badge_col3.metric("Offline Capability", "100% On-Device (Airplane Mode)")

st.markdown("---")

col_input, col_pred = st.columns([1, 1.2])

with col_input:
    st.subheader("1. Provide Leaf Foliage Photo")

    # Dropdown for sample leaf
    sample_files = list(config.SAMPLE_LEAVES_DIR.glob("*.png")) + list(config.SAMPLE_LEAVES_DIR.glob("*.jpg"))
    sample_names = ["-- Upload my own photo --"] + [f.name for f in sample_files]

    selected_sample = st.selectbox("Quick Test with Sample Leaves:", sample_names)

    uploaded_file = st.file_uploader("Or Upload Leaf Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])

    # Resolve image
    selected_image = None
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
    elif selected_sample != "-- Upload my own photo --":
        sample_path = config.SAMPLE_DATA_DIR / "leaves" / selected_sample
        if sample_path.exists():
            selected_image = Image.open(sample_path)
    else:
        # Default fallback sample
        if sample_files:
            selected_image = Image.open(sample_files[0])

    if selected_image is not None:
        st.image(selected_image, caption="Target Foliage Image", width=280)

with col_pred:
    st.subheader("2. Diagnostic Inference")
    if selected_image is not None:
        diag = predict.predict_disease(selected_image)

        if diag.get("mock", False):
            st.warning("⚠️ **Mock Mode Active:** ExecuTorch model (`leaf_mobilenetv2.pte`) not yet detected in `models/`. Showing simulated benchmark.")

        # Primary class alert
        is_healthy = diag.get("is_healthy", False)
        if is_healthy:
            st.success(f"### Diagnosis: **{diag['primary_class']}** (Healthy Foliage)")
        else:
            st.error(f"### Diagnosis: **{diag['primary_class']}** ({diag.get('category', 'Pathogen')})")

        m1, m2 = st.columns(2)
        m1.metric("Confidence Score", f"{diag['confidence']}%")
        m2.metric("Inference Latency", f"{diag['latency_ms']} ms", help="On-device ExecuTorch latency")

        # Top 3 Bar Chart
        st.markdown("#### Top-3 Class Probabilities")
        top3_df = pd.DataFrame(diag.get("top_3", []))
        if not top3_df.empty:
            st.bar_chart(top3_df.set_index("class")[["confidence"]])

        # Optional UNet extension
        st.markdown("#### 🔬 UNet Disease Localization Overlay")
        overlay, affected_pct = generate_unet_heatmap(selected_image)
        st.image(overlay, caption=f"UNet Lesion Segmentation ({affected_pct}% leaf coverage)", width=280)

