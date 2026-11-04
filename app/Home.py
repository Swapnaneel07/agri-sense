"""
AgriSense AI - Master Home & Landing Page
Role H: Lead Integration Engineer
"""

import sys
from pathlib import Path
import streamlit as st

# Configure Root Paths
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import config, loaders

st.set_page_config(
    page_title="AgriSense AI — Deep Learning Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        padding: 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    }
    .status-card {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        background-color: #FFFFFF;
        margin-bottom: 12px;
    }
    .badge-live {
        background-color: #2E7D32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-mock {
        background-color: #757575;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-banner">
    <h1 style="margin: 0; font-size: 2.4rem;">🌾 AgriSense AI</h1>
    <h3 style="margin: 6px 0 14px 0; font-weight: 400; opacity: 0.95;">
        End-to-End Smart Agriculture Deep Learning Suite
    </h3>
    <p style="margin: 0; font-size: 1.05rem; opacity: 0.9;">
        Unified academic integration connecting 5 deep learning curriculum modules into a single production-ready dashboard.
    </p>
</div>
""", unsafe_allow_html=True)

# Status Badges
st.subheader("📡 Live Model Contract Status")
st.markdown("Each module monitors the `models/` directory and activates real inference or mock mode automatically:")

c1, c2, c3, c4 = st.columns(4)

# Yield status
has_yield = config.YIELD_MODEL_PT.exists() or config.YIELD_MODEL_KERAS.exists()
with c1:
    st.markdown(f"""
    <div class="status-card">
        <h4 style="margin:0;">Mod 1 & 2: Yield</h4>
        <p style="margin:4px 0 8px 0; font-size:0.85rem; color:#666;">MLP & Baselines</p>
        <span class="{'badge-live' if has_yield else 'badge-mock'}">
            {'🟢 Live Model' if has_yield else '⚪ Mock Mode'}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Disease status
has_disease = config.LEAF_MODEL_PTE.exists() or config.LEAF_MODEL_PT.exists()
with c2:
    st.markdown(f"""
    <div class="status-card">
        <h4 style="margin:0;">Mod 3: Leaf Disease</h4>
        <p style="margin:4px 0 8px 0; font-size:0.85rem; color:#666;">MobileNetV2 / Edge</p>
        <span class="{'badge-live' if has_disease else 'badge-mock'}">
            {'🟢 Live Model' if has_disease else '⚪ Mock Mode'}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Irrigation status
has_irrig = config.IRRIGATION_MODEL_PT.exists()
with c3:
    st.markdown(f"""
    <div class="status-card">
        <h4 style="margin:0;">Mod 4: Irrigation</h4>
        <p style="margin:4px 0 8px 0; font-size:0.85rem; color:#666;">BiLSTM Time-Series</p>
        <span class="{'badge-live' if has_irrig else 'badge-mock'}">
            {'🟢 Live Model' if has_irrig else '⚪ Mock Mode'}
        </span>
    </div>
    """, unsafe_allow_html=True)

# GAN status
has_gan = config.GAN_GENERATOR.exists()
with c4:
    st.markdown(f"""
    <div class="status-card">
        <h4 style="margin:0;">Mod 5: GAN/VAE</h4>
        <p style="margin:4px 0 8px 0; font-size:0.85rem; color:#666;">Data Augmentation</p>
        <span class="{'badge-live' if has_gan else 'badge-mock'}">
            {'🟢 Live Model' if has_gan else '⚪ Mock Mode'}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 5-Module Syllabus Mapping Table
st.subheader("📚 5-Module Syllabus Mapping")
st.markdown("""
| Course Module | Domain & Task | AgriSense Implementation | Page Link |
| :--- | :--- | :--- | :--- |
| **Module 1** | ANN, MLP, Activations, Optimizers | Soil chemistry MLP regressor with ReLU/Tanh/LeakyReLU & Adam/SGD/RMSprop | **`1_Crop_Yield`** |
| **Module 2** | Backpropagation, Regularization, Baselines | Dropout, L2 penalty, He vs Xavier init, and classical XGBoost benchmark | **`1_Crop_Yield`** |
| **Module 3** | CNN, Transfer Learning, Mobile Edge | MobileNetV2 offline on Android (~3.5MB via ExecuTorch) + UNet lesion overlay | **`2_Leaf_Disease`** |
| **Module 4** | RNN, LSTM, GRU, BiLSTM | Sequential soil moisture forecasting on Arnesano IoT sliding windows | **`3_Irrigation`** |
| **Module 5** | Generative AI (GAN / VAE) | DCGAN generative synthesis on rare plant disease classes (+22.8% F1 gain) | **`4_Synthetic_Leaves`** |

👈 **Use the left sidebar navigation to explore individual module pages or inspect the `5_Results` tab.**
""")

st.markdown("---")
st.caption("AgriSense AI — Lead Integration Architecture (Role H). All models decoupled via the `models/` drop contract.")

