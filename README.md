# 🌾 AgriSense AI — Deep Learning Suite

AgriSense AI is an end-to-end precision agriculture deep learning suite uniting 5 academic curriculum modules into a single production-ready dashboard.

---

## ⚡ Quickstart (Under 5 Minutes)

### 1. Clone & Setup Environment
```bash
git clone <repo-url>
cd agri-sense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app/Home.py
```

The application runs **100% locally and offline**. If a model file has not yet been placed into `models/`, the dashboard gracefully runs in **Mock Mode** with yellow warning banners, allowing full UI testing without crashes.

---

## 📁 Repository Layout

```text
agri-sense/
├── app/
│   ├── Home.py                  # Landing page & 5-module status overview
│   ├── pages/
│   │   ├── 1_Crop_Yield.py       # Module 1 & 2: Dynamic NPK form + MLP/XGBoost
│   │   ├── 2_Leaf_Disease.py     # Module 3: MobileNetV2 + UNet segmentation
│   │   ├── 3_Irrigation.py       # Module 4: Sequential time-series forecasting
│   │   ├── 4_Synthetic_Leaves.py # Module 5: DCGAN augmentation & F1 gain
│   │   └── 5_Results.py          # Academic benchmark comparisons (reads from results/)
│   └── core/
│       ├── config.py             # Central path constants
│       ├── loaders.py            # Cached model & schema loaders
│       ├── predict.py            # Clean unified prediction interface
│       └── mock.py               # Fallback simulation engine
├── models/                       # DROP ZONE for teammates (nobody edits app/)
│   ├── yield_schema.json         # Feature specification driving the Yield form
│   ├── irrigation_best.pt        # Trained BiLSTM weights (Member C)
│   ├── irrigation_scaler.joblib  # Scikit-learn feature scalers
│   └── leaf_classes.json         # 38 PlantVillage target classes
├── results/                      # DROP ZONE for metrics, tables, and figures
│   ├── irrigation_comparison.csv # 3-way model comparison table
│   └── plots/                    # Loss curves & evaluation charts
├── sample_data/                  # Instant demo data (zero-configuration)
│   ├── leaves/                   # 5 sample foliage photos (healthy & diseased)
│   └── irrigation/               # 24-step IoT time-series sample CSV
└── requirements.txt              # Unified dependencies
```

---

## 🤝 File Drop Contract for Teammates

> **Rule:** Nobody touches `app/`. Place your files directly into `models/` or `results/`.

| Module | Responsible Roles | Place in `models/` | Place in `results/` |
| :--- | :--- | :--- | :--- |
| **Yield** | Role D & E | `yield_mlp.pt` (or `.keras`), `yield_scaler.joblib`, `yield_xgb.joblib` | `yield_metrics.json` |
| **Irrigation** | Role A, B, C | `irrigation_best.pt`, `irrigation_scaler.joblib`, `irrigation_meta.json` | `irrigation_comparison.csv`, loss curve PNGs |
| **Disease** | Leaf Model Owner | `leaf_mobilenetv2.pte` (or `leaf_mobilenetv2.pt`), `leaf_classes.json` | `leaf_metrics.json` |
| **GAN/VAE** | Role F & G | `generator.pt`, `CLASSES.txt`, `samples/grid_<class>.png` | `results.csv`, `per_class_f1.png`, `notes.md` |

---

## 📱 On-Device ExecuTorch Proof (Module 3)
- **Model Size:** 3.5 MB (INT8 Quantized)
- **Latency:** ~18.4 ms per frame on ARM64 mobile hardware
- **Offline Proof:** Runs with **Airplane Mode enabled** (zero cloud API reliance).

