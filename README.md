# 🌾 AgriSense AI — Deep Learning Academic Suite

AgriSense AI is an end-to-end precision agriculture deep learning suite designed for a 5-module Deep Learning academic curriculum. It integrates all course modules into a unified multi-page Streamlit dashboard (`app/Home.py`), featuring interactive real-time inference, academic ablation studies, loss curves, confusion matrices, and benchmark comparison tables for viva evaluation.

---

## ⚡ Quickstart Guide (Run in Under 3 Minutes)

### Prerequisites
- Python 3.10 or 3.11 recommended.

### 1. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
# Navigate to project root
cd agri-sense

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

#### On macOS / Linux (Bash):
```bash
cd agri-sense
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Launch the Streamlit Dashboard

Run with the environment activated:
```bash
streamlit run app/Home.py
```

**Alternative (Direct execution without activating PowerShell scripts):**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Once launched, open your web browser at: **`http://localhost:8501`**

> [!NOTE]
> The app runs **100% offline and locally**. When a teammate's model checkpoint is not yet available, the dashboard gracefully runs in **Mock Mode** with clear UI status banners, allowing full testing of forms and visualizations without crashing.

---

## 📚 5-Module Syllabus Architecture

| Module | Core Academic Topics | Implementation in AgriSense | Dashboard Page |
| :--- | :--- | :--- | :--- |
| **Module 1** | ANN, MLP Regressors, Activations, Optimizers | Multi-Layer Perceptron on soil chemistry & climate parameters (N, P, K, Temp, Humidity, pH, Rainfall). Activation comparison (ReLU vs Tanh vs LeakyReLU) and optimizer convergence (Adam vs SGD vs RMSprop). | `1_Crop_Yield` |
| **Module 2** | Backpropagation, Regularization, Initialization, Baselines | Dropout ($p=0.2$) and L2 Weight Decay ($10^{-4}$) ablation to prevent overfitting; He (Kaiming) vs Xavier initialization; benchmarked against XGBoost and Random Forest baselines. | `1_Crop_Yield` |
| **Module 3** | CNN, Transfer Learning, Mobile Edge Deployment, UNet | MobileNetV2 fine-tuned on PlantVillage leaf photos. Compiled to **ExecuTorch `.pte` format** for lightweight offline edge inference. Includes UNet disease lesion heatmap overlay. | `2_Leaf_Disease` |
| **Module 4** | RNN, LSTM, GRU, BiLSTM | Sequential soil moisture forecasting on Arnesano IoT sliding windows (24 timesteps × 10 min). Side-by-side comparison across LSTM, GRU, BiLSTM, and Persistence Baseline. | `3_Irrigation` |
| **Module 5** | Generative AI, GAN, VAE | Conditional VAE (cVAE) trained on underrepresented leaf pathology classes to resolve extreme class imbalance. Produces +13% to +15% F1 score improvement on rare classes. | `4_Synthetic_Leaves` |
| **Results** | Master Benchmarks & Viva Evaluation | Centralized metrics browser reading directly from `results/` for examiner viva defense. | `5_Results` |

---

## 📊 Live Benchmark Highlights

### Module 4: Irrigation Forecasting (Arnesano IoT Test Split)
Evaluated on 7,411 test samples:
| Model Architecture | RMSE | MAE | R² | Inference Time | Directionality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 6.9603 | 4.6683 | 0.9088 | 0.158 s | Unidirectional |
| **GRU** | 5.6535 | 3.6262 | 0.9399 | 0.145 s | Unidirectional |
| **BiLSTM** | **4.8395** | **3.3505** | **0.9559** | 0.444 s | Bidirectional (Winner) |
| *Persistence Baseline* | *0.7971* | *0.0881* | *0.9981* | *0.0005 s* | Reference |

*Viva Insight:* The persistence baseline outperforms neural models due to the high auto-correlation of soil moisture across 10-minute intervals.

### Module 5: Generative cVAE Class Imbalance Gain
| Minority Class | Baseline F1 | Augmented F1 (Post-cVAE) | Relative Gain | Real Samples | Synthetic Samples Added |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Potato — healthy** | 0.742 | **0.895** | **+15.3%** | 30 | +400 |
| **Apple — Cedar apple rust** | 0.785 | **0.921** | **+13.6%** | 30 | +400 |
| **Peach — healthy** | 0.761 | **0.908** | **+14.7%** | 30 | +400 |
| **Raspberry — healthy** | 0.803 | **0.934** | **+13.1%** | 30 | +400 |
| **Tomato — Tomato mosaic virus** | 0.738 | **0.887** | **+14.9%** | 30 | +400 |

---

## 📁 Repository Directory Structure

```text
agri-sense/
├── app/
│   ├── Home.py                  # Landing page & live status badges
│   ├── pages/
│   │   ├── 1_Crop_Yield.py       # Module 1 & 2: Dynamic NPK form + MLP/XGBoost
│   │   ├── 2_Leaf_Disease.py     # Module 3: MobileNetV2 + ExecuTorch edge inference
│   │   ├── 3_Irrigation.py       # Module 4: Sequential time-series forecasting
│   │   ├── 4_Synthetic_Leaves.py # Module 5: cVAE augmentation gallery & F1 metrics
│   │   └── 5_Results.py          # Academic benchmark comparisons (reads from results/)
│   └── core/
│       ├── config.py             # Central path constants
│       ├── loaders.py            # Cached model & schema loaders (st.cache_resource)
│       ├── predict.py            # Clean unified prediction dispatchers
│       └── mock.py               # Fallback simulation engine
├── models/                       # DROP ZONE for teammates (nobody edits app/)
│   ├── yield_schema.json         # Feature specification driving the Yield form
│   ├── leaf_classes.json         # 38 PlantVillage target classes
│   ├── leaf_mobilenetv2.pte      # ExecuTorch edge model binary (9.0 MB)
│   ├── CLASSES.txt               # 5 target rare classes for cVAE
│   ├── generator.pt              # Trained cVAE generator weights (35.7 MB)
│   ├── samples/                  # Real synthetic generated leaf photos
│   ├── irrigation_best.pt        # BiLSTM model weights
│   ├── lstm_best.pt              # LSTM model weights
│   ├── gru_best.pt               # GRU model weights
│   ├── irrigation_scaler.joblib  # Standardized feature scalers
│   └── irrigation_meta.json      # Time-series sliding window metadata
├── results/                      # DROP ZONE for metrics, tables, and figures
│   ├── irrigation_comparison.csv # 3-way sequential model comparison table
│   ├── results.csv               # Role G before vs after F1 score improvements
│   ├── pte_confusion_matrix.png  # ExecuTorch quantized confusion matrix
│   ├── pytorch_confusion_matrix.png # PyTorch baseline confusion matrix
│   └── [plots]                   # Loss curves & actual vs predicted trajectories
├── sample_data/                  # Instant demo data (zero configuration required)
│   ├── leaves/                   # 5 foliage photos (healthy & diseased)
│   └── irrigation/               # 24-step IoT time-series sample window CSV
├── cnn/                          # Module 3 raw training, evaluation & export scripts
├── irrigation/                   # Module 4 recurrent training pipelines & source data
├── modules/                      # Domain Python packages
└── requirements.txt              # Unified dependencies
```

---

## 🤝 Team Drop-in Contract

> **Rule:** Nobody edits `app/`. Place your files directly into `models/` or `results/`.

| Module | Responsible Roles | Place in `models/` | Place in `results/` |
| :--- | :--- | :--- | :--- |
| **Yield** | Role D & E | `yield_mlp.pt` (or `.keras`), `yield_scaler.joblib`, `yield_xgb.joblib` | `yield_metrics.json` |
| **Irrigation** | Role A, B, C | `irrigation_best.pt`, `irrigation_scaler.joblib`, `irrigation_meta.json` | `irrigation_comparison.csv`, loss curve PNGs |
| **Disease** | Leaf Owner | `disease_cnn.pte` (or `.pt`), `leaf_classes.json` | `pte_confusion_matrix.png` |
| **GAN/VAE** | Role F & G | `generator.pt`, `CLASSES.txt`, `samples/` | `results.csv`, `gallery_samples/` |
