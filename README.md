# 🌾 AgriSense AI — Deep Learning Suite

AgriSense AI is an end-to-end precision agriculture deep learning suite uniting 5 academic curriculum modules into a single production-ready dashboard.

---

## ⚡ Quickstart (Under 5 Minutes)

### 1. Setup Environment
```bash
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

The application runs **100% locally and offline**. If any model file is pending, the dashboard gracefully runs in **Mock Mode** with status banners, allowing full evaluation without crashes.

---

## 📁 Repository Layout

| Path | Description |
| :--- | :--- |
| [`app/`](./app/) | Master Streamlit multipage application (`Home.py` + 5 module pages) |
| [`models/`](./models/) | Centralized drop zone for trained models & schema contracts |
| [`results/`](./results/) | Drop zone for metrics JSONs, comparison tables, and evaluation figures |
| [`sample_data/`](./sample_data/) | One-click demo assets for leaf images and irrigation windows |
| [`cnn/`](./cnn/) | Module 3: Leaf disease CNN & ExecuTorch edge deployment artifacts (`disease_cnn.pte`) |
| [`irrigation/`](./irrigation/) | Module 4: Arnesano soil moisture forecasting pipeline (LSTM, GRU, BiLSTM) |
| [`gallery_samples/`](./gallery_samples/) | Module 5: Synthetic leaf images generated via cVAE |
| [`data/`](./data/) | Cleaned datasets (Arnesano, TOPRAQ, Edge-IoT) |

---

## 📊 Irrigation Benchmarks at a Glance (Module 4)

Evaluated on the 7,411-sample Arnesano test split:

| Model | RMSE | MAE | R² | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | 6.9603 | 4.6683 | 0.9088 | 0.1577 |
| **GRU** | 5.6535 | 3.6262 | 0.9399 | 0.1455 |
| **BiLSTM** | **4.8395** | **3.3505** | **0.9559** | 0.4440 |
| *Persistence Baseline* | *0.7971* | *0.0881* | *0.9981* | *0.0005* |

---

## 🤝 Drop Contract for Teammates

> **Rule:** Nobody edits `app/`. Place your files directly into `models/` or `results/`.

| Module | Responsible Roles | Place in `models/` | Place in `results/` |
| :--- | :--- | :--- | :--- |
| **Yield** | Role D & E | `yield_mlp.pt`, `yield_scaler.joblib`, `yield_xgb.joblib` | `yield_metrics.json` |
| **Irrigation** | Role A, B, C | `irrigation_best.pt`, `irrigation_scaler.joblib`, `irrigation_meta.json` | `irrigation_comparison.csv` |
| **Disease** | Leaf Owner | `disease_cnn.pte` (or `.pt`), `leaf_classes.json` | `pte_confusion_matrix.png` |
| **GAN/VAE** | Role F & G | `generator.pt`, `CLASSES.txt`, `samples/` | `f1_comparison.csv`, `gallery_samples/` |
