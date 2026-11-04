# 🌾 AgriSense AI — Teammate Drop-in Integration Guide

This guide explains where each member should place their code and model weights so they automatically connect to the master dashboard (`app.py`).

---

### Directory Map

```text
agri-sense/
├── app.py                         # Master Streamlit dashboard (Role H)
├── requirements.txt               # Unified project dependencies
├── saved_models/                  # Central model weight depository
│   ├── irrigation/
│   │   ├── bilstm_best.pt         # ✅ Uploaded (Role C)
│   │   ├── lstm_best.pt           # ⏳ Role B
│   │   ├── gru_best.pt            # ⏳ Role B
│   │   └── scaler.joblib          # ✅ Uploaded (Role A)
│   ├── yield_pred/
│   │   ├── mlp_best.pt            # ⏳ Role D
│   │   ├── scaler.pkl             # ⏳ Role D
│   │   └── xgboost.pkl            # ⏳ Role E
│   ├── leaf_disease/
│   │   ├── mobilenetv2.pt         # ⏳ Module 3 team
│   │   └── unet.pt                # ⏳ Module 3 team (Role I stretch)
│   └── gan_vae/
│       └── dcgan_generator.pt     # ⏳ Role F
└── modules/
    ├── irrigation/                # Module 4
    ├── yield_pred/                # Modules 1 & 2
    ├── leaf_disease/              # Module 3
    └── gan_vae/                   # Module 5
```

---

### Checklist by Role

- [ ] **Role B (Irrigation LSTM & GRU)**:
  - Save LSTM weights to `saved_models/irrigation/lstm_best.pt`
  - Save GRU weights to `saved_models/irrigation/gru_best.pt`
  - Ensure metrics match the schema in `saved_models/irrigation/bilstm_metrics.json`.

- [ ] **Role D (Crop Yield MLP)**:
  - Save PyTorch model state_dict to `saved_models/yield_pred/mlp_best.pt`
  - Save feature scaler to `saved_models/yield_pred/scaler.pkl`

- [ ] **Role E (Baselines & Ablations)**:
  - Drop trained XGBoost model to `saved_models/yield_pred/xgboost.pkl`
  - Drop ablation results into `outputs/yield/`

- [ ] **Module 3 Team (Leaf Disease & MobileNetV2)**:
  - Save MobileNetV2 weights to `saved_models/leaf_disease/mobilenetv2.pt`

- [ ] **Role F & G (GAN/VAE)**:
  - Save DCGAN generator weights to `saved_models/gan_vae/dcgan_generator.pt`
  - Provide sample synthetic images to `outputs/gan_vae/`

