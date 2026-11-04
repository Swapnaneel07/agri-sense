# Agri-Sense Irrigation Forecasting

This subproject prepares Arnesano sensor data and trains recurrent neural networks to forecast soil moisture one 10-minute step ahead. It includes LSTM, GRU, and BiLSTM models, evaluation against a persistence baseline, and prediction/comparison plots.

## Results

All models were evaluated on the same chronological test split (7,411 samples). Lower RMSE and MAE are better; higher R² is better.

| Model | RMSE | MAE | R² | Inference time (s) |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 6.9603 | 4.6683 | 0.9088 | 0.1577 |
| GRU | 5.6535 | 3.6262 | 0.9399 | 0.1455 |
| BiLSTM | **4.8395** | **3.3505** | **0.9559** | 0.4440 |
| Persistence baseline | **0.7971** | **0.0881** | — | — |

BiLSTM is best among the trained recurrent models, and GRU outperforms LSTM. However, the persistence baseline's errors are much lower than every recurrent model's errors. Treat this as a significant evaluation warning: review target alignment, split behavior, units, and baseline calculations before using the learned models for decisions.

Plots and the comparison table are in [`outputs/`](./outputs/):

- [`model_comparison.csv`](./outputs/model_comparison.csv)
- [`plots/model_comparison_rmse.png`](./outputs/plots/model_comparison_rmse.png)
- [`plots/model_comparison_mae.png`](./outputs/plots/model_comparison_mae.png)
- Per-model training-loss and actual-vs-predicted plots in `outputs/plots/`

## Dataset and preprocessing

- Source file: `data/arnesano_uniform.csv`
- Target: `soil_moisture`
- Inputs: 11 features, including soil and weather measurements plus cyclical time-of-day features
- Window: 24 observations (four hours at 10-minute sampling intervals)
- Horizon: one observation (10 minutes)
- Split: chronological 70% train, 15% validation, 15% test
- The current preprocessing excludes sector 3 and does not create windows across sector/segment boundaries.
- Scalers are fitted on training data only.

The repository-level [`data/data_README.md`](../data/data_README.md) contains dataset provenance and broader field documentation.

## Setup and run

Use Python with PyTorch, NumPy, pandas, scikit-learn, joblib, and Matplotlib installed. From the repository root:

```powershell
python irrigation\src\prepare_data.py
python irrigation\src\train_bilstm.py
python irrigation\src\train_recurrent.py --model lstm
python irrigation\src\train_recurrent.py --model gru
python irrigation\src\compare.py
```

`prepare_data.py` regenerates the prepared windows and scaler from the CSV. The training scripts expect these prepared files. To run the BiLSTM prediction example after training:

```powershell
python irrigation\src\predict.py
```

## Outputs

Each model directory under `outputs/` contains:

- `best.pt`: best validation-loss checkpoint
- `config.json` and `history.json`: training configuration and epoch history
- `metrics.json`: test metrics, inference timing, and persistence-baseline metrics
- `actual_test.npy` and `preds_test.npy`: test targets and predictions in original units

`outputs/plots/` contains training curves, actual-vs-predicted plots, and model-comparison plots. `compare.py` rebuilds the comparison CSV and comparison plots from the available model metrics.

## Member B

Member B's contribution documented here is the LSTM and GRU training and evaluation workflow, along with the comparison of recurrent models using consistent prepared data splits and metrics. The existing BiLSTM implementation and results are included as the comparison reference.

## Member C

Member C's contribution documented here is the BiLSTM training and  data splits and metrics. The BiLSTM implementation and results are included in the comparison for the best models.

## Legacy files

`models/` and `scripts/train_model.py` contain a separate Keras classification workflow that refers to preprocessing and dataset files not present in this repository. It is not used by the PyTorch soil-moisture forecasting commands above and is retained as legacy code.
