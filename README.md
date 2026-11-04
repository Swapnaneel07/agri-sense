# Agri-Sense

Agri-Sense is a time-series forecasting project for agricultural sensor data. It includes several irrigation datasets and a soil-moisture forecasting workflow that compares recurrent neural networks with a persistence baseline.

## Project layout

| Path | Contents |
| --- | --- |
| [`data/`](./data/) | Cleaned and uniformly named datasets (to be used for broad study)|
| [`data/README.md`](./data/README.md) | Dataset descriptions, fields, sources, and known data-quality notes |
| [`irrigation/`](./irrigation/) | Arnesano soil-moisture forecasting pipeline, models, and results |
| [`irrigation/README.md`](./irrigation/README.md) | Irrigation workflow instructions and detailed model results |

Available datasets include TOPRAQ, Arnesano, and Edge-IoT. The irrigation experiments documented here use the Arnesano dataset.

## Results at a glance

On the 7,411-sample Arnesano test split, BiLSTM had the lowest error of the three trained recurrent models (RMSE 4.8395, MAE 3.3505). The persistence baseline performed substantially better (RMSE 0.7971, MAE 0.0881), so these results should be treated as a signal to investigate the forecasting setup and baseline before relying on a learned model.


| Model | RMSE | MAE | R² | Inference time (s) |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 6.9603 | 4.6683 | 0.9088 | 0.1577 |
| GRU | 5.6535 | 3.6262 | 0.9399 | 0.1455 |
| BiLSTM | **4.8395** | **3.3505** | **0.9559** | 0.4440 |
| Persistence baseline | **0.7971** | **0.0881** | — | — |

BiLSTM is best among the trained recurrent models, and GRU outperforms LSTM. However, the persistence baseline's errors are much lower than every recurrent model's errors. Treat this as a significant evaluation warning: review target alignment, split behavior, units, and baseline calculations before using the learned models for decisions.

See the [irrigation README](./irrigation/README.md) for all metrics, generated artifacts, and commands to reproduce the experiments.
