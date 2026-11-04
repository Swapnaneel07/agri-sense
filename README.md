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

See the [irrigation README](./irrigation/README.md) for all metrics, generated artifacts, and commands to reproduce the experiments.
