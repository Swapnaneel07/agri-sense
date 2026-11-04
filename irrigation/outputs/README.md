# AgriSense AI – LSTM, GRU, BiLSTM Documentation


## 1. Dataset

* Dataset: Arnesano
* Target variable: `soil_moisture`
* Input features: 11
* Input window: 24 time steps
* Sampling interval: 10 minutes
* Forecast horizon: 1 time step (10 minutes)

The dataset is divided chronologically into training, validation, and testing sets.

## 3\2. BiLSTM Architecture

The implemented model uses:

* Bidirectional LSTM
* Hidden size: 64
* Number of layers: 2
* Dropout: 0.2
* Fully connected output layer
* Output size: 1

The forward and backward hidden states of the final LSTM layer are concatenated before prediction.

Trainable parameters: 138,881.

## 4. Training Configuration

| Parameter               |         Value |
| ----------------------- | ------------: |
| Random seed             |            42 |
| Batch size              |            64 |
| Learning rate           |         0.001 |
| Weight decay            |       0.00001 |
| Maximum epochs          |           100 |
| Early stopping patience |            10 |
| Best epoch              |             8 |
| Best validation loss    |      0.007614 |
| Training time           | 54.61 seconds |

Optimizer: AdamW
Loss function: Mean Squared Error (MSE)
Gradient clipping: 1.0

## 5. Evaluation Results

Test samples: 7,411

| Metric                |         BiLSTM |
| --------------------- | -------------: |
| RMSE                  |         4.8395 |
| MAE                   |         3.3505 |
| R²                    |         0.9559 |
| Inference time        | 0.4440 seconds |
| Inference time/sample |      0.0599 ms |

### Persistence Baseline

| Metric | Persistence |
| ------ | ----------: |
| RMSE   |      0.7971 |
| MAE    |      0.0881 |

The persistence baseline currently outperforms BiLSTM. This is an important limitation and must be considered during final model selection.

## 6. Visualizations

Generated plots:

1. `bilstm_training_loss.png`
2. `bilstm_actual_vs_predicted.png`
3. `model_comparison_rmse.png`
4. `model_comparison_mae.png`

Location: `outputs/plots/`

## 7. Prediction Pipeline

The `predict.py` module:

1. Loads the trained BiLSTM checkpoint.
2. Loads the saved feature and target scalers.
3. Accepts 24 time steps of input data.
4. Scales the input features.
5. Generates a 10-minute soil moisture forecast.
6. Converts the prediction back to the original scale.
7. Provides basic irrigation-related advice.

## 8. Model Comparison

The `compare.py` module reads available model evaluation files and generates:

* Model comparison CSV
* RMSE comparison plot
* MAE comparison plot

LSTM and GRU were trained using the same prepared data splits and evaluation pipeline as BiLSTM. Their recurrent layers use 64 hidden units, two layers, and 0.2 dropout; training uses AdamW, MSE loss, gradient clipping, and early stopping.

| Model | RMSE | MAE | R² | Inference time (s) |
| ----- | ---: | --: | -: | -----------------: |
| LSTM | 6.9603 | 4.6683 | 0.9088 | 0.1577 |
| GRU | 5.6535 | 3.6262 | 0.9399 | 0.1455 |
| BiLSTM | 4.8395 | 3.3505 | 0.9559 | 0.4440 |

BiLSTM has the best error metrics among the three recurrent models, and GRU outperforms LSTM. However, the persistence baseline (RMSE 0.7971, MAE 0.0881) remains substantially better than all three learned models.

## 9. Output Files

```text
outputs/
│
├── bilstm/
│   ├── best.pt
│   ├── config.json
│   ├── history.json
│   ├── metrics.json
│   ├── actual_test.npy
│   └── preds_test.npy
├── gru/
│   ├── best.pt
│   ├── config.json
│   ├── history.json
│   ├── metrics.json
│   ├── actual_test.npy
│   └── preds_test.npy
├── lstm/
│   ├── best.pt
│   ├── config.json
│   ├── history.json
│   ├── metrics.json
│   ├── actual_test.npy
│   └── preds_test.npy
│
├── plots/
│   ├── bilstm_training_loss.png
│   ├── bilstm_actual_vs_predicted.png
│   ├── model_comparison_rmse.png
│   └── model_comparison_mae.png
│
└── model_comparison.csv
```

Additional source files:

```text
src/
├── bilstm_model.py
├── train_bilstm.py
├── evaluate.py
├── plot_results.py
├── compare.py
├── prepare_data.py
├── train_recurrent
└── predict.py
```


## 10. Conclusion

The LSTM, GRU, BiLSTM training, evaluation, visualization, and prediction pipeline have been implemented successfully for the Arnesano soil moisture forecasting task.


