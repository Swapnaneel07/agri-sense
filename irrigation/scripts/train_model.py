"""
Unified Training & Evaluation Script for LSTM/GRU Models
Agri-Sense Irrigation Scheduling Project

Usage:
    python scripts/train_model.py --model lstm
    python scripts/train_model.py --model gru
"""

import sys, os, argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def _import_module(name, path):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preprocess = _import_module("preprocess", os.path.join(SCRIPT_DIR, "preprocess.py"))
load_and_preprocess = preprocess.load_and_preprocess


def train_and_evaluate(model_type):
    """Train and evaluate a model, generating all output artifacts."""

    print(f"\n{'='*60}")
    print(f"  Training {model_type.upper()} Model for Irrigation Scheduling")
    print(f"{'='*60}\n")

    # --- Load Data ---
    data_path = os.path.join(PROJECT_ROOT, 'data', 'merged_irrigation_dataset.csv')
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = load_and_preprocess(
        data_path=data_path, save_splits=False, output_dir=os.path.join(PROJECT_ROOT, 'data')
    )
    print(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {list(le.classes_)}")

    # --- Build Model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    if model_type == 'lstm':
        mod = _import_module("lstm_model", os.path.join(models_dir, "lstm_model.py"))
        model = mod.build_lstm_model(input_shape, num_classes)
    else:
        mod = _import_module("gru_model", os.path.join(models_dir, "gru_model.py"))
        model = mod.build_gru_model(input_shape, num_classes)

    model.summary()

    # --- Train ---
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # --- Predictions ---
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    class_names = list(le.classes_)

    # --- Classification Report ---
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print(f"\nClassification Report:\n{report}")

    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    prefix = model_type

    report_path = os.path.join(outputs_dir, f'{prefix}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{model_type.upper()} Model - Classification Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
        f.write(report)
    print(f"Saved: {report_path}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type.upper()} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, f'{prefix}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    # --- Loss & Accuracy Curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_type.upper()} - Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{model_type.upper()} - Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    curves_path = os.path.join(outputs_dir, f'{prefix}_loss_accuracy_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved: {curves_path}")

    # --- Predictions vs Actual ---
    _, pred_counts = np.unique(y_pred_classes, return_counts=True)
    _, true_counts = np.unique(y_true_classes, return_counts=True)

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, true_counts, width, label='Actual', color='steelblue')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='coral')
    plt.xlabel('Irrigation Class')
    plt.ylabel('Count')
    plt.title(f'{model_type.upper()} - Predictions vs Actual')
    plt.xticks(x, class_names, rotation=15)
    plt.legend()
    plt.tight_layout()
    pva_path = os.path.join(outputs_dir, f'{prefix}_predictions_vs_actual.png')
    plt.savefig(pva_path, dpi=150)
    plt.close()
    print(f"Saved: {pva_path}")

    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} Training Complete!")
    print(f"  Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    print(f"{'='*60}\n")

    return test_loss, test_acc, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM/GRU for Irrigation Scheduling')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model type: lstm or gru')
    args = parser.parse_args()
    train_and_evaluate(args.model)
