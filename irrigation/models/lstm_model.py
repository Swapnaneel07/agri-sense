"""
LSTM Model for Irrigation Scheduling Prediction
Agri-Sense Project
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_lstm_model(input_shape, num_classes=3):
    """
    Build a stacked LSTM model for irrigation scheduling classification.

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        num_classes: Number of output classes (default: 3)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=input_shape),

        # LSTM Layer 1
        LSTM(128, return_sequences=True),
        Dropout(0.3),

        # LSTM Layer 2
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_lstm_model(input_shape=(1, 10), num_classes=3)
    model.summary()
