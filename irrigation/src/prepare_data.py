import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "arnesano_uniform.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "windows")

WINDOW_SIZE = 24
HORIZONS = [1]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TARGET = "soil_moisture"

FEATURES = [
    "soil_moisture",
    "soil_temperature",
    "soil_temperature_deep",
    "air_temperature",
    "relative_humidity",
    "rainfall",
    "air_pressure",
    "wind_speed",
    "solar_radiation",
    "hour_sin",
    "hour_cos"
]


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

def load_data():

    print("\nLoading Arnesano dataset...")

    df = pd.read_csv(CSV_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove sector 3
    df = df[df["sector_id"] != 3].copy()

    # Sort chronologically
    df = df.sort_values(
        ["sector_id", "timestamp"]
    ).reset_index(drop=True)

    # Extract time-of-day features
    minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * minutes / 1440
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * minutes / 1440
    )

    # Convert features to numeric
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # Drop rows with missing required values
    df = df.dropna(
        subset=FEATURES + ["segment_id"]
    ).copy()

    print("Total usable rows:", len(df))
    print("Sectors:", sorted(df["sector_id"].unique()))
    print("Features:", len(FEATURES))

    return df


# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

def split_data(df):

    timestamps = np.sort(
        df["timestamp"].unique()
    )

    train_cutoff = timestamps[
        int(len(timestamps) * TRAIN_RATIO)
    ]

    val_cutoff = timestamps[
        int(len(timestamps) * (TRAIN_RATIO + VAL_RATIO))
    ]

    train_df = df[df["timestamp"] < train_cutoff].copy()

    val_df = df[
        (df["timestamp"] >= train_cutoff)
        & (df["timestamp"] < val_cutoff)
    ].copy()

    test_df = df[df["timestamp"] >= val_cutoff].copy()

    print("\nChronological split:")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Testing rows:", len(test_df))

    return train_df, val_df, test_df, train_cutoff, val_cutoff


# ==========================================
# 4. CREATE SLIDING WINDOWS
# ==========================================

def create_windows(df, split_name):

    X_list = []
    y_list = []
    time_list = []
    sector_list = []

    # Never cross sectors or segments
    groups = df.groupby(
        ["sector_id", "segment_id"],
        sort=False
    )

    for (sector, segment), group in groups:

        group = group.sort_values("timestamp")

        features = group[FEATURES].to_numpy(
            dtype=np.float32
        )

        target = group[TARGET].to_numpy(
            dtype=np.float32
        )

        timestamps = group["timestamp"].to_numpy()

        total = len(group)

        for i in range(
            WINDOW_SIZE,
            total - max(HORIZONS) + 1
        ):

            # Previous 24 readings
            x = features[i - WINDOW_SIZE:i]

            # Future soil moisture
            y = [
                target[i + horizon - 1]
                for horizon in HORIZONS
            ]

            X_list.append(x)
            y_list.append(y)

            time_list.append(
                timestamps[i - 1]
            )

            sector_list.append(sector)

    if len(X_list) == 0:
        raise ValueError(
            f"No windows created for {split_name}. "
            "Check segment lengths and split dates."
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    print(f"\n{split_name.upper()} windows:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return (
        X,
        y,
        np.asarray(time_list),
        np.asarray(sector_list)
    )


# ==========================================
# 5. MAIN PROCESS
# ==========================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    train_df, val_df, test_df, train_cutoff, val_cutoff = (
        split_data(df)
    )

    # Create windows separately for each split
    X_train, y_train, t_train, s_train = create_windows(
        train_df, "train"
    )

    X_val, y_val, t_val, s_val = create_windows(
        val_df, "validation"
    )

    X_test, y_test, t_test, s_test = create_windows(
        test_df, "test"
    )

    # ======================================
    # 6. SCALE DATA (TRAINING ONLY)
    # ======================================

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit using training rows only
    x_scaler.fit(
        train_df[FEATURES].to_numpy()
    )

    y_scaler.fit(
        train_df[[TARGET]].to_numpy()
    )

    def scale_X(X):
        shape = X.shape

        return x_scaler.transform(
            X.reshape(-1, shape[-1])
        ).reshape(shape).astype(np.float32)

    def scale_y(y):
        return y_scaler.transform(
            y.reshape(-1, 1)
        ).reshape(y.shape).astype(np.float32)

    X_train = scale_X(X_train)
    X_val = scale_X(X_val)
    X_test = scale_X(X_test)

    y_train = scale_y(y_train)
    y_val = scale_y(y_val)
    y_test = scale_y(y_test)

    # ======================================
    # 7. SAVE DATASETS
    # ======================================

    datasets = {
        "train": (X_train, y_train, t_train, s_train),
        "val": (X_val, y_val, t_val, s_val),
        "test": (X_test, y_test, t_test, s_test)
    }

    for name, (X, y, t, sector) in datasets.items():

        np.savez_compressed(
            os.path.join(
                OUTPUT_DIR,
                f"arnesano_{name}.npz"
            ),
            X=X,
            y=y,
            t=t,
            sector=sector
        )

    # Save scalers
    joblib.dump(
        {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler
        },
        os.path.join(BASE_DIR, "data", "scaler.joblib")
    )

    # Save metadata
    metadata = {
        "dataset": "Arnesano",
        "features": FEATURES,
        "target": TARGET,
        "window_size": WINDOW_SIZE,
        "horizons": HORIZONS,
        "input_shape": list(X_train.shape[1:]),
        "output_shape": list(y_train.shape[1:]),
        "train_cutoff": str(train_cutoff),
        "validation_cutoff": str(val_cutoff),
        "train_windows": len(X_train),
        "validation_windows": len(X_val),
        "test_windows": len(X_test),
        "scaling": "StandardScaler fitted on training rows only"
    }

    with open(
        os.path.join(OUTPUT_DIR, "arnesano_meta.json"),
        "w"
    ) as f:

        json.dump(metadata, f, indent=4)

    print("\n================================")
    print("DATA PREPARATION COMPLETED!")
    print("================================")

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    print("\nSaved files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()