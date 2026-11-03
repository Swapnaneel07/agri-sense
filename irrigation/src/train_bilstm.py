import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from bilstm_model import BiLSTM


# =====================================
# CONFIGURATION
# =====================================

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "windows")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bilstm")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================
# LOAD DATA
# =====================================

def load_dataset(name):

    path = os.path.join(DATA_DIR, f"arnesano_{name}.npz")
    data = np.load(path)

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    return TensorDataset(X, y)


train_dataset = load_dataset("train")
val_dataset = load_dataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================
# CREATE MODEL
# =====================================

model = BiLSTM(
    input_size=11,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    output_size=1
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nBiLSTM model created.")
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters()))


# =====================================
# TRAINING
# =====================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": []
}

start_time = time.time()

for epoch in range(MAX_EPOCHS):

    model.train()
    train_loss = 0.0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        predictions = model(X)
        loss = criterion(predictions, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            predictions = model(X)
            loss = criterion(predictions, y)

            val_loss += loss.item() * X.size(0)

    val_loss /= len(val_dataset)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(OUTPUT_DIR, "best.pt")
        )

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =====================================
# SAVE RESULTS
# =====================================

training_time = time.time() - start_time

with open(
    os.path.join(OUTPUT_DIR, "history.json"), "w"
) as f:

    json.dump(history, f, indent=4)

config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "training_time_seconds": training_time
}

with open(
    os.path.join(OUTPUT_DIR, "config.json"), "w"
) as f:

    json.dump(config, f, indent=4)

print("\n================================")
print("BiLSTM TRAINING COMPLETED!")
print("================================")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Training time:", round(training_time, 2), "seconds")
print("Model saved at:", os.path.join(OUTPUT_DIR, "best.pt"))