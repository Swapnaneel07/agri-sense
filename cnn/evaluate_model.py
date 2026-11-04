import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from executorch.runtime import Runtime, Verification


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(r"D:\deep learning\plantvillage dataset\color")

CHECKPOINT_PATH = Path(r"D:\deep learning\checkpoints\best_model.pt")

PTE_PATH = Path(r"D:\deep learning\disease_cnn.pte")

CLASS_NAMES_PATH = Path(
    r"D:\deep learning\checkpoints\class_names.json"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Set to None to evaluate the entire dataset.
# For a quick test, use something like 100.
MAX_IMAGES = None


# ============================================================
# BUILD MOBILE NET V2
# ============================================================

def build_model(num_classes):

    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# DATASET
# ============================================================

def create_dataset():

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    return dataset


# ============================================================
# PYTORCH MODEL EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_pytorch(model, loader, classes):

    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    inference_times = []

    for images, labels in loader:

        start = time.perf_counter()

        outputs = model(images)

        end = time.perf_counter()

        inference_times.append(
            (end - start) / images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            labels.tolist()
        )

        if MAX_IMAGES is not None:
            if total >= MAX_IMAGES:
                break

    accuracy = correct / total

    avg_time = sum(inference_times) / len(
        inference_times
    )

    return (
        accuracy,
        avg_time,
        all_labels,
        all_predictions
    )


# ============================================================
# EXECUTORCH MODEL EVALUATION
# ============================================================

def evaluate_pte(dataset, classes):

    print("\nLoading ExecuTorch model...")

    runtime = Runtime.get()

    program = runtime.load_program(
        PTE_PATH,
        verification=Verification.Minimal
    )

    forward_method = program.load_method("forward")

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    inference_times = []

    print("Running ExecuTorch evaluation...")

    for index in range(len(dataset)):

        if MAX_IMAGES is not None:
            if total >= MAX_IMAGES:
                break

        image, label = dataset[index]

        # Add batch dimension
        image = image.unsqueeze(0)

        start = time.perf_counter()

        outputs = forward_method.execute(
            (image,)
        )

        end = time.perf_counter()

        inference_time = end - start

        inference_times.append(
            inference_time
        )

        logits = outputs[0]

        prediction = logits.argmax(
            dim=1
        ).item()

        all_predictions.append(
            prediction
        )

        all_labels.append(
            label
        )

        if prediction == label:
            correct += 1

        total += 1

        if total % 500 == 0:
            print(
                f"Processed {total} images..."
            )

    accuracy = correct / total

    avg_time = sum(
        inference_times
    ) / len(inference_times)

    return (
        accuracy,
        avg_time,
        all_labels,
        all_predictions
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    classes
):

    num_classes = len(classes)

    # Confusion matrix
    confusion = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for actual, predicted in zip(
        labels,
        predictions
    ):

        confusion[actual][predicted] += 1

    # Per-class metrics
    print("\n")
    print("=" * 100)
    print("PER-CLASS PERFORMANCE")
    print("=" * 100)

    print(
        f"{'Class':55}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Samples':>12}"
    )

    print("-" * 100)

    precisions = []
    recalls = []
    f1_scores = []

    for i, class_name in enumerate(classes):

        tp = confusion[i][i]

        fp = sum(
            confusion[j][i]
            for j in range(num_classes)
            if j != i
        )

        fn = sum(
            confusion[i][j]
            for j in range(num_classes)
            if j != i
        )

        samples = sum(confusion[i])

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(
            f"{class_name[:55]:55}"
            f"{precision * 100:11.2f}%"
            f"{recall * 100:11.2f}%"
            f"{f1 * 100:11.2f}%"
            f"{samples:12d}"
        )

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes

    print("-" * 100)

    print(
        f"{'MACRO AVERAGE':55}"
        f"{macro_precision * 100:11.2f}%"
        f"{macro_recall * 100:11.2f}%"
        f"{macro_f1 * 100:11.2f}%"
    )

    return confusion


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    confusion,
    classes,
    filename
):

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(18, 16)
    )

    plt.imshow(confusion)

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(classes)),
        classes,
        fontsize=7
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to: {filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AgriSense AI - MODEL PERFORMANCE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not PTE_PATH.exists():
        raise FileNotFoundError(
            f"PTE model not found:\n{PTE_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json not found:\n"
            f"{CLASS_NAMES_PATH}"
        )

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as f:

        classes = json.load(f)

    print(
        f"\nNumber of classes: {len(classes)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = create_dataset()

    print(
        f"Number of images: {len(dataset)}"
    )

    print(
        f"Number of classes detected: "
        f"{len(dataset.classes)}"
    )

    # Check class ordering
    if dataset.classes != classes:

        print("\nWARNING:")
        print(
            "Dataset class ordering does not "
            "match class_names.json."
        )

        print("\nDataset classes:")
        print(dataset.classes)

        print("\nJSON classes:")
        print(classes)

        raise ValueError(
            "Class ordering mismatch!"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # PYTORCH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PYTORCH MODEL")
    print("=" * 70)

    model = build_model(
        len(classes)
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    (
        pytorch_accuracy,
        pytorch_time,
        pytorch_labels,
        pytorch_predictions
    ) = evaluate_pytorch(
        model,
        loader,
        classes
    )

    print(
        f"\nPyTorch Accuracy: "
        f"{pytorch_accuracy * 100:.2f}%"
    )

    print(
        f"Average PyTorch inference time: "
        f"{pytorch_time * 1000:.2f} ms/image"
    )

    # ========================================================
    # PYTORCH METRICS
    # ========================================================

    pytorch_confusion = calculate_metrics(
        pytorch_labels,
        pytorch_predictions,
        classes
    )

    save_confusion_matrix(
        pytorch_confusion,
        classes,
        "pytorch_confusion_matrix.png"
    )

    # ========================================================
    # EXECUTORCH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXECUTORCH .PTE MODEL")
    print("=" * 70)

    (
        pte_accuracy,
        pte_time,
        pte_labels,
        pte_predictions
    ) = evaluate_pte(
        dataset,
        classes
    )

    print(
        f"\nExecuTorch Accuracy: "
        f"{pte_accuracy * 100:.2f}%"
    )

    print(
        f"Average ExecuTorch inference time: "
        f"{pte_time * 1000:.2f} ms/image"
    )

    # ========================================================
    # EXECUTORCH METRICS
    # ========================================================

    pte_confusion = calculate_metrics(
        pte_labels,
        pte_predictions,
        classes
    )

    save_confusion_matrix(
        pte_confusion,
        classes,
        "pte_confusion_matrix.png"
    )

    # ========================================================
    # MODEL SIZE
    # ========================================================

    pte_size_mb = (
        PTE_PATH.stat().st_size
        / (1024 * 1024)
    )

    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size
        / (1024 * 1024)
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Metric':35}"
        f"{'PyTorch':>15}"
        f"{'ExecuTorch':>15}"
    )

    print("-" * 70)

    print(
        f"{'Accuracy':35}"
        f"{pytorch_accuracy * 100:14.2f}%"
        f"{pte_accuracy * 100:14.2f}%"
    )

    print(
        f"{'Inference time/image':35}"
        f"{pytorch_time * 1000:14.2f} ms"
        f"{pte_time * 1000:14.2f} ms"
    )

    print(
        f"\nPyTorch model size: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    print(
        f"ExecuTorch .pte size: "
        f"{pte_size_mb:.2f} MB"
    )

    # Check prediction agreement

    agreement = sum(
        a == b
        for a, b in zip(
            pytorch_predictions,
            pte_predictions
        )
    )

    total_comparison = len(
        pytorch_predictions
    )

    agreement_percentage = (
        agreement / total_comparison
    ) * 100

    print(
        f"\nPyTorch vs ExecuTorch "
        f"prediction agreement: "
        f"{agreement_percentage:.2f}%"
    )

    print("\nEvaluation completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()