import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision import models

from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str
    )

    parser.add_argument(
        "--class_names",
        required=True,
        type=str
    )

    parser.add_argument(
        "--out",
        default="disease_cnn.pte",
        type=str
    )

    parser.add_argument(
        "--img_size",
        default=224,
        type=int
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AgriSense AI - ExecuTorch Export")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load class names
    # ---------------------------------------------------------

    print("\n[1/4] Loading class names...")

    with open(args.class_names, "r") as f:
        classes = json.load(f)

    num_classes = len(classes)

    print(f"Found {num_classes} classes.")

    # ---------------------------------------------------------
    # 2. Build MobileNetV2
    # ---------------------------------------------------------

    print("\n[2/4] Building MobileNetV2...")

    model = build_model(num_classes)

    # ---------------------------------------------------------
    # 3. Load trained model
    # ---------------------------------------------------------

    print("\n[3/4] Loading trained checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Checkpoint loaded successfully.")

    # ---------------------------------------------------------
    # 4. Export to ExecuTorch
    # ---------------------------------------------------------

    print("\n[4/4] Exporting to ExecuTorch...")

    example_input = (
        torch.randn(
            1,
            3,
            args.img_size,
            args.img_size
        ),
    )

    print("Running torch.export...")

    exported_program = torch.export.export(
        model,
        example_input
    )

    print("torch.export successful.")

    print("Converting to Edge representation...")

    edge_program = to_edge(exported_program)

    print("Edge conversion successful.")

    print("Applying XNNPACK partitioner...")

    edge_program = edge_program.to_backend(
        XnnpackPartitioner()
    )

    print("XNNPACK partitioning successful.")

    print("Creating ExecuTorch program...")

    executorch_program = edge_program.to_executorch()

    # ---------------------------------------------------------
    # Save .pte file
    # ---------------------------------------------------------

    output_path = os.path.abspath(args.out)

    print(f"Saving model to:")
    print(output_path)

    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)

    print(f"\nCreated: {output_path}")
    print(f"Number of classes: {num_classes}")

    print("\nThe .pte model is ready.")
    print("Now run:")
    print("python test_pte.py")

    print("=" * 60)


if __name__ == "__main__":
    main()