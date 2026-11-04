"""
AgriSense AI — Module 2: Train the CNN (MobileNetV2 fine-tune)

Trains a MobileNetV2 classifier on a leaf-disease image dataset laid out as:
    data_dir/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

Saves:
    checkpoints/best_model.pt      (state_dict of the best-val-accuracy epoch)
    checkpoints/class_names.json   (ordered list of class names, index-aligned)

Run:
    python train_cnn.py --data_dir ./data --epochs 15 --batch_size 32 \
        --out_dir ./checkpoints
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_dataloaders(data_dir, img_size, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    for d in (train_dir, val_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Expected dataset split at {d}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    # ImageFolder assigns class indices alphabetically by folder name, in both
    # splits independently -- this assertion guards against a mismatch that
    # would silently corrupt labels if train/val folders differ.
    assert train_ds.classes == val_ds.classes, "train/val class folders don't match"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze feature extractor, train classifier head only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    model = build_model(num_classes, pretrained=True).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen -- training classifier head only")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (val_acc {best_val_acc:.4f}), saved to {best_path}")

    class_names_path = os.path.join(args.out_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(classes, f, indent=2)

    print(f"\nDone. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {best_path}")
    print(f"Class names: {class_names_path}")
    print("\nNext: run export_to_executorch.py on this checkpoint to produce a .pte for on-device inference.")


if __name__ == "__main__":
    main()