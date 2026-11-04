"""
AgriSense AI — Module 5: Role G (Augmentation & Evaluation)
File: modules/gan_vae/evaluation.py

This module evaluates whether synthetic leaf images produced by Role F improve
MobileNetV2 classification on rare disease classes under Regime B (Scarce Data).
"""

import os
import sys
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

TARGET_CLASSES = [
    "Potato___healthy",
    "Apple___Cedar_apple_rust",
    "Peach___healthy",
    "Raspberry___healthy",
    "Tomato___Tomato_mosaic_virus"
]
NUM_CLASSES = len(TARGET_CLASSES)
class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
idx_to_class = {idx: cls for idx, cls in enumerate(TARGET_CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------- DATASET & TRANSFORMS -----------------
class MobileNetLeafDataset(Dataset):
    def __init__(self, records_df, transform=None, apply_blur_defense=False):
        self.records = records_df.reset_index(drop=True)
        self.transform = transform
        self.apply_blur_defense = apply_blur_defense

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_path = row['path']
        cls_name = row['class']
        is_synth = row.get('is_synth', 0)
        label = class_to_idx[cls_name]
        
        image = Image.open(img_path).convert('RGB')
        
        # Blur shortcut countermeasure: randomly soften real leaves in synthetic runs
        if self.apply_blur_defense and (not is_synth) and random.random() < 0.5:
            image = image.resize((64, 64), Image.BILINEAR).resize((224, 224), Image.BICUBIC)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])

classic_aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    NORMALIZE
])


# ----------------- MOBILENETV2 HELPERS -----------------
def get_mobilenet_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model.to(device)


def train_and_eval(model, train_loader, val_loader, test_loader, criterion, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_f1 = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds = model(images).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict()
            
    if best_weights:
        model.load_state_dict(best_weights)
        
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    per_class = f1_score(test_targets, test_preds, average=None, zero_division=0)
    return acc, macro_f1, per_class, test_preds, test_targets


# ----------------- MAIN EVALUATION API -----------------
def run_classifier_evaluation(raw_data_dir="data/raw",
                              splits_file="data/raw/splits.csv",
                              synthetic_dir="data/processed/synthetic",
                              results_dir="data/processed/results",
                              scarce_count=30, epochs=10):
    """
    Executes benchmark experiments:
    E0: Real Only (Scarce Baseline)
    E1: Real + Classic Augmentation
    E2: Real + Class-Weighted Loss
    E3: Real + Synthetic (cVAE)
    """
    print(f"[*] Running Classifier Evaluation on device: {device}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Dataset splits
    if os.path.exists(splits_file):
        splits_df = pd.read_csv(splits_file)
    elif os.path.exists(raw_data_dir):
        real_samples = []
        for cls_name in TARGET_CLASSES:
            c_dir = Path(raw_data_dir) / cls_name
            if c_dir.exists():
                for img_p in c_dir.iterdir():
                    if img_p.is_file() and img_p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        real_samples.append({'path': str(img_p), 'class': cls_name})
        df = pd.DataFrame(real_samples)
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['class'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['class'], random_state=42)
        train_df = train_df.copy(); train_df['split'] = 'train'
        val_df = val_df.copy(); val_df['split'] = 'val'
        test_df = test_df.copy(); test_df['split'] = 'test'
        splits_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        os.makedirs(os.path.dirname(splits_file) or ".", exist_ok=True)
        splits_df.to_csv(splits_file, index=False)
        print(f"[+] Saved '{splits_file}' with {len(splits_df)} real samples.")
    else:
        demo_real = Path(raw_data_dir) / "demo_plantvillage"
        demo_real.mkdir(parents=True, exist_ok=True)
        colors = [(34, 139, 34), (139, 69, 19), (60, 179, 113), (46, 139, 87), (205, 92, 92)]
        real_samples = []
        for cls_name, col in zip(TARGET_CLASSES, colors):
            c_dir = demo_real / cls_name
            c_dir.mkdir(parents=True, exist_ok=True)
            for i in range(120):
                p = c_dir / f"real_{i:04d}.png"
                if not p.exists():
                    img = Image.new('RGB', (224, 224), color=(col[0] + random.randint(-15, 15), col[1] + random.randint(-15, 15), col[2] + random.randint(-15, 15)))
                    img.save(p)
                real_samples.append({'path': str(p), 'class': cls_name})
        df = pd.DataFrame(real_samples)
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['class'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['class'], random_state=42)
        train_df = train_df.copy(); train_df['split'] = 'train'
        val_df = val_df.copy(); val_df['split'] = 'val'
        test_df = test_df.copy(); test_df['split'] = 'test'
        splits_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        os.makedirs(os.path.dirname(splits_file) or ".", exist_ok=True)
        splits_df.to_csv(splits_file, index=False)
        print(f"[+] Saved '{splits_file}'")

    # 2. Regime B (Scarce)
    train_records = []
    for cls_name in TARGET_CLASSES:
        cls_train = splits_df[(splits_df['split'] == 'train') & (splits_df['class'] == cls_name)]
        subsampled = cls_train.sample(n=min(scarce_count, len(cls_train)), random_state=42)
        for _, row in subsampled.iterrows():
            train_records.append({'path': row['path'], 'class': row['class'], 'is_synth': 0})
    real_train_df = pd.DataFrame(train_records)
    val_df = splits_df[splits_df['split'] == 'val'].copy()
    test_df = splits_df[splits_df['split'] == 'test'].copy()

    val_loader = DataLoader(MobileNetLeafDataset(val_df, transform=base_transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(MobileNetLeafDataset(test_df, transform=base_transform), batch_size=32, shuffle=False)

    # 3. Collect synthetic samples
    synth_records = []
    for cls_name in TARGET_CLASSES:
        c_dir = Path(synthetic_dir) / cls_name
        if c_dir.exists():
            for p in list(c_dir.glob("*.png"))[:150]:
                synth_records.append({'path': str(p), 'class': cls_name, 'is_synth': 1})
    synth_df = pd.DataFrame(synth_records)
    print(f"[*] Found {len(synth_df)} synthetic images from {synthetic_dir}")

    experiment_results = []
    test_preds_map = {}
    crit_std = nn.CrossEntropyLoss()

    # E0: Baseline
    print("[*] Running E0: Real Only (Scarce)...")
    ds_e0 = MobileNetLeafDataset(real_train_df, transform=base_transform)
    acc0, f10, pc0, p0, targets0 = train_and_eval(get_mobilenet_model(), DataLoader(ds_e0, batch_size=32, shuffle=True), val_loader, test_loader, crit_std, epochs=epochs)
    experiment_results.append({'ID': 'E0', 'Config': 'Real Only (Baseline)', 'Accuracy': acc0, 'Macro_F1': f10, 'Per_Class_F1': pc0})
    test_preds_map['E0'] = p0

    # E1: Classic Aug
    print("[*] Running E1: Real + Classic Aug...")
    ds_e1 = MobileNetLeafDataset(real_train_df, transform=classic_aug_transform)
    acc1, f11, pc1, p1, _ = train_and_eval(get_mobilenet_model(), DataLoader(ds_e1, batch_size=32, shuffle=True), val_loader, test_loader, crit_std, epochs=epochs)
    experiment_results.append({'ID': 'E1', 'Config': 'Real + Classic Aug', 'Accuracy': acc1, 'Macro_F1': f11, 'Per_Class_F1': pc1})
    test_preds_map['E1'] = p1

    # E2: Class Weights
    print("[*] Running E2: Real + Class Weights...")
    counts = real_train_df['class'].value_counts()
    w = torch.tensor([1.0 / counts[c] for c in TARGET_CLASSES], dtype=torch.float).to(device)
    crit_w = nn.CrossEntropyLoss(weight=w / w.sum())
    ds_e2 = MobileNetLeafDataset(real_train_df, transform=base_transform)
    acc2, f12, pc2, p2, _ = train_and_eval(get_mobilenet_model(), DataLoader(ds_e2, batch_size=32, shuffle=True), val_loader, test_loader, crit_w, epochs=epochs)
    experiment_results.append({'ID': 'E2', 'Config': 'Real + Class Weights', 'Accuracy': acc2, 'Macro_F1': f12, 'Per_Class_F1': pc2})
    test_preds_map['E2'] = p2

    # E3: Real + Synthetic
    print("[*] Running E3: Real + Synthetic (cVAE)...")
    df_e3 = pd.concat([real_train_df, synth_df]).reset_index(drop=True)
    ds_e3 = MobileNetLeafDataset(df_e3, transform=base_transform, apply_blur_defense=True)
    acc3, f13, pc3, p3, _ = train_and_eval(get_mobilenet_model(), DataLoader(ds_e3, batch_size=32, shuffle=True), val_loader, test_loader, crit_std, epochs=epochs)
    experiment_results.append({'ID': 'E3', 'Config': 'Real + Synthetic (cVAE)', 'Accuracy': acc3, 'Macro_F1': f13, 'Per_Class_F1': pc3})
    test_preds_map['E3'] = p3

    # Export results table
    rows = []
    for r in experiment_results:
        rw = {'ID': r['ID'], 'Config': r['Config'], 'Accuracy': round(r['Accuracy'], 4), 'Macro_F1': round(r['Macro_F1'], 4)}
        for idx, c in idx_to_class.items():
            rw[f'F1_{c}'] = round(r['Per_Class_F1'][idx], 4)
        rows.append(rw)
    res_df = pd.DataFrame(rows)
    csv_path = Path(results_dir) / "results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\n[+] Results saved -> {csv_path}")

    # Plot Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(NUM_CLASSES)
    w = 0.28
    plt.bar(x - w, res_df.iloc[0][[f'F1_{c}' for c in TARGET_CLASSES]], width=w, label='E0: Real Only (Baseline)', color='gray')
    plt.bar(x, res_df.iloc[1][[f'F1_{c}' for c in TARGET_CLASSES]], width=w, label='E1: Classic Augmentation', color='steelblue')
    plt.bar(x + w, res_df.iloc[3][[f'F1_{c}' for c in TARGET_CLASSES]], width=w, label='E3: Real + Synthetic (cVAE)', color='forestgreen')
    plt.xlabel('Leaf Disease Class', fontweight='bold')
    plt.ylabel('F1 Score on Real Test Set', fontweight='bold')
    plt.title('Impact of Generative Augmentation on Rare Leaf Classes', fontweight='bold')
    plt.xticks(x, [c.split('___')[1] if '___' in c else c for c in TARGET_CLASSES], rotation=15)
    plt.legend()
    plt.tight_layout()
    chart_path = Path(results_dir) / "per_class_f1.png"
    plt.savefig(chart_path, dpi=150)

    delta = f13 - f10
    decision = "YES - Re-export to ExecuTorch" if delta > 0.02 else "NO - Keep existing on-device .pte"
    notes_path = Path(results_dir) / "notes.md"
    with open(notes_path, "w") as f:
        f.write(f"# Part G Evaluation Summary\n\n- E0 Macro-F1: {f10:.4f}\n- E3 Macro-F1: {f13:.4f}\n- Delta: +{delta*100:.2f}%\n- ExecuTorch Decision: {decision}\n")

    print(f"[+] Evaluation finished. Outputs saved in '{results_dir}'.")
    return res_df


if __name__ == "__main__":
    run_classifier_evaluation()

