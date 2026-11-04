"""
AgriSense AI — Module 5: Role F (Generator)
File: modules/gan_vae/generator.py

This module defines the Conditional VAE (cVAE) architecture, training procedure,
and synthetic image generator for rare plant disease classes.
"""

import os
import sys
import json
import time
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Default Target Rare Classes
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
class RareLeafDataset(Dataset):
    def __init__(self, samples_list, transform=None):
        self.samples = samples_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transform(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def build_dataset_samples(raw_data_dir="data/raw", splits_csv="data/raw/splits.csv"):
    """
    Discovers training samples:
    1. Checks splits_csv if available (strictly split == 'train').
    2. Checks raw_data_dir for class folders.
    3. Falls back to generating demo mock images if no real dataset is present.
    """
    samples = []
    
    # Priority 1: splits.csv
    if os.path.exists(splits_csv):
        print(f"[*] Reading split file: {splits_csv}")
        df = pd.read_csv(splits_csv)
        train_df = df[df['split'] == 'train']
        for _, row in train_df.iterrows():
            cls_name = row['class']
            if cls_name in class_to_idx and os.path.exists(row['path']):
                samples.append((row['path'], class_to_idx[cls_name]))
                
    # Priority 2: Direct raw folder scan
    if not samples and os.path.exists(raw_data_dir):
        print(f"[*] Scanning raw dataset directory: {raw_data_dir}")
        for cls_name, cls_idx in class_to_idx.items():
            folder = Path(raw_data_dir) / cls_name
            if folder.exists():
                for img_p in folder.iterdir():
                    if img_p.is_file() and img_p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        samples.append((str(img_p), cls_idx))
                        
    # Priority 3: Fallback demo mock images for dry runs
    if not samples:
        print("[!] No real PlantVillage data found. Creating demo mock leaves for testing...")
        demo_dir = Path("data/raw/demo_plantvillage")
        demo_dir.mkdir(parents=True, exist_ok=True)
        colors = [(34, 139, 34), (139, 69, 19), (60, 179, 113), (46, 139, 87), (205, 92, 92)]
        for cls_idx, (cls_name, col) in enumerate(zip(TARGET_CLASSES, colors)):
            cls_dir = demo_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(120):
                img = Image.new('RGB', (128, 128), color=(220, 220, 220))
                draw = ImageDraw.Draw(img)
                box = [20 + random.randint(-5, 5), 20 + random.randint(-5, 5), 108 + random.randint(-5, 5), 108 + random.randint(-5, 5)]
                draw.ellipse(box, fill=(col[0] + random.randint(-15, 15), col[1] + random.randint(-15, 15), col[2] + random.randint(-15, 15)))
                img_path = cls_dir / f"sample_{i:04d}.png"
                img.save(img_path)
                samples.append((str(img_path), cls_idx))
        print(f"[+] Created {len(samples)} demo images under {demo_dir}")

    return samples


# ----------------- CONDITIONAL VAE ARCHITECTURE -----------------
class ConditionalVAE(nn.Module):
    def __init__(self, num_classes=5, latent_dim=128, embed_dim=32):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Encoder: 64x64 -> 4x4x512
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc_fc_mu = nn.Linear(512 * 4 * 4 + embed_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(512 * 4 * 4 + embed_dim, latent_dim)
        
        # Decoder: 4x4x512 -> 64x64x3
        self.dec_input = nn.Linear(latent_dim + embed_dim, 512 * 4 * 4)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x, y):
        h = self.enc_conv1(x)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)
        h = self.enc_conv4(h)
        h = torch.flatten(h, start_dim=1)
        y_embed = self.label_embedding(y)
        h_cond = torch.cat([h, y_embed], dim=1)
        mu = self.enc_fc_mu(h_cond)
        logvar = self.enc_fc_logvar(h_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_embed = self.label_embedding(y)
        z_cond = torch.cat([z, y_embed], dim=1)
        h = self.dec_input(z_cond)
        h = h.view(-1, 512, 4, 4)
        h = self.dec_conv1(h)
        h = self.dec_conv2(h)
        h = self.dec_conv3(h)
        out = self.dec_conv4(h)
        return out

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

    def sample(self, num_samples, class_idx, target_device=None, truncation=1.5):
        if target_device is None:
            target_device = device
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=target_device)
            if truncation > 0:
                z = torch.clamp(z, -truncation, truncation)
            y = torch.full((num_samples,), class_idx, dtype=torch.long, device=target_device)
            samples = self.decode(z, y)
        return samples


def cvae_loss_fn(recon_x, x, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ----------------- TRAINING & GENERATION API -----------------
def train_cvae(raw_data_dir="data/raw", splits_csv="data/raw/splits.csv",
               save_model_path="modules/gan_vae/cvae.pt",
               epochs=60, batch_size=64, lr=1e-3):
    """
    Trains the Conditional VAE on rare class images.
    """
    print(f"[*] Training cVAE on device: {device}")
    samples_data = build_dataset_samples(raw_data_dir=raw_data_dir, splits_csv=splits_csv)
    dataset = RareLeafDataset(samples_data, transform=get_train_transform(64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ConditionalVAE(num_classes=NUM_CLASSES, latent_dim=128, embed_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        beta = min(0.001, 0.001 * (epoch / 15.0))
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_y)
            loss, _, _ = cvae_loss_fn(recon_x, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {running_loss/len(dataset):.4f}")
            
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"[+] Saved model checkpoint -> {save_model_path}")
    return model


def generate_synthetic_samples(model=None, weights_path="modules/gan_vae/cvae.pt",
                               output_dir="data/processed/synthetic",
                               num_per_class=400, upscale_size=224):
    """
    Generates synthetic leaf images (224x224 bicubic) and saves them into output_dir/<class_name>/*.png
    """
    if model is None:
        model = ConditionalVAE(num_classes=NUM_CLASSES, latent_dim=128, embed_dim=32).to(device)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print(f"[!] Warning: {weights_path} not found. Generating with uninitialized model.")
            
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Generating {num_per_class} images/class upscaled to {upscale_size}x{upscale_size}...")
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = output_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        while count < num_per_class:
            bsz = min(50, num_per_class - count)
            s = model.sample(bsz, cls_idx, device, truncation=1.5)
            s = torch.clamp((s + 1.0) / 2.0, 0.0, 1.0)
            s_up = F.interpolate(s, size=(upscale_size, upscale_size), mode='bicubic', align_corners=False)
            for i in range(bsz):
                save_image(s_up[i], str(cls_dir / f"{count + 1:04d}.png"))
                count += 1
        print(f"    [+] Saved {count} images -> {cls_dir}")
        
    print(f"[+] Generation complete in: {output_dir}")
    return str(output_path)


if __name__ == "__main__":
    trained_model = train_cvae()
    generate_synthetic_samples(model=trained_model)

