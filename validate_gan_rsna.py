# -*- coding: utf-8 -*-
"""
validate_gan_rsna.py

Amaç: 
- Maskesiz çalışan (Leakage-free) generator modelini test etmek.
- Modelin lezyonlu bölgeleri kendi başına bulup bulmadığını sayısal (L/B Ratio) 
  ve görsel (Difference Heatmap) olarak kanıtlamak.
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------
SCRIPT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/scripts"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from gan_model_rsna import LesionFocusedGenerator
except ImportError as e:
    print(f"[HATA] gan_model_rsna.py dosyasına ulaşılamadı. Lütfen yolu kontrol et: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# HELPERS (TEMEL YARDIMCILAR)
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_grayscale(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

def read_mask(path: str, image_size: int, threshold: int = 127) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((image_size, image_size), dtype=np.float32)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.float32)

def normalize_to_tanh(img_uint8: np.ndarray) -> np.ndarray:
    return (img_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0

def denormalize_from_tanh(img_tensor: torch.Tensor) -> torch.Tensor:
    return (img_tensor.clamp(-1, 1) + 1.0) / 2.0

def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 3: x = x.squeeze(0)
    x = x.detach().cpu().numpy()
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

# ---------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------
def make_overlay_contour(image_uint8: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    mask_u8 = (mask_float > 0.1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)
    return rgb

def sobel_magnitude(img_uint8: np.ndarray) -> np.ndarray:
    img = img_uint8.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

def paste_text(img_bgr: np.ndarray, text: str, y: int = 18) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return out

def make_panel(img_u8, enh_u8, diff_u8, mask_f, l_diff, b_diff, edge_s, boost):
    p1 = make_overlay_contour(img_u8, mask_f)
    p1 = paste_text(p1, "Input + GT Contour")
    
    p2 = make_overlay_contour(enh_u8, mask_f)
    p2 = paste_text(p2, f"Enhanced (Edge:{edge_s:.3f})")
    
    diff_vis = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    diff_vis = paste_text(diff_vis, f"JET Map ({boost}x) L:{l_diff:.3f}")
    
    return np.concatenate([p1, p2, diff_vis], axis=1)

# ---------------------------------------------------------
# DATASET CLASS (HATAYI ÇÖZEN KISIM BURASI!)
# ---------------------------------------------------------
class RSNALesionInferenceDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 256):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        
        valid_indices = []
        for i, row in self.df.iterrows():
            if os.path.exists(str(row["image_path_png"])):
                valid_indices.append(i)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_u8 = read_grayscale(str(row["image_path_png"]), self.image_size)
        mask_f = read_mask(str(row["mask_path_png"]), self.image_size)
        return {
            "patient_id": str(row["patientId"]),
            "image": torch.from_numpy(normalize_to_tanh(img_u8)).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_f).unsqueeze(0).float()
        }

# ---------------------------------------------------------
# CORE VALIDATION LOGIC
# ---------------------------------------------------------
@torch.no_grad()
def run_validation(model, loader, device, save_dir, max_panels=64, boost_factor=10.0):
    ensure_dir(save_dir)
    panel_dir = os.path.join(save_dir, "panels"); ensure_dir(panel_dir)
    
    results = {"lesion_diff": [], "bg_diff": [], "lb_ratio": [], "edge_score": []}
    model.eval()
    panel_count = 0

    for batch in tqdm(loader, desc="Sayısal ve Görsel Analiz"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        patient_ids = batch["patient_id"]

        gen_out = model(images)
        fake_01 = denormalize_from_tanh(gen_out["enhanced"])
        inp_01 = denormalize_from_tanh(images)
        abs_diff_tensor = torch.abs(fake_01 - inp_01)

        for i in range(images.size(0)):
            img_u8 = tensor_to_uint8(inp_01[i])
            enh_u8 = tensor_to_uint8(fake_01[i])
            mask_f = masks[i].squeeze().cpu().numpy()

            diff_f = abs_diff_tensor[i].squeeze().cpu().numpy()
            l_pix = mask_f.sum()
            b_pix = (1.0 - mask_f).sum()
            
            l_diff = float((diff_f * mask_f).sum() / (l_pix + 1e-8)) if l_pix > 0 else 0
            b_diff = float((diff_f * (1.0 - mask_f)).sum() / (b_pix + 1e-8))
            ratio = l_diff / (b_diff + 1e-8)
            
            edge_in = sobel_magnitude(img_u8); edge_out = sobel_magnitude(enh_u8)
            bg_edge_change = np.mean(np.abs(edge_out - edge_in) * (1.0 - mask_f))
            edge_s = 1.0 / (1.0 + bg_edge_change)

            results["lesion_diff"].append(l_diff)
            results["bg_diff"].append(b_diff)
            results["lb_ratio"].append(ratio)
            results["edge_score"].append(edge_s)

            if panel_count < max_panels:
                boosted_diff = (abs_diff_tensor[i] * boost_factor).clamp(0, 1)
                diff_u8_boosted = tensor_to_uint8(boosted_diff)
                
                panel = make_panel(img_u8, enh_u8, diff_u8_boosted, mask_f, l_diff, b_diff, edge_s, boost_factor)
                cv2.imwrite(os.path.join(panel_dir, f"{patient_ids[i]}.png"), panel)
                panel_count += 1

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in results.items()}
    with open(os.path.join(save_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PHD BİLİMSEL VALIDASYON ÖZETİ")
    print("="*60)
    print(f"L/B Ratio Mean       : {summary['lb_ratio']['mean']:.4f}")
    print(f"Edge Consistency     : {summary['edge_score']['mean']:.4f}")
    print(f"Visual Boost         : {boost_factor}x")
    print("="*60)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--gen_base_channels", type=int, default=64)
    parser.add_argument("--max_panels", type=int, default=64)
    parser.add_argument("--boost_factor", type=float, default=10.0)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LesionFocusedGenerator(in_channels=1, base_channels=args.gen_base_channels).to(device)
    
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("generator_state_dict", ckpt.get("generator", ckpt))
    model.load_state_dict(state_dict)

    loader = DataLoader(
        RSNALesionInferenceDataset(args.csv_path, args.image_size), 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )

    run_validation(model, loader, device, args.output_dir, args.max_panels, args.boost_factor)

if __name__ == "__main__":
    main()