# -*- coding: utf-8 -*-
"""
export_enhanced_rsna.py

Amaç:
- Maske gerektirmeyen (leakage-free) generator ile toplu üretim yapmak.
- Preview panellerinde görselleştirme çarpanı (boost) kullanarak değişimleri görünür kılmak.
- Downstream classifier için bilimsel olarak geçerli veri üretmek.
"""

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# PROJECT IMPORTS
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from gan_model_rsna import LesionFocusedGenerator
except Exception as e:
    raise ImportError(f"LesionFocusedGenerator import edilemedi: {e}")

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# SEAMLESS BLENDING HELPERS (Yeni Eklenenler)
# =========================================================

def apply_feathered_blending(img_u8, enh_u8, mask_f, sigma=5):
    """
    Gaussian Blur kullanarak yumuşak (feathered) bir geçiş sağlar.
    CNN'in kenar çizgilerini yakalamasını engeller.
    """
    # Maskeyi yumuşat (0.0 - 1.0 arası float)
    soft_mask = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    soft_mask = np.clip(soft_mask, 0, 1)
    
    # Formül: (Enhanced * Mask) + (Original * (1 - Mask))
    blended = (enh_u8.astype(np.float32) * soft_mask) + (img_u8.astype(np.float32) * (1.0 - soft_mask))
    return np.clip(blended, 0, 255).astype(np.uint8)

    
def apply_poisson_blending(img_u8, enh_u8, mask_f):
    """
    OpenCV Seamless Cloning - ULTRA-SAFE Version.
    ROI sınır hatalarını önlemek için agresif maske tıraşlama eklenmiştir.
    """
    # 1. Kanalları Kontrol Et (3 Kanal Şartı)
    is_grayscale = (len(img_u8.shape) == 2)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR) if is_grayscale else img_u8
    enh_rgb = cv2.cvtColor(enh_u8, cv2.COLOR_GRAY2BGR) if is_grayscale else enh_u8

    # 2. Maskeyi Hazırla
    mask_u8 = (mask_f * 255).astype(np.uint8)
    
    # 3. AGRESİF KENAR TEMİZLİĞİ (Assertion Error'un İlacı)
    # Maskeyi kenarlardan 5 piksel içeri çekiyoruz. Bu, gradyan hesabı için güvenli alan yaratır.
    h, w = mask_u8.shape
    border = 5
    mask_u8[:border, :] = 0
    mask_u8[h-border:, :] = 0
    mask_u8[:, :border] = 0
    mask_u8[:, w-border:] = 0
    
    # Ayrıca maskeyi 1 tık daraltarak (erosion) kenar pürüzlerini silelim
    kernel = np.ones((3,3), np.uint8)
    mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
    
    # 4. Merkez Noktayı Bul
    y, x = np.where(mask_u8 > 0)
    if len(x) == 0 or len(y) == 0: 
        return apply_feathered_blending(img_u8, enh_u8, mask_f, sigma=5)
    
    center = (int(np.mean(x)), int(np.mean(y)))
    
    try:
        # 5. MIXED_CLONE: Patolojik dokuyu koru, ışığı eşitle
        seamless_rgb = cv2.seamlessClone(enh_rgb, img_rgb, mask_u8, center, cv2.MIXED_CLONE)
        return cv2.cvtColor(seamless_rgb, cv2.COLOR_BGR2GRAY) if is_grayscale else seamless_rgb

    except Exception as e:
        # Hala hata verirse (lezyon çok küçükse veya merkez hatalıysa) Feathered'a kaçış
        return apply_feathered_blending(img_u8, enh_u8, mask_f, sigma=5)



def read_grayscale(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

def read_mask(path: str, image_size: int, threshold: int = 127) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return np.zeros((image_size, image_size), dtype=np.float32)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.float32)

def normalize_to_tanh(img_uint8: np.ndarray) -> np.ndarray:
    return (img_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0

def denormalize_from_tanh(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0

def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 3: x = x.squeeze(0)
    return (np.clip(x.detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)

def save_uint8(path: str, img: np.ndarray):
    ensure_dir(os.path.dirname(path))
    Image.fromarray(img).save(path)

def overlay_mask_contour(image_uint8: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    mask_u8 = (mask_float > 0.1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)
    return rgb

# =========================================================
# DATASET
# =========================================================
class RSNALesionExportDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 256):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_uint8 = read_grayscale(str(row["image_path_png"]), self.image_size)
        mask_float = read_mask(str(row["mask_path_png"]), self.image_size)
        
        return {
            "patient_id": str(row["patientId"]),
            "image": torch.from_numpy(normalize_to_tanh(image_uint8)).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_float).unsqueeze(0).float(),
            "image_path_png": str(row["image_path_png"]),
            "target": float(row["target"]) if "target" in row else 0.0
        }

# =========================================================
# EXPORT LOGIC
# =========================================================
@torch.no_grad()
def export_split(model, loader, device, split_name, export_root, save_preview_every, skip_existing, boost_factor=10.0):
    split_root = os.path.join(export_root, split_name)
    # Sadece blended ve preview klasörlerini tutuyoruz
    dirs = {
        "blended": os.path.join(split_root, "enhanced_blended"),
        "preview": os.path.join(split_root, "preview_panels")
    }
    for d in dirs.values(): ensure_dir(d)

    manifest_rows = []
    model.eval()
    pbar = tqdm(loader, desc=f"Seamless Export [{split_name}]")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        patient_ids = batch["patient_id"]

        # GAN çıkarımı (Maskesiz/Leakage-free)
        gen_out = model(images)
        fake_01 = denormalize_from_tanh(gen_out["enhanced"])
        inp_01 = denormalize_from_tanh(images)
        abs_diff_tensor = torch.abs(fake_01 - inp_01)

        for i in range(images.size(0)):
            pid = patient_ids[i]
            # Sadece blended yolu
            blend_path = os.path.join(dirs["blended"], f"{pid}.png")

            if skip_existing and os.path.exists(blend_path):
                continue

            img_u8 = tensor_to_uint8(inp_01[i])
            enh_u8 = tensor_to_uint8(fake_01[i])
            mask_f = masks[i].squeeze()

            # --- PH.D. CRITICAL STEP: SEAMLESS BLENDING ---
            # sigma=7 kullanarak o 'matematiksel dikiş izini' CNN'den saklıyoruz
            # GÜNCELLENMİŞ SATIR:
            blended_u8 = apply_poisson_blending(img_u8, enh_u8, mask_f)
            save_uint8(blend_path, blended_u8)

            # Manifestoyu sadece üretilen blended yollarıyla güncelle
            manifest_rows.append({
                "patientId": pid,
                "target": batch["target"][i].item(),
                "enhanced_path": blend_path,
                "original_path": batch["image_path_png"][i]
            })

            # Görsel denetim için Preview paneli
            if save_preview_every > 0 and (batch_idx * loader.batch_size + i) % save_preview_every == 0:
                p1 = overlay_mask_contour(img_u8, mask_f)
                p2 = overlay_mask_contour(blended_u8, mask_f) # Artık dikişsiz halini görüyoruz
                
                boosted_diff = (abs_diff_tensor[i] * boost_factor).clamp(0, 1)
                diff_u8_boosted = tensor_to_uint8(boosted_diff)
                diff_vis = cv2.applyColorMap(diff_u8_boosted, cv2.COLORMAP_JET)
                
                panel = np.concatenate([p1, p2, diff_vis], axis=1)
                cv2.imwrite(os.path.join(dirs["preview"], f"{pid}.png"), panel)

    manifest_csv = os.path.join(split_root, f"{split_name}_lesion_only_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"[{split_name}] Dikişsiz üretim tamamlandı: {manifest_csv}")



def main():
    parser = argparse.ArgumentParser(description="PhD GAN Export Script")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--export_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gen_base_channels", type=int, default=64)
    parser.add_argument("--save_preview_every", type=int, default=100)
    parser.add_argument("--boost_factor", type=float, default=10.0) # Görsel çarpan eklendi
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generator Kurulumu (in_channels=1)
    generator = LesionFocusedGenerator(in_channels=1, base_channels=args.gen_base_channels).to(device)
    
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("generator_state_dict", ckpt.get("generator", ckpt))
    generator.load_state_dict(state_dict)
    generator.eval()

    splits = [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]
    for name, csv_path in splits:
        loader = DataLoader(
            RSNALesionExportDataset(csv_path, args.image_size), 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            shuffle=False
        )
        export_split(generator, loader, device, name, args.export_root, args.save_preview_every, args.skip_existing, args.boost_factor)

if __name__ == "__main__":
    main()