# -*- coding: utf-8 -*-
"""
export_blended_label_free.py

Label-bağımsız blended görüntü export'u.
- Generator (mask-free) tüm görüntülere uygulanır.
- delta'dan label-bağımsız bir soft mask çıkartılır.
- Bu mask ile enhanced ve original arasında alpha-blend yapılır.
- Pozitif/negatif fark gözetilmez.

Kullanım:
python export_blended_label_free.py \
  --checkpoint /path/to/best.pt \
  --train_csv /path/train.csv --val_csv /path/val.csv --test_csv /path/test.csv \
  --image_root /path/to/images_png \
  --output_root /path/to/enhanced_images_v6_label_free \
  --image_size 256 --batch_size 32
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/scripts"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from gan_model_rsna import LesionFocusedGenerator


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def resolve_image_path(raw_path, image_id, root):
    candidates = []
    if raw_path and str(raw_path).strip().lower() != "nan":
        rp = str(raw_path).strip()
        candidates.append(rp)
        if not os.path.isabs(rp):
            candidates.append(os.path.join(root, rp))
        candidates.append(os.path.join(root, os.path.basename(rp)))
    if image_id and str(image_id).strip().lower() != "nan":
        stem = os.path.splitext(str(image_id).strip())[0]
        candidates.extend([
            os.path.join(root, f"{stem}.png"),
            os.path.join(root, "images_png", f"{stem}.png"),
            os.path.join(root, "train", f"{stem}.png"),
            os.path.join(root, "val", f"{stem}.png"),
            os.path.join(root, "test", f"{stem}.png"),
        ])
    for p in dict.fromkeys(candidates):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Görüntü bulunamadı. id={image_id}, raw={raw_path}, root={root}")


class ExportDataset(Dataset):
    """
    Sadece görüntüyü yükler. ETİKETİ OKUMAZ. MASKEYİ OKUMAZ.
    Bu, label-bağımsız export'un garantörüdür.
    """
    def __init__(self, csv_path, image_root, image_size, split_name):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_root = image_root
        self.image_size = image_size
        self.split_name = split_name

        # Sütun adı tespiti
        self.image_col = None
        for c in ("image_path_png", "path", "image_path"):
            if c in self.df.columns:
                self.image_col = c; break
        self.id_col = None
        for c in ("patientId", "patient_id", "id"):
            if c in self.df.columns:
                self.id_col = c; break
        if self.id_col is None:
            raise ValueError("CSV'de patientId / id sütunu bulunamadı.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])
        raw = str(row[self.image_col]) if self.image_col else None
        path = resolve_image_path(raw, sample_id, self.image_root)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # tanh aralığına normalize: [0,255] -> [-1,1]
        img_norm = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
        img_t = torch.from_numpy(img_norm).unsqueeze(0)  # [1,H,W]

        return {
            "image": img_t,
            "patient_id": sample_id,
            "split": self.split_name,
        }


@torch.no_grad()
def export_split(generator, loader, out_root, split_name, device, percentile=0.90,
                 sigmoid_temp=50.0, blend_strength=1.0):
    """
    percentile: delta_abs içinde threshold için kullanılan percentile (0.90 = en üst %10).
    sigmoid_temp: soft mask geçiş keskinliği. Yüksek = sert maske.
    blend_strength: 1.0 = tam enhancement, 0.5 = yumuşak.
    """
    out_dir = os.path.join(out_root, split_name, "enhanced_blended")
    ensure_dir(out_dir)

    n_saved = 0
    for batch in tqdm(loader, desc=f"Export [{split_name}]"):
        images = batch["image"].to(device, non_blocking=True)
        out = generator(images)
        enhanced = out["enhanced"]   # [-1,1]
        delta = out["delta"]         # [-1,1]

        # Label-free soft mask: |delta| üzerinde per-image percentile threshold
        delta_abs = delta.abs()
        b, c, h, w = delta_abs.shape
        flat = delta_abs.view(b, -1)
        k = max(1, int(percentile * flat.shape[1]))
        thr_vals = flat.kthvalue(k, dim=1).values
        thr = thr_vals.view(b, 1, 1, 1)
        soft_mask = torch.sigmoid((delta_abs - thr) * sigmoid_temp)
        soft_mask = soft_mask * blend_strength

        # Blend: enhanced * mask + image * (1-mask)
        # [-1,1] -> [0,1]
        img01 = (images + 1.0) / 2.0
        enh01 = (enhanced.clamp(-1, 1) + 1.0) / 2.0
        blended01 = enh01 * soft_mask + img01 * (1.0 - soft_mask)

        # uint8'e dönüştür ve kaydet
        blended_u8 = (blended01.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

        for i in range(b):
            arr = blended_u8[i, 0]
            pid = batch["patient_id"][i]
            cv2.imwrite(os.path.join(out_dir, f"{pid}.png"), arr)
            n_saved += 1

    print(f"[{split_name}] {n_saved} görüntü kaydedildi -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gen_base_channels", type=int, default=64)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--sigmoid_temp", type=float, default=50.0)
    parser.add_argument("--blend_strength", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_root)

    # Generator yükle
    generator = LesionFocusedGenerator(
        in_channels=1, base_channels=args.gen_base_channels,
        out_channels=1, use_tanh=True
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("generator_state_dict", ckpt.get("generator", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    print(f"[OK] Generator yüklendi: {args.checkpoint}")

    splits = [
        ("train", args.train_csv),
        ("val",   args.val_csv),
        ("test",  args.test_csv),
    ]

    for split_name, csv_p in splits:
        ds = ExportDataset(csv_p, args.image_root, args.image_size, split_name)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        export_split(
            generator, loader, args.output_root, split_name, device,
            percentile=args.percentile,
            sigmoid_temp=args.sigmoid_temp,
            blend_strength=args.blend_strength,
        )

    print("\n[BİTTİ] Label-free blended export tamamlandı.")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()