# -*- coding: utf-8 -*-
"""
leakage_sanity_check.py

Shuffled-label test:
- Train CSV'sindeki etiketleri rastgele permüte et.
- 3 epoch eğit.
- Val AUC ≈ 0.50 olmalı. > 0.60 ise pipeline hâlâ sızdırıyor demektir.
"""

import os, sys, copy, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/scripts"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Aynı dataset/model yardımcılarını yeniden kullanıyoruz
from run_downstream_classifier_evaluation3 import (
    Config, load_and_prepare_csv, RSNADownstreamDataset,
    build_model, run_one_epoch, get_pos_weight,
)
from torch.utils.data import DataLoader


def shuffled_label_test(input_type="gan_blended", model_name="resnet50",
                        epochs=3, seed=999):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    cfg = Config()
    cfg.epochs = epochs
    cfg.early_stopping_patience = 999  # erken durmasın
    cfg.batch_size = 64
    cfg.num_workers = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = load_and_prepare_csv(cfg.train_csv, cfg, "train")
    val_df   = load_and_prepare_csv(cfg.val_csv,   cfg, "val")

    # ETİKETLERİ KARIŞTIR
    rng = np.random.default_rng(seed)
    train_df["label_bin"] = rng.permutation(train_df["label_bin"].values)
    val_df["label_bin"]   = rng.permutation(val_df["label_bin"].values)
    print(f"[INFO] Etiketler permüte edildi. "
          f"train pos oranı: {train_df['label_bin'].mean():.3f}, "
          f"val pos oranı: {val_df['label_bin'].mean():.3f}")

    train_ds = RSNADownstreamDataset(train_df, cfg, input_type, train=True)
    val_ds   = RSNADownstreamDataset(val_df,   cfg, input_type, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(model_name, cfg.pretrained, cfg.dropout).to(device)
    pw = get_pos_weight(train_df)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=="cuda"))

    print("\n" + "="*80)
    print(f"SHUFFLED-LABEL TEST  |  input={input_type}  model={model_name}")
    print("="*80)

    val_aucs = []
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, _, _ = run_one_epoch(model, train_loader, criterion, optimizer, device,
                                      cfg.amp and device.type=="cuda", scaler, True)
        vl_loss, vl_m, _ = run_one_epoch(model, val_loader, criterion, optimizer, device,
                                         cfg.amp and device.type=="cuda", None, False)
        val_aucs.append(vl_m["roc_auc"])
        print(f"[ep {ep+1}/{epochs}] tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_m['roc_auc']:.4f}  ({time.time()-t0:.1f}s)")

    max_auc = max(val_aucs)
    print("\n" + "-"*80)
    print(f"En yüksek val AUC (shuffled): {max_auc:.4f}")
    if max_auc < 0.55:
        print(f"✓ TEMİZ: input_type={input_type} sızdırmıyor (AUC ≈ 0.5).")
        return True
    elif max_auc < 0.60:
        print(f"~ ŞÜPHELİ: AUC = {max_auc:.3f}. Hafif sızıntı veya istatistiksel gürültü olabilir.")
        return None
    else:
        print(f"✗ SIZINTI VAR: input_type={input_type} hâlâ sızdırıyor. AUC = {max_auc:.3f}.")
        return False


if __name__ == "__main__":
    results = {}
    for it in ["original", "global", "gan_full", "gan_blended"]:
        results[it] = shuffled_label_test(input_type=it, model_name="resnet50", epochs=3)

    print("\n" + "="*80)
    print("ÖZET")
    print("="*80)
    for k, v in results.items():
        tag = "✓ temiz" if v is True else ("? şüpheli" if v is None else "✗ sızıntı")
        print(f"  {k:15s} -> {tag}")