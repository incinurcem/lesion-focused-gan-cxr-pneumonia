import os
import csv
import random

"train_gan_rsna.py"

import cv2
import numpy as np
import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dataset_rsna_lesion import RSNADatasetConfig, create_rsna_dataloaders
from gan_model_rsna import LesionFocusedGenerator, PatchDiscriminator
from gan_losses_rsna import GANLoss, compute_generator_loss, compute_discriminator_loss


# =========================================================
# USER PATHS
# =========================================================
TRAIN_CSV = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/train_preprocessed.csv"
VAL_CSV   = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/val_preprocessed.csv"
TEST_CSV  = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"

# NOT: Karışıklık olmaması için versiyonu v4_aggressive yapmanı öneririm
OUTPUT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/gan_rsna_v4_aggressive"


# =========================================================
# TRAIN SETTINGS (A100 Optimized)
# =========================================================
SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 48
NUM_WORKERS = 16
PIN_MEMORY = True
NUM_EPOCHS = 40

LR_G = 2e-4
LR_D = 1e-4
BETAS = (0.5, 0.999)

SAVE_SAMPLES_EVERY = 5
NUM_SAMPLE_IMAGES = 8

USE_AUGMENTATIONS = True
USE_WEIGHTED_SAMPLER = False

# =========================================================
# LOSS WEIGHTS (AGRESİF AYARLAR)
# =========================================================
LAMBDA_ADV           = 1.0
LAMBDA_LESION        = 150.0  # KRİTİK: 12.0'dan uçurduk. Model artık lezyonu parlatmak ZORUNDA.
LAMBDA_BG            = 2.0    # Düşürüldü. Arka plan koruması çok baskıcı olmasın.
LAMBDA_ID            = 0.1    # Düşürüldü. Orijinal resmi birebir üretme zorunluluğunu gevşettik.
LAMBDA_TV            = 0.02
LAMBDA_EDGE          = 0.5    # Düşürüldü. Kenarlar çok sert korunursa pikseller yerinden oynamıyor.
LAMBDA_ROI_RECON     = 0.0
LAMBDA_DELTA_BG      = 2.0    # Düşürüldü.
LAMBDA_CONTRAST      = 10.0   # Artırıldı. Lezyon ve doku arasındaki farkı belirginleştirsin.
LESION_TARGET_MARGIN = 0.20   # %6'dan %20'ye çıkardık. "Ciddi bir parlaklık farkı istiyorum" diyoruz.


# =========================================================
# HELPERS
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 3:
        x = x.squeeze(0)
    x = x.detach().cpu().float().numpy()
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return x


def save_visual_samples(epoch: int, batch: dict, fake: torch.Tensor, save_dir: str, max_items: int = 8):
    ensure_dir(save_dir)
    images = batch["image"]
    masks = batch["mask"]
    patient_ids = batch["patient_id"]
    n = min(max_items, images.size(0))

    for i in range(n):
        image_u8 = tensor_to_uint8(images[i])
        mask_u8 = tensor_to_uint8(masks[i])
        fake_u8 = tensor_to_uint8(fake[i])
        diff_u8 = cv2.absdiff(fake_u8, image_u8)

        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        fake_bgr = cv2.cvtColor(fake_u8, cv2.COLOR_GRAY2BGR)
        
        # Farkı vurgulamak için JET colormap
        diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
        canvas = np.concatenate([image_bgr, mask_bgr, fake_bgr, diff_color], axis=1)
        cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch:03d}_{patient_ids[i]}.png"), canvas)


def init_csv_logger(csv_path: str):
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_g_total", "train_g_adv", "train_g_lesion", "train_g_bg", "train_g_id",
                "train_g_tv", "train_g_edge", "train_g_roi_recon", "train_g_delta_bg", "train_g_contrast",
                "train_d_total", "train_d_real", "train_d_fake",
                "val_g_total", "val_g_adv", "val_g_lesion", "val_g_bg", "val_g_id",
                "val_g_tv", "val_g_edge", "val_g_roi_recon", "val_g_delta_bg", "val_g_contrast",
                "val_d_total", "val_d_real", "val_d_fake",
            ])


def append_csv_logger(csv_path: str, row: list):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def build_running_dict():
    return {
        "g_total": 0.0, "g_adv": 0.0, "g_lesion": 0.0, "g_bg": 0.0, "g_id": 0.0,
        "g_tv": 0.0, "g_edge": 0.0, "g_roi_recon": 0.0, "g_delta_bg": 0.0, "g_contrast": 0.0,
        "d_total": 0.0, "d_real": 0.0, "d_fake": 0.0,
    }


def update_running_stats(running: dict, g_dict: dict, d_dict: dict):
    for k in running.keys():
        if k in g_dict:
            running[k] += float(g_dict[k].detach().item())
        elif k in d_dict:
            running[k] += float(d_dict[k].detach().item())


def average_running_stats(running: dict, num_batches: int):
    for k in running.keys():
        running[k] /= max(num_batches, 1)
    return running


# =========================================================
# TRAIN / VAL
# =========================================================
def train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, gan_loss_fn, device, scaler_g, scaler_d, use_amp: bool = True):
    generator.train(); discriminator.train()
    running = build_running_dict()
    pbar = tqdm(train_loader, desc="Train", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        image = batch["image"]; mask = batch["mask"]

        with autocast(device_type="cuda", enabled=use_amp):
            gen_out = generator(image)
            fake = gen_out["enhanced"]; delta = gen_out["delta"]

        # Discriminator step
        opt_d.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            pred_real = discriminator(image)
            pred_fake_detached = discriminator(fake.detach())
            d_dict = compute_discriminator_loss(pred_real=pred_real, pred_fake=pred_fake_detached, gan_loss_fn=gan_loss_fn)
            d_loss = d_dict["d_total"]
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d); scaler_d.update()

        # Generator step
        opt_g.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            pred_fake = discriminator(fake)
            g_dict = compute_generator_loss(
                pred_fake=pred_fake, enhanced=fake, image=image, mask=mask, delta=delta,
                gan_loss_fn=gan_loss_fn, lambda_adv=LAMBDA_ADV, lambda_lesion=LAMBDA_LESION,
                lambda_bg=LAMBDA_BG, lambda_id=LAMBDA_ID, lambda_tv=LAMBDA_TV,
                lambda_edge=LAMBDA_EDGE, lambda_roi_recon=LAMBDA_ROI_RECON,
                lambda_delta_bg=LAMBDA_DELTA_BG, lesion_target_margin=LESION_TARGET_MARGIN,
                lambda_contrast=LAMBDA_CONTRAST,
            )
            g_loss = g_dict["g_total"]
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g); scaler_g.update()

        update_running_stats(running, g_dict, d_dict)
        pbar.set_postfix({"g_total": f"{g_dict['g_total'].item():.4f}", "d_total": f"{d_dict['d_total'].item():.4f}"})

    return average_running_stats(running, len(train_loader))


@torch.no_grad()
def validate_one_epoch(generator, discriminator, val_loader, gan_loss_fn, device, use_amp: bool = True):
    generator.eval(); discriminator.eval()
    running = build_running_dict()
    first_batch_cpu = None; first_fake_cpu = None
    pbar = tqdm(val_loader, desc="Val", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        image = batch["image"]; mask = batch["mask"]

        with autocast(device_type="cuda", enabled=use_amp):
            gen_out = generator(image)
            fake = gen_out["enhanced"]; delta = gen_out["delta"]
            pred_real = discriminator(image); pred_fake = discriminator(fake)
            d_dict = compute_discriminator_loss(pred_real, pred_fake, gan_loss_fn)
            g_dict = compute_generator_loss(
                pred_fake=pred_fake, enhanced=fake, image=image, mask=mask, delta=delta,
                gan_loss_fn=gan_loss_fn, lambda_adv=LAMBDA_ADV, lambda_lesion=LAMBDA_LESION,
                lambda_bg=LAMBDA_BG, lambda_id=LAMBDA_ID, lambda_tv=LAMBDA_TV,
                lambda_edge=LAMBDA_EDGE, lambda_roi_recon=LAMBDA_ROI_RECON,
                lambda_delta_bg=LAMBDA_DELTA_BG, lesion_target_margin=LESION_TARGET_MARGIN,
                lambda_contrast=LAMBDA_CONTRAST,
            )

        update_running_stats(running, g_dict, d_dict)
        if batch_idx == 0:
            first_batch_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            first_fake_cpu = fake.detach().cpu()

    return average_running_stats(running, len(val_loader)), first_batch_cpu, first_fake_cpu


# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(SEED)
    ensure_dir(OUTPUT_DIR)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    sample_dir = os.path.join(OUTPUT_DIR, "samples")
    log_csv = os.path.join(OUTPUT_DIR, "training_log.csv")
    ensure_dir(checkpoint_dir); ensure_dir(sample_dir); init_csv_logger(log_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    print("=" * 90); print("TRAIN CONFIG (AGGRESSIVE MODE)"); print("=" * 90)
    print(f"Device               : {device}")
    print(f"LAMBDA_LESION        : {LAMBDA_LESION}")
    print(f"LESION_TARGET_MARGIN : {LESION_TARGET_MARGIN}")
    print("=" * 90)

    config = RSNADatasetConfig(
        image_size=IMAGE_SIZE, image_mean=0.5, image_std=0.5, soft_mask=True,
        soft_mask_kernel=21, use_augmentations=USE_AUGMENTATIONS, task_mode="gan",
        return_original_mask=True,
    )

    _, _, _, train_loader, val_loader, _ = create_rsna_dataloaders(
        TRAIN_CSV, VAL_CSV, TEST_CSV, config, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_WEIGHTED_SAMPLER
    )

    generator = LesionFocusedGenerator(in_channels=1, base_channels=64, out_channels=1, use_tanh=True).to(device)
    discriminator = PatchDiscriminator(in_channels=1, base_channels=64).to(device)

    opt_g = Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    opt_d = Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)

    scaler_g = GradScaler("cuda", enabled=use_amp); scaler_d = GradScaler("cuda", enabled=use_amp)
    gan_loss_fn = GANLoss(); best_val_g = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, gan_loss_fn, device, scaler_g, scaler_d, use_amp)
        val_stats, val_first_batch, val_first_fake = validate_one_epoch(generator, discriminator, val_loader, gan_loss_fn, device, use_amp)

        print(f"[Epoch {epoch:03d}] train_g={train_stats['g_total']:.4f} val_g={val_stats['g_total']:.4f} val_d={val_stats['d_total']:.4f}")
        append_csv_logger(log_csv, [epoch] + [train_stats[k] for k in train_stats] + [val_stats[k] for k in val_stats])

        if val_first_batch is not None and val_first_fake is not None:
            if epoch == 1 or epoch % SAVE_SAMPLES_EVERY == 0:
                save_visual_samples(epoch, val_first_batch, val_first_fake, os.path.join(sample_dir, f"epoch_{epoch:03d}"))

        torch.save({"generator_state_dict": generator.state_dict()}, os.path.join(checkpoint_dir, "last.pt"))
        if val_stats["g_total"] < best_val_g:
            best_val_g = val_stats["g_total"]
            torch.save({"generator_state_dict": generator.state_dict()}, os.path.join(checkpoint_dir, "best.pt"))
            print(f"[INFO] Yeni best checkpoint kaydedildi. best_val_g={best_val_g:.6f}")

    print("=" * 90); print("TRAINING FINISHED"); print("=" * 90)


if __name__ == "__main__":
    main()