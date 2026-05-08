# ============================================================
# train_baseline_classifiers_rsna.py
# ------------------------------------------------------------
# Baseline experiments for seminar project:
#   1) original image -> classifier
#   2) classical/global enhancement -> classifier
#   3) lesion-focused GAN enhanced image -> classifier
#
# Works with RSNA pneumonia classification CSVs.
# Designed for Google Colab / A100 / single script usage.
# ============================================================

import os
import gc
import cv2
import json
import math
import time
import copy
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

warnings.filterwarnings("ignore")


# ============================================================
# 0. GLOBAL SETTINGS
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except:
        pass

seed_everything(42)


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    # ---------------------------
    # CSV paths
    # ---------------------------
    train_csv: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/train_preprocessed.csv"
    val_csv: str   = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/val_preprocessed.csv"
    test_csv: str  = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"

    # ---------------------------
    # Image roots
    # original images
    # These are used if image_path in CSV is relative or missing.
    # ---------------------------
    original_image_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/images_png"

    # lesion-focused GAN enhanced export root
    lesion_enhanced_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images2"

    # output root
    output_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/baseline_classifier_comparison"

    # ---------------------------
    # training params
    # ---------------------------
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    amp: bool = True
    model_name: str = "resnet50"   # resnet18 / resnet34 / resnet50 / densenet121
    pretrained: bool = True
    early_stopping_patience: int = 5

    # ---------------------------
    # experiment names
    # ---------------------------
    run_original: bool = True
    run_global_enhanced: bool = True
    run_lesion_gan_enhanced: bool = True

    # ---------------------------
    # global enhancement settings
    # ---------------------------
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 1.0
    unsharp_sigma: float = 1.0

    # ---------------------------
    # normalization
    # grayscale X-ray -> repeat 3 channels
    # ---------------------------
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

    # ---------------------------
    # column names (auto-infer if possible)
    # ---------------------------
    image_col_candidates: Tuple[str, ...] = (
        "image_path","image_path_png", "path", "png_path", "img_path", "filepath", "file_path"
    )
    label_col_candidates: Tuple[str, ...] = (
        "label", "target", "class", "pneumonia", "has_pneumonia"
    )
    id_col_candidates: Tuple[str, ...] = (
        "patientId", "patient_id", "id", "image_id"
    )


CFG = Config()


# ============================================================
# 2. UTILS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_first_existing_column(df: pd.DataFrame, candidates: Tuple[str, ...], required=True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of these columns in CSV: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def infer_binary_label(series: pd.Series) -> pd.Series:
    """
    Converts label column to binary int {0,1} robustly.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "0": 0, "1": 1,
        "false": 0, "true": 1,
        "negative": 0, "positive": 1,
        "normal": 0, "pneumonia": 1,
        "no": 0, "yes": 1
    }

    out = []
    for x in series.astype(str).str.strip().str.lower().tolist():
        if x in mapping:
            out.append(mapping[x])
        else:
            raise ValueError(f"Unknown label value encountered: {x}")
    return pd.Series(out, index=series.index, dtype=np.int64)


def build_path_from_id(image_id: str, root: str) -> str:
    """
    Assumes file name is <patientId>.png
    """
    return os.path.join(root, f"{image_id}.png")


def read_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    return img


def apply_global_enhancement(
    img: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    unsharp_amount: float = 1.0,
    unsharp_sigma: float = 1.0
) -> np.ndarray:
    """
    Classical/global enhancement.
    Entire image is enhanced equally.
    """
    out = img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size)
        )
        out = clahe.apply(out)

    if unsharp_amount > 0:
        blur = cv2.GaussianBlur(out, (0, 0), unsharp_sigma)
        out = cv2.addWeighted(out, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def build_transforms(image_size: int, mean, std, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ============================================================
# 3. DATA PREP
# ============================================================

def load_and_prepare_csv(csv_path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    image_col = find_first_existing_column(df, cfg.image_col_candidates, required=False)
    label_col = find_first_existing_column(df, cfg.label_col_candidates, required=True)
    id_col    = find_first_existing_column(df, cfg.id_col_candidates, required=False)

    df = df.copy()
    df["label_bin"] = infer_binary_label(df[label_col])

    # image path resolution
    if image_col is not None:
        df["resolved_image_path"] = df[image_col].astype(str)

        # if path is relative, join with original_image_root
        def _resolve(p):
            if os.path.isabs(p):
                return p
            return os.path.join(cfg.original_image_root, p)

        df["resolved_image_path"] = df["resolved_image_path"].apply(_resolve)

    elif id_col is not None:
        df["resolved_image_path"] = df[id_col].astype(str).apply(lambda x: build_path_from_id(x, cfg.original_image_root))

    else:
        raise ValueError(
            "CSV must contain either an image path column "
            f"{cfg.image_col_candidates} or an ID column {cfg.id_col_candidates}"
        )

    # patient/image id for lesion enhanced image matching
    if id_col is not None:
        df["sample_id"] = df[id_col].astype(str)
    else:
        # derive from file name
        df["sample_id"] = df["resolved_image_path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

    return df


# ============================================================
# 4. DATASET
# ============================================================

class RSNAClassifierDataset(Dataset):
    """
    mode:
      - original
      - global
      - lesion_gan
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        mode: str = "original",
        train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.train = train
        self.transform = build_transforms(cfg.image_size, cfg.mean, cfg.std, train=train)

        assert self.mode in ["original", "global", "lesion_gan"]

    def __len__(self):
        return len(self.df)

    def _get_original_path(self, row) -> str:
        return row["resolved_image_path"]

    def _get_lesion_gan_path(self, row) -> str:
        sample_id = row["sample_id"]

        # common naming assumptions
        candidate_paths = [
            os.path.join(self.cfg.lesion_enhanced_root, f"{sample_id}.png"),
            os.path.join(self.cfg.lesion_enhanced_root, f"{sample_id}.jpg"),
            os.path.join(self.cfg.lesion_enhanced_root, "images", f"{sample_id}.png"),
            os.path.join(self.cfg.lesion_enhanced_root, "images", f"{sample_id}.jpg"),
            os.path.join(self.cfg.lesion_enhanced_root, "enhanced", f"{sample_id}.png"),
            os.path.join(self.cfg.lesion_enhanced_root, "enhanced", f"{sample_id}.jpg"),
        ]

        for p in candidate_paths:
            if os.path.exists(p):
                return p

        # recursive fallback: expensive only if missing
        for root, _, files in os.walk(self.cfg.lesion_enhanced_root):
            for f in files:
                stem = os.path.splitext(f)[0]
                if stem == sample_id:
                    return os.path.join(root, f)

        raise FileNotFoundError(
            f"Lesion-focused GAN enhanced image not found for ID={sample_id} "
            f"under {self.cfg.lesion_enhanced_root}"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label_bin"])

        if self.mode == "original":
            img_path = self._get_original_path(row)
            img = read_grayscale_image(img_path)

        elif self.mode == "global":
            img_path = self._get_original_path(row)
            img = read_grayscale_image(img_path)
            img = apply_global_enhancement(
                img=img,
                use_clahe=self.cfg.use_clahe,
                clahe_clip_limit=self.cfg.clahe_clip_limit,
                clahe_tile_grid_size=self.cfg.clahe_tile_grid_size,
                unsharp_amount=self.cfg.unsharp_amount,
                unsharp_sigma=self.cfg.unsharp_sigma
            )

        elif self.mode == "lesion_gan":
            img_path = self._get_lesion_gan_path(row)
            img = read_grayscale_image(img_path)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        img = gray_to_rgb(img)
        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "id": row["sample_id"],
            "path": img_path
        }


# ============================================================
# 5. MODEL
# ============================================================

def build_model(model_name: str = "resnet50", pretrained: bool = True, dropout: float = 0.30):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


# ============================================================
# 6. METRICS
# ============================================================

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp + 1e-12)
    npv = tn / (tn + fn + 1e-12)
    ppv = tp / (tp + fp + 1e-12)
    sensitivity = rec
    balanced_acc = (sensitivity + specificity) / 2.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 7. TRAIN / VALID / TEST
# ============================================================

def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    train: bool = True,
    amp: bool = True
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_ids = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labs = labels.detach().cpu().numpy().reshape(-1)

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_ids.extend(batch["id"])

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_probs))

    return epoch_loss, metrics, np.array(all_labels), np.array(all_probs), all_ids


def create_loaders(cfg: Config, mode: str):
    train_df = load_and_prepare_csv(cfg.train_csv, cfg)
    val_df   = load_and_prepare_csv(cfg.val_csv, cfg)
    test_df  = load_and_prepare_csv(cfg.test_csv, cfg)

    train_ds = RSNAClassifierDataset(train_df, cfg, mode=mode, train=True)
    val_ds   = RSNAClassifierDataset(val_df, cfg, mode=mode, train=False)
    test_ds  = RSNAClassifierDataset(test_df, cfg, mode=mode, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def train_single_experiment(cfg: Config, experiment_name: str, mode: str) -> Dict:
    print("\n" + "=" * 90)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 90)

    experiment_dir = os.path.join(cfg.output_root, experiment_name)
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    pred_dir = os.path.join(experiment_dir, "predictions")
    ensure_dir(experiment_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(pred_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_loaders(cfg, mode)

    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")
    print(f"Test size : {len(test_df)}")

    model = build_model(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_loss, train_metrics, _, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            amp=cfg.amp and device.type == "cuda"
        )

        val_loss, val_metrics, val_y, val_prob, val_ids = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            train=False,
            amp=cfg.amp and device.type == "cuda"
        )

        scheduler.step(val_metrics["roc_auc"])

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - t0
        }
        history.append(epoch_row)

        print(
            f"[{epoch+1:03d}/{cfg.epochs:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"time={epoch_row['time_sec']:.1f}s"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "config": asdict(cfg),
                "experiment_name": experiment_name,
                "mode": mode
            }
            torch.save(best_state, os.path.join(ckpt_dir, "best.pt"))

            val_pred_df = pd.DataFrame({
                "id": val_ids,
                "y_true": val_y.astype(int),
                "y_prob": val_prob
            })
            val_pred_df.to_csv(os.path.join(pred_dir, "val_predictions_best.csv"), index=False)

        else:
            patience_counter += 1

        pd.DataFrame(history).to_csv(os.path.join(experiment_dir, "history.csv"), index=False)

        if patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # load best
    checkpoint = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"])

    # final test
    test_loss, test_metrics, test_y, test_prob, test_ids = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=None,
        train=False,
        amp=cfg.amp and device.type == "cuda"
    )

    test_pred_df = pd.DataFrame({
        "id": test_ids,
        "y_true": test_y.astype(int),
        "y_prob": test_prob,
        "y_pred": (test_prob >= 0.5).astype(int)
    })
    test_pred_df.to_csv(os.path.join(pred_dir, "test_predictions.csv"), index=False)

    result = {
        "experiment_name": experiment_name,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_loss": test_loss,
        **test_metrics
    }

    save_json(result, os.path.join(experiment_dir, "test_metrics.json"))
    print("\nTEST RESULTS")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.6f}")
        else:
            print(f"{k:20s}: {v}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# 8. MAIN EXPERIMENT MANAGER
# ============================================================

def run_all_experiments(cfg: Config):
    ensure_dir(cfg.output_root)
    save_json(asdict(cfg), os.path.join(cfg.output_root, "config.json"))

    all_results = []

    if cfg.run_original:
        result_original = train_single_experiment(
            cfg=cfg,
            experiment_name="baseline_original_image",
            mode="original"
        )
        all_results.append(result_original)

    if cfg.run_global_enhanced:
        result_global = train_single_experiment(
            cfg=cfg,
            experiment_name="baseline_global_enhanced_image",
            mode="global"
        )
        all_results.append(result_global)

    if cfg.run_lesion_gan_enhanced:
        result_lesion = train_single_experiment(
            cfg=cfg,
            experiment_name="baseline_lesion_gan_enhanced_image",
            mode="lesion_gan"
        )
        all_results.append(result_lesion)

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(cfg.output_root, "all_experiment_results.csv")
    results_df.to_csv(results_csv, index=False)

    # ranking
    sort_cols = ["roc_auc", "pr_auc", "f1", "accuracy"]
    existing_sort_cols = [c for c in sort_cols if c in results_df.columns]
    ranked_df = results_df.sort_values(existing_sort_cols, ascending=False).reset_index(drop=True)
    ranked_df.to_csv(os.path.join(cfg.output_root, "all_experiment_results_ranked.csv"), index=False)

    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE")
    print("=" * 90)
    print(ranked_df)

    return ranked_df


# ============================================================
# 9. OPTIONAL SANITY CHECK
# ============================================================

def sanity_check_paths(cfg: Config, max_samples: int = 20):
    print("=" * 80)
    print("SANITY CHECK: original and lesion-enhanced path availability")
    print("=" * 80)

    train_df = load_and_prepare_csv(cfg.train_csv, cfg)

    ok_original = 0
    ok_lesion = 0

    subset = train_df.head(max_samples).copy()

    for _, row in subset.iterrows():
        orig_path = row["resolved_image_path"]
        if os.path.exists(orig_path):
            ok_original += 1

        sample_id = row["sample_id"]
        candidate_paths = [
            os.path.join(cfg.lesion_enhanced_root, f"{sample_id}.png"),
            os.path.join(cfg.lesion_enhanced_root, f"{sample_id}.jpg"),
            os.path.join(cfg.lesion_enhanced_root, "images", f"{sample_id}.png"),
            os.path.join(cfg.lesion_enhanced_root, "images", f"{sample_id}.jpg"),
            os.path.join(cfg.lesion_enhanced_root, "enhanced", f"{sample_id}.png"),
            os.path.join(cfg.lesion_enhanced_root, "enhanced", f"{sample_id}.jpg"),
        ]
        found = any(os.path.exists(p) for p in candidate_paths)
        if found:
            ok_lesion += 1

    print(f"Original found in first {len(subset)} samples: {ok_original}/{len(subset)}")
    print(f"Lesion GAN found in first {len(subset)} samples: {ok_lesion}/{len(subset)}")


# ============================================================
# 10. EXECUTE
# ============================================================

if __name__ == "__main__":
    print("=" * 100)
    print("BASELINE CLASSIFIER COMPARISON")
    print("=" * 100)
    print(json.dumps(asdict(CFG), indent=2))

    sanity_check_paths(CFG, max_samples=20)
    final_df = run_all_experiments(CFG)