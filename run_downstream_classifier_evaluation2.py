# ============================================================
# run_downstream_classifier_evaluation.py
# ------------------------------------------------------------
# ROI CROP EXPERIMENT (Patch Classifier) for Ph.D. Seminar
#
# Model is FORCED to look only at the Region of Interest (ROI).
# - Positives: Cropped exactly at the bounding box.
# - Negatives: Randomly cropped healthy lung tissue.
# ============================================================

import os
import gc
import cv2
import json
import time
import copy
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, log_loss, roc_curve, precision_recall_curve, auc
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    # --- Drive Paths ---
    train_csv: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/train_preprocessed.csv"
    val_csv: str   = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/val_preprocessed.csv"
    test_csv: str  = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"
    original_image_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/images_png"
    
    # YENİ ÇIKTI KLASÖRÜ
    output_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v7_ROI_CROP_EXPERIMENT"

    gan_roots: Dict[str, str] = None 
    gan_variant_map: Dict[str, str] = None
    positive_class_weight: Optional[float] = None
    
    # --- Deney Seçenekleri ---
    input_types: Tuple[str, ...] = ("gan_blended", "gan_full", "original")
    model_names: Tuple[str, ...] = ("resnet50", "efficientnet_b0")

    # --- Global Enhancement Parametreleri ---
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 0.5
    unsharp_sigma: float = 1.0

    # --- Hiperparametreler ---
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 32
    epochs: int = 3   
    lr: float = 1e-4
    weight_decay: float = 5e-4 
    dropout: float = 0.45      
    amp: bool = True
    pretrained: bool = True
    early_stopping_patience: int = 20

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    bootstrap_samples: int = 1000
    bootstrap_seed: int = 123
    ece_bins: int = 15
    default_threshold: float = 0.5
    threshold_grid_size: int = 201

    image_col_candidates: Tuple[str, ...] = ("image_path_png", "path", "image_path")
    label_col_candidates: Tuple[str, ...] = ("target", "label", "class")
    id_col_candidates: Tuple[str, ...]    = ("patientId", "patient_id", "id")

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

    def __post_init__(self):
        self.gan_roots = {
            "gan_full": "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v3", 
            "gan_blended": "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v5_poisson"
        }
        self.gan_variant_map = {
            "gan_full": "enhanced_full",    
            "gan_blended": "enhanced_blended", 
        }

CFG = Config()

# ============================================================
# 2. IO / UTILS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def find_first_existing_column(df: pd.DataFrame, candidates: Tuple[str, ...], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Required columns not found. Candidates={candidates}, available={list(df.columns)}")
    return None

def infer_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    mapping = {"0": 0, "1": 1, "false": 0, "true": 1, "negative": 0, "positive": 1, "normal": 0, "pneumonia": 1, "no": 0, "yes": 1}
    out = []
    for x in series.astype(str).str.strip().str.lower():
        if x not in mapping:
            raise ValueError(f"Unknown label value: {x}")
        out.append(mapping[x])
    return pd.Series(out, index=series.index, dtype=np.int64)

def resolve_image_path(raw_path: Optional[str], image_id: Optional[str], root: str) -> str:
    candidates = []
    if raw_path is not None and str(raw_path).strip() != "" and str(raw_path).lower() != "nan":
        raw_path = str(raw_path).strip()
        candidates.append(raw_path)
        if not os.path.isabs(raw_path):
            candidates.append(os.path.join(root, raw_path))
        candidates.append(os.path.join(root, os.path.basename(raw_path)))

    if image_id is not None and str(image_id).strip() != "" and str(image_id).lower() != "nan":
        image_id = str(image_id).strip()
        stem = os.path.splitext(image_id)[0]
        candidates.extend([
            os.path.join(root, f"{stem}.png"),
            os.path.join(root, "images_png", f"{stem}.png"),
            os.path.join(root, "train", f"{stem}.png"),
            os.path.join(root, "val", f"{stem}.png"),
            os.path.join(root, "test", f"{stem}.png"),
        ])

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[KRİTİK HATA] Resim bulunamadı! ID: {image_id}")

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def apply_global_enhancement(img: np.ndarray, use_clahe: bool = True, clahe_clip_limit: float = 2.0, clahe_tile_grid_size: int = 8, unsharp_amount: float = 1.0, unsharp_sigma: float = 1.0) -> np.ndarray:
    out = img.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
        out = clahe.apply(out)
    if unsharp_amount > 0:
        blur = cv2.GaussianBlur(out, (0, 0), unsharp_sigma)
        out = cv2.addWeighted(out, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# 3. DATA PREP
# ============================================================

def load_and_prepare_csv(csv_path: str, cfg: Config, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    image_col = find_first_existing_column(df, cfg.image_col_candidates, required=False)
    label_col = find_first_existing_column(df, cfg.label_col_candidates, required=True)
    id_col = find_first_existing_column(df, cfg.id_col_candidates, required=False)

    df["label_bin"] = infer_binary_label(df[label_col])

    # Bounding Box Koordinatlarını Çek (Eğer yoksalar 0 atanacak)
    x_col = find_first_existing_column(df, ("x_min", "x", "bbox_x"), required=False)
    y_col = find_first_existing_column(df, ("y_min", "y", "bbox_y"), required=False)
    w_col = find_first_existing_column(df, ("width", "w", "bbox_w"), required=False)
    h_col = find_first_existing_column(df, ("height", "h", "bbox_h"), required=False)

    df["roi_x"] = df[x_col].fillna(0) if x_col else 0
    df["roi_y"] = df[y_col].fillna(0) if y_col else 0
    df["roi_w"] = df[w_col].fillna(100) if w_col else 100
    df["roi_h"] = df[h_col].fillna(100) if h_col else 100

    raw_paths = df[image_col].astype(str) if image_col is not None else pd.Series([None] * len(df), index=df.index)
    image_ids = df[id_col].astype(str) if id_col is not None else pd.Series([None] * len(df), index=df.index)

    resolved_paths = []
    sample_ids = []

    for idx in df.index:
        raw_path = raw_paths.loc[idx] if image_col is not None else None
        image_id = image_ids.loc[idx] if id_col is not None else None

        resolved = resolve_image_path(raw_path=raw_path, image_id=image_id, root=cfg.original_image_root)
        resolved_paths.append(resolved)

        if image_id is not None and str(image_id).strip() != "" and str(image_id).lower() != "nan":
            sample_ids.append(str(image_id))
        else:
            sample_ids.append(os.path.splitext(os.path.basename(resolved))[0])

    df["resolved_image_path"] = resolved_paths
    df["sample_id"] = sample_ids
    df["split_name"] = split_name

    return df

# ============================================================
# 4. DATASET (ROI CROP LOGIC EKLENDİ)
# ============================================================

def build_transforms(image_size: int, mean, std, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class RSNADownstreamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, input_type: str, train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.input_type = input_type
        self.transform = build_transforms(cfg.image_size, cfg.mean, cfg.std, train=train)

        assert input_type in ["original", "global", "gan_full", "gan_blended"]

    def __len__(self):
        return len(self.df)

    def _find_gan_path(self, sample_id: str, split_name: str, input_type: str) -> str:
        root_dir = self.cfg.gan_roots[input_type]
        variant_subfolder = self.cfg.gan_variant_map[input_type]
        
        path = os.path.join(root_dir, split_name, variant_subfolder, f"{sample_id}.png")
        if os.path.exists(path): return path
        
        for ext in [".jpg", ".jpeg"]:
            alt_path = path.replace(".png", ext)
            if os.path.exists(alt_path): return alt_path
            
        raise FileNotFoundError(f"[KRİTİK] GAN görüntüsü bulunamadı: {path}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        label = int(row["label_bin"])
        split_name = row["split_name"]
        
        orig_path = row["resolved_image_path"]

        # --- 1. İLGİLİ GÖRÜNTÜYÜ OKU ---
        if self.input_type == "original":
            img = read_gray(orig_path)
            img_path = orig_path

        elif self.input_type == "global":
            img = read_gray(orig_path)
            img_path = orig_path
            img = apply_global_enhancement(
                img,
                use_clahe=self.cfg.use_clahe,
                clahe_clip_limit=self.cfg.clahe_clip_limit,
                clahe_tile_grid_size=self.cfg.clahe_tile_grid_size,
                unsharp_amount=self.cfg.unsharp_amount,
                unsharp_sigma=self.cfg.unsharp_sigma
            )

        elif self.input_type == "gan_full":
            img_path = self._find_gan_path(sample_id, split_name, "gan_full")
            img = read_gray(img_path)

        elif self.input_type == "gan_blended":
            img_path = self._find_gan_path(sample_id, split_name, "gan_blended")
            img = read_gray(img_path)

        else:
            raise ValueError(f"Bilinmeyen girdi tipi: {self.input_type}")

        # --- 2. ROI CROP (SADECE LEZYONA/DOKUYA ODAKLANMA) ---
        h_img, w_img = img.shape
        
        if label == 1:
            # POZİTİF: CSV'den Bounding Box koordinatlarına göre kırp
            x = int(row["roi_x"])
            y = int(row["roi_y"])
            w = int(row["roi_w"])
            h = int(row["roi_h"])
            
            x = max(0, x)
            y = max(0, y)
            x_end = min(w_img, x + w)
            y_end = min(h_img, y + h)
            
            if (y_end > y) and (x_end > x):
                img = img[y:y_end, x:x_end]
        else:
            # NEGATİF: Hata almamak için güvenli rastgele kesim (Safe Random Crop)
            # Boyutları resmin %20'si ile %50'si arasında olacak şekilde dinamik ayarla
            min_dim = int(min(w_img, h_img) * 0.2)
            max_dim = int(min(w_img, h_img) * 0.5)
            
            w = random.randint(min_dim, max_dim)
            h = random.randint(min_dim, max_dim)
            
            # Koordinatların taşmamasını garantile
            max_x = max(1, w_img - w - 1)
            max_y = max(1, h_img - h - 1)
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            x_end = min(w_img, x + w)
            y_end = min(h_img, y + h)
            
            img = img[y:y_end, x:x_end]

        # --- 3. SON İŞLEMLER ---
        img = gray_to_rgb(img)
        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "id": sample_id,
            "path": img_path,
        }

# ============================================================
# 5. MODEL
# ============================================================

def build_model(model_name: str, pretrained: bool, dropout: float):
    model_name = model_name.lower()
    print(f"[INFO] Model kuruluyor: {model_name} (Dropout: {dropout})")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    return model

# ============================================================
# 6. CALIBRATION & METRICS
# ============================================================

def compute_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, total = 0.0, 0.0, len(y_true)
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= left) & (y_prob <= right) if i == n_bins - 1 else (y_prob >= left) & (y_prob < right)
        if np.sum(mask) == 0: continue
        bin_acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int))
        bin_conf = np.mean(y_prob[mask])
        gap = abs(bin_acc - bin_conf)
        ece += (np.sum(mask) / total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, n_bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs = [], []
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= left) & (y_prob <= right) if i == n_bins - 1 else (y_prob >= left) & (y_prob < right)
        if np.sum(mask) == 0: continue
        pred = (y_prob[mask] >= 0.5).astype(int)
        accs.append(np.mean(pred == y_true[mask]))
        confs.append(np.mean(y_prob[mask]))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compute_confusion_stats(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    sensitivity = recall
    specificity = tn / (tn + fp + 1e-12)
    ppv = tp / (tp + fp + 1e-12)
    npv = tn / (tn + fn + 1e-12)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    try: mcc = matthews_corrcoef(y_true, y_pred)
    except Exception: mcc = 0.0

    return {
        "threshold": float(threshold), "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "sensitivity": float(sensitivity),
        "specificity": float(specificity), "ppv": float(ppv), "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy), "mcc": float(mcc),
        "youden_j": float(sensitivity + specificity - 1.0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def compute_global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, ece_bins: int = 15):
    out = compute_confusion_stats(y_true, y_prob, threshold=threshold)
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"], out["pr_auc"] = float("nan"), float("nan")

    out["brier_score"] = float(np.mean((y_prob - y_true) ** 2))
    try: out["nll"] = float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1]))
    except Exception: out["nll"] = float("nan")

    ece, mce = compute_ece_mce(y_true, y_prob, n_bins=ece_bins)
    out["ece"], out["mce"] = ece, mce
    return out

def find_best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 201):
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_j, best_threshold, best_metrics = -999.0, 0.5, None
    rows = []
    for th in thresholds:
        stats = compute_confusion_stats(y_true, y_prob, threshold=float(th))
        rows.append(stats)
        if stats["youden_j"] > best_j:
            best_j, best_threshold, best_metrics = stats["youden_j"], float(th), stats
    return best_threshold, best_metrics, pd.DataFrame(rows)

def bootstrap_metric_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str, n_boot: int = 1000, seed: int = 123, threshold: float = 0.5, ece_bins: int = 15):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2 and metric_name in ["roc_auc", "pr_auc"]: continue
        metrics = compute_global_metrics(yt, yp, threshold=threshold, ece_bins=ece_bins)
        val = metrics.get(metric_name, np.nan)
        if not np.isnan(val): scores.append(val)
    scores = np.array(scores, dtype=np.float64)
    if len(scores) == 0: return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    return {"mean": float(np.mean(scores)), "ci_lower": float(np.percentile(scores, 2.5)), "ci_upper": float(np.percentile(scores, 97.5))}

# ============================================================
# 7. PLOTS
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Negative", "Positive"])
    plt.yticks(ticks, ["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    if len(np.unique(y_true)) < 2: return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    if len(np.unique(y_true)) < 2: return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC = {auc(recall, precision):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_threshold_sweep(threshold_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["sensitivity"], label="Sensitivity")
    plt.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="Balanced Acc")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ============================================================
# 8. TRAIN / EVAL LOOP
# ============================================================

def create_loaders(cfg: Config, input_type: str):
    train_df = load_and_prepare_csv(cfg.train_csv, cfg, split_name="train")
    val_df   = load_and_prepare_csv(cfg.val_csv, cfg, split_name="val")
    test_df  = load_and_prepare_csv(cfg.test_csv, cfg, split_name="test")

    train_ds = RSNADownstreamDataset(train_df, cfg, input_type=input_type, train=True)
    val_ds   = RSNADownstreamDataset(val_df, cfg, input_type=input_type, train=False)
    test_ds  = RSNADownstreamDataset(test_df, cfg, input_type=input_type, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, persistent_workers=(cfg.num_workers > 0))
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, persistent_workers=(cfg.num_workers > 0))

    return train_loader, val_loader, test_loader, train_df

def get_pos_weight_from_train_df(train_df: pd.DataFrame) -> float:
    pos = float((train_df["label_bin"] == 1).sum())
    neg = float((train_df["label_bin"] == 0).sum())
    return neg / (pos + 1e-12)

def run_one_epoch(model, loader, criterion, optimizer, device, amp: bool = True, scaler=None, train: bool = True):
    if train: model.train()
    else: model.eval()

    running_loss = 0.0
    all_probs, all_labels, all_ids, all_paths = [], [], [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).view(-1, 1)

            if train: optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            labs = y.detach().cpu().numpy().reshape(-1)

            running_loss += loss.item() * x.size(0)
            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())
            all_ids.extend(batch["id"])
            all_paths.extend(batch["path"])

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.int64)
    epoch_metrics = compute_global_metrics(all_labels, all_probs, threshold=0.5, ece_bins=15)
    pred_df = pd.DataFrame({"id": all_ids, "path": all_paths, "y_true": all_labels.astype(int), "y_prob": all_probs})
    return epoch_loss, epoch_metrics, pred_df

def train_and_evaluate_single_experiment(cfg: Config, input_type: str, model_name: str):
    exp_name = f"{model_name}_{input_type}"
    exp_dir = os.path.join(cfg.output_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    pred_dir = os.path.join(exp_dir, "predictions")
    plot_dir = os.path.join(exp_dir, "plots")
    report_dir = os.path.join(exp_dir, "reports")
    for p in [exp_dir, ckpt_dir, pred_dir, plot_dir, report_dir]: ensure_dir(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 100); print(f"EXPERIMENT: {exp_name}"); print("=" * 100)

    train_loader, val_loader, test_loader, train_df = create_loaders(cfg, input_type=input_type)
    model = build_model(model_name=model_name, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)

    pos_weight_value = float(cfg.positive_class_weight) if cfg.positive_class_weight else get_pos_weight_from_train_df(train_df)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    history_rows, best_val_auc, best_epoch, patience_counter = [], -1.0, -1, 0

    for epoch in range(cfg.epochs):
        start_t = time.time()
        train_loss, train_metrics, train_pred_df = run_one_epoch(model, train_loader, criterion, optimizer, device, amp=cfg.amp, scaler=scaler, train=True)
        val_loss, val_metrics, val_pred_df = run_one_epoch(model, val_loader, criterion, optimizer, device, amp=cfg.amp, scaler=None, train=False)

        val_auc_for_scheduler = val_metrics["roc_auc"] if not np.isnan(val_metrics["roc_auc"]) else 0.0
        scheduler.step(val_auc_for_scheduler)

        history_rows.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"], "train_f1": train_metrics["f1"], "train_roc_auc": train_metrics["roc_auc"], "train_pr_auc": train_metrics["pr_auc"],
            "val_accuracy": val_metrics["accuracy"], "val_f1": val_metrics["f1"], "val_roc_auc": val_metrics["roc_auc"], "val_pr_auc": val_metrics["pr_auc"],
            "val_sensitivity": val_metrics["sensitivity"], "val_specificity": val_metrics["specificity"], "val_ppv": val_metrics["ppv"], "val_npv": val_metrics["npv"],
            "val_brier_score": val_metrics["brier_score"], "val_nll": val_metrics["nll"], "val_ece": val_metrics["ece"], "val_mce": val_metrics["mce"],
            "lr": optimizer.param_groups[0]["lr"], "time_sec": time.time() - start_t,
        })
        print(f"[{epoch+1:03d}/{cfg.epochs:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_metrics['roc_auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} val_f1={val_metrics['f1']:.4f} val_sens={val_metrics['sensitivity']:.4f} val_spec={val_metrics['specificity']:.4f}")
        pd.DataFrame(history_rows).to_csv(os.path.join(exp_dir, "history.csv"), index=False)

        current_val_auc = val_metrics["roc_auc"] if not np.isnan(val_metrics["roc_auc"]) else -1.0
        if current_val_auc > best_val_auc:
            best_val_auc, best_epoch, patience_counter = current_val_auc, epoch + 1, 0
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()), "epoch": best_epoch, "best_val_auc": best_val_auc,
                "config": asdict(cfg), "model_name": model_name, "input_type": input_type,
            }, os.path.join(ckpt_dir, "best.pt"))
            train_pred_df.to_csv(os.path.join(pred_dir, "train_predictions_best_epoch.csv"), index=False)
            val_pred_df.to_csv(os.path.join(pred_dir, "val_predictions_best_epoch.csv"), index=False)
        else: patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss, val_metrics, val_pred_df = run_one_epoch(model, val_loader, criterion, optimizer, device, amp=cfg.amp, scaler=None, train=False)
    val_pred_df.to_csv(os.path.join(pred_dir, "val_predictions_final_best_model.csv"), index=False)

    best_threshold, _, threshold_df = find_best_threshold_by_youden(val_pred_df["y_true"].values, val_pred_df["y_prob"].values, grid_size=cfg.threshold_grid_size)
    threshold_df.to_csv(os.path.join(report_dir, "val_threshold_sweep.csv"), index=False)
    plot_threshold_sweep(threshold_df, os.path.join(plot_dir, "val_threshold_sweep.png"))

    test_loss, _, test_pred_df = run_one_epoch(model, test_loader, criterion, optimizer, device, amp=cfg.amp, scaler=None, train=False)
    test_pred_df.to_csv(os.path.join(pred_dir, "test_predictions.csv"), index=False)

    test_y, test_prob = test_pred_df["y_true"].values.astype(int), test_pred_df["y_prob"].values.astype(float)
    test_metrics_05 = compute_global_metrics(test_y, test_prob, threshold=cfg.default_threshold, ece_bins=cfg.ece_bins)
    test_metrics_best_th = compute_global_metrics(test_y, test_prob, threshold=best_threshold, ece_bins=cfg.ece_bins)

    plot_confusion_matrix(confusion_matrix(test_y, (test_prob >= cfg.default_threshold).astype(int), labels=[0, 1]), os.path.join(plot_dir, "confusion_matrix_threshold_0_5.png"))
    plot_confusion_matrix(confusion_matrix(test_y, (test_prob >= best_threshold).astype(int), labels=[0, 1]), os.path.join(plot_dir, "confusion_matrix_best_threshold.png"))
    plot_roc_curve(test_y, test_prob, os.path.join(plot_dir, "roc_curve.png"))
    plot_pr_curve(test_y, test_prob, os.path.join(plot_dir, "pr_curve.png"))
    plot_reliability_diagram(test_y, test_prob, os.path.join(plot_dir, "reliability_diagram.png"), n_bins=cfg.ece_bins)

    ci_metrics = {m: bootstrap_metric_ci(test_y, test_prob, metric_name=m, n_boot=cfg.bootstrap_samples, seed=cfg.bootstrap_seed, threshold=best_threshold, ece_bins=cfg.ece_bins) for m in ["roc_auc", "pr_auc", "accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "mcc"]}
    save_json(ci_metrics, os.path.join(report_dir, "bootstrap_confidence_intervals.json"))

    save_json({
        "experiment_name": exp_name, "model_name": model_name, "input_type": input_type, "best_epoch": int(best_epoch), "best_val_auc": float(best_val_auc),
        "val_best_threshold_by_youden": float(best_threshold), "test_default_threshold_0_5": test_metrics_05, "test_best_threshold_from_val": test_metrics_best_th,
        "bootstrap_ci": ci_metrics,
    }, os.path.join(report_dir, "final_summary.json"))

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "experiment_name": exp_name, "model_name": model_name, "input_type": input_type, "best_epoch": int(best_epoch), "best_val_auc": float(best_val_auc), "val_best_threshold_by_youden": float(best_threshold),
        "test_accuracy": float(test_metrics_best_th["accuracy"]), "test_precision": float(test_metrics_best_th["precision"]), "test_recall": float(test_metrics_best_th["recall"]), "test_f1": float(test_metrics_best_th["f1"]), "test_roc_auc": float(test_metrics_best_th["roc_auc"]),
        "test_pr_auc": float(test_metrics_best_th["pr_auc"]), "test_sensitivity": float(test_metrics_best_th["sensitivity"]), "test_specificity": float(test_metrics_best_th["specificity"]), "test_ppv": float(test_metrics_best_th["ppv"]), "test_npv": float(test_metrics_best_th["npv"]),
        "test_balanced_accuracy": float(test_metrics_best_th["balanced_accuracy"]), "test_mcc": float(test_metrics_best_th["mcc"]), "test_youden_j": float(test_metrics_best_th["youden_j"]), "test_brier_score": float(test_metrics_best_th["brier_score"]), "test_nll": float(test_metrics_best_th["nll"]),
        "test_ece": float(test_metrics_best_th["ece"]), "test_mce": float(test_metrics_best_th["mce"]), "tn": int(test_metrics_best_th["tn"]), "fp": int(test_metrics_best_th["fp"]), "fn": int(test_metrics_best_th["fn"]), "tp": int(test_metrics_best_th["tp"]),
    }

def run_full_pipeline(cfg: Config):
    ensure_dir(cfg.output_root)
    save_json(asdict(cfg), os.path.join(cfg.output_root, "config.json"))

    all_rows = []
    for model_name in cfg.model_names:
        for input_type in cfg.input_types:
            all_rows.append(train_and_evaluate_single_experiment(cfg, input_type=input_type, model_name=model_name))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(cfg.output_root, "all_results.csv"), index=False)
    ranked_df = results_df.sort_values(by=["test_roc_auc", "test_pr_auc", "test_f1", "test_balanced_accuracy"], ascending=False).reset_index(drop=True)
    ranked_df.to_csv(os.path.join(cfg.output_root, "all_results_ranked.csv"), index=False)

    print("\n" + "=" * 100); print("FINAL RANKED RESULTS"); print("=" * 100); print(ranked_df)
    return ranked_df

def sanity_check(cfg: Config, max_samples: int = 20):
    print("=" * 100); print("SANITY CHECK (ROI ABLATION STUDY)"); print("=" * 100)
    for csv_p in [cfg.train_csv, cfg.val_csv, cfg.test_csv]:
        if not os.path.exists(csv_p): raise FileNotFoundError(f"CSV not found: {csv_p}")
    print("✅ Paths verified. Starting Patch Classifier (ROI) Training...")

if __name__ == "__main__":
    sanity_check(CFG, max_samples=20)
    run_full_pipeline(CFG)