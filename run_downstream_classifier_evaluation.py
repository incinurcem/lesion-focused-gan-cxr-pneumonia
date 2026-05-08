# ============================================================
# run_downstream_classifier_evaluation.py
# ------------------------------------------------------------
# Downstream classifier stage for Ph.D. Seminar Project
#
# COMPARES:
#   1) GAN Full-Image Enhanced (v3_aggressive)
#   2) GAN Lesion-Blended Enhanced (v3_aggressive)
#
# FULL SCIENTIFIC SUITE INCLUDES:
#   - ResNet50 & EfficientNet-B0 Architectures
#   - Bootstrap 95% Confidence Intervals (N=1000)
#   - Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
#   - Youden's J-Index Threshold Optimization Sweep
#   - Sensitivity, Specificity, MCC, Brier Score, NLL, F1, Accuracy
#   - Automated Visualization (ROC, PR, Calibration, Threshold Sweep)
#   - Detailed Predictions & Manifesto Exports
#
# Designed for Google Colab / A100 / PyTorch / Ph.D. Defense
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    log_loss,
    roc_curve,
    precision_recall_curve,
    auc
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import os, gc, cv2, json, time, copy, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
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
# 1. CONFIG (V3 PATHS & A100 OPTIMIZED)
# ============================================================

@dataclass
class Config:
    # --- Drive Paths ---
    train_csv: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/train_preprocessed.csv"
    val_csv: str   = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/val_preprocessed.csv"
    test_csv: str  = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"
    original_image_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/images_png"
    output_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v7"

    # --- PhD: Klasör ve Varyant Haritaları ---
    gan_roots: Dict[str, str] = None 
    gan_variant_map: Dict[str, str] = None
    positive_class_weight: Optional[float] = None

    use_alpha_injection: bool = True  # Bu özelliği aktif eder
    alpha_min: float = 0.5  # %50 GAN, %50 Orijinal doku (Zorlaştırıyoruz)
    alpha_max: float = 0.7  # Üst sınır %70 GAN olsun          # GAN ağırlığı üst sınırı (0.8 idealdir)
    
    # --- Deney Seçenekleri ---
    input_types: Tuple[str, ...] = ("gan_blended",)
    #input_types: Tuple[str, ...] = ("gan_full", "gan_blended")
    model_names: Tuple[str, ...] = ("resnet50", "efficientnet_b0")
    #model_names: Tuple[str, ...] = ("resnet50",)


    # --- Global Enhancement Parametreleri ---
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 0.5
    unsharp_sigma: float = 1.0

    # --- Hiperparametreler (A100 ve 100 Epoch Optimizasyonu) ---
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 32
    epochs: int = 100     
    lr: float = 1e-4
    weight_decay: float = 5e-4 
    dropout: float = 0.45      
    amp: bool = True
    pretrained: bool = True
    early_stopping_patience: int = 20

    # --- Scheduler Ayarları (HATA BURADAYDI - EKLENDİ! ✅) ---
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    # --- Bootstrap & Calibration ---
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 123
    ece_bins: int = 15
    default_threshold: float = 0.5
    threshold_grid_size: int = 201

    # --- Sütun Adayları ---
    # Sütun adayları (Dataset'in sütunları bulması için hayati!)
    image_col_candidates: Tuple[str, ...] = ("image_path_png", "path", "image_path")
    label_col_candidates: Tuple[str, ...] = ("target", "label", "class")
    id_col_candidates: Tuple[str, ...]    = ("patientId", "patient_id", "id")

    # --- Normalizasyon ---
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

 
    def __post_init__(self):
        self.gan_roots = {
            "gan_full": "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v3", 
            "gan_blended": "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v5_poisson"
        }
        # Ekran görüntüsü 3 (image_383a20.jpg) 'enhanced_blended' klasörünü doğruladı ✅
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
        raise ValueError(
            f"Required columns not found. Candidates={candidates}, available={list(df.columns)}"
        )
    return None


def infer_binary_label(series: pd.Series) -> pd.Series:
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
    for x in series.astype(str).str.strip().str.lower():
        if x not in mapping:
            raise ValueError(f"Unknown label value: {x}")
        out.append(mapping[x])

    return pd.Series(out, index=series.index, dtype=np.int64)


def build_path_from_id(image_id: str, root: str) -> str:
    return os.path.join(root, f"{image_id}.png")

def resolve_image_path(raw_path: Optional[str], image_id: Optional[str], root: str) -> str:
    """
    RSNA klasör yapısında resmi esnek olarak bulur. 
    Ph.D. tezi için geliştirilmiş, hata payı düşük ve v3 hiyerarşisi ile uyumlu resolver.
    """
    candidates = []

    # 1. Adım: CSV'deki ham yol (raw_path) verisini temizle ve aday listesine ekle
    if raw_path is not None and str(raw_path).strip() != "" and str(raw_path).lower() != "nan":
        raw_path = str(raw_path).strip()
        # Doğrudan tam yol
        candidates.append(raw_path)
        # Eğer yol bağılsa (relative), root ile birleştir
        if not os.path.isabs(raw_path):
            candidates.append(os.path.join(root, raw_path))
        # Sadece dosya adını alıp root altında ara (Fallback)
        candidates.append(os.path.join(root, os.path.basename(raw_path)))

    # 2. Adım: patientId (image_id) üzerinden bilinen tüm muhtemel Ph.D. alt klasörlerini tara
    if image_id is not None and str(image_id).strip() != "" and str(image_id).lower() != "nan":
        image_id = str(image_id).strip()
        # ID'nin sonunda zaten uzantı yoksa .png ekleyerek adayları oluştur
        stem = os.path.splitext(image_id)[0]
        candidates.extend([
            os.path.join(root, f"{stem}.png"),
            os.path.join(root, "images_png", f"{stem}.png"),
            os.path.join(root, "train", f"{stem}.png"),
            os.path.join(root, "val", f"{stem}.png"),
            os.path.join(root, "test", f"{stem}.png"),
        ])

    # 3. Adım: Adayları teker teker fiziksel olarak kontrol et
    # (seen kümesi ile aynı yolu iki kez kontrol etmiyoruz, performans önemli)
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

    # 4. Adım: Eğer hiçbir yerde bulunamadıysa jüriye sunulacak kadar detaylı bir hata fırlat
    raise FileNotFoundError(
        f"[KRİTİK HATA] Resim hiçbir aday yolda bulunamadı!\n"
        f"Hasta ID: {image_id}\n"
        f"CSV'deki Yol: {raw_path}\n"
        f"Kök Dizin: {root}\n"
        f"Kontrol edilen bazı yollar: {list(seen)[:3]}..."
    )

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def apply_global_enhancement(
    img: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    unsharp_amount: float = 1.0,
    unsharp_sigma: float = 1.0
) -> np.ndarray:
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

    raw_paths = df[image_col].astype(str) if image_col is not None else pd.Series([None] * len(df), index=df.index)
    image_ids = df[id_col].astype(str) if id_col is not None else pd.Series([None] * len(df), index=df.index)

    resolved_paths = []
    sample_ids = []

    for idx in df.index:
        raw_path = raw_paths.loc[idx] if image_col is not None else None
        image_id = image_ids.loc[idx] if id_col is not None else None

        resolved = resolve_image_path(
            raw_path=raw_path,
            image_id=image_id,
            root=cfg.original_image_root
        )
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
# 4. DATASET
# ============================================================

def build_transforms(image_size: int, mean, std, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Çılgın Rotasyon ve Kaydırma (Kenardaki izleri bozmak için)
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Kontrast ve Parlaklığı Sürekli Değiştir (Poisson izlerini bulanıklaştırmak için)
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            # Modelin resmin %30'unu veya %40'ını görmesini engelle (Cutout)
            # Bu onu belli bir izi değil, genel yapıyı öğrenmeye zorlar.
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.33), ratio=(0.3, 3.3), value=0),
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
        """
        v4 (Full) ve v5 (Blended) klasörleri farklı olduğu için 
        doğru root dizinini seçerek yolu döner.
        """
        root_dir = self.cfg.gan_roots[input_type]
        variant_subfolder = self.cfg.gan_variant_map[input_type]
        
        path = os.path.join(root_dir, split_name, variant_subfolder, f"{sample_id}.png")
        
        if os.path.exists(path):
            return path
        
        # Fallback for extensions
        for ext in [".jpg", ".jpeg"]:
            alt_path = path.replace(".png", ext)
            if os.path.exists(alt_path):
                return alt_path

        raise FileNotFoundError(f"[KRİTİK] GAN görüntüsü bulunamadı: {path}")

    def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            sample_id = row["sample_id"]
            label = int(row["label_bin"])
            split_name = row["split_name"]
            
            # Orijinal (ham) röntgenin yolu - Dürüstlük kaynağımız ⚓
            orig_path = row["resolved_image_path"]

            # --- 1. ORIGINAL (HAM VERİ) ---
            if self.input_type == "original":
                img_path = orig_path
                img = read_gray(img_path)

            # --- 2. GLOBAL ENHANCEMENT (KLASİK YÖNTEMLER) ---
            elif self.input_type == "global":
                img_path = orig_path
                img = read_gray(img_path)
                img = apply_global_enhancement(
                    img,
                    use_clahe=self.cfg.use_clahe,
                    clahe_clip_limit=self.cfg.clahe_clip_limit,
                    clahe_tile_grid_size=self.cfg.clahe_tile_grid_size,
                    unsharp_amount=self.cfg.unsharp_amount,
                    unsharp_sigma=self.cfg.unsharp_sigma
                )

            # --- 3. GAN FULL (v3 - DÜRÜST SAF GAN MODU) ---
            elif self.input_type == "gan_full":
                img_path = self._find_gan_path(sample_id, split_name, "gan_full")
                img = read_gray(img_path)

            # --- 4. GAN BLENDED (v5 - POISSON + ALPHA INJECTION & MOCK ARTIFACTS) ---
            elif self.input_type == "gan_blended":
                img_orig = read_gray(orig_path)
                img_path = orig_path # Varsayılan olarak orijinal yolu tutalım

                if label == 1:
                    # POZİTİF VAKALAR: Gerçek GAN lezyonunu orijinal ile harmanla
                    gan_path = self._find_gan_path(sample_id, split_name, "gan_blended")
                    img_gan = read_gray(gan_path)
                    img_path = gan_path # Rapor için GAN yolunu güncelleyelim
                    
                    if img_orig.shape != img_gan.shape:
                        img_orig = cv2.resize(
                            img_orig, 
                            (img_gan.shape[1], img_gan.shape[0]), 
                            interpolation=cv2.INTER_AREA
                        )

                    if self.cfg.use_alpha_injection:
                        alpha = random.uniform(self.cfg.alpha_min, self.cfg.alpha_max)
                        img = cv2.addWeighted(img_gan, alpha, img_orig, (1.0 - alpha), 0)
                    else:
                        img = img_gan

                else:
                    # NEGATİF VAKALAR: Sahte Yama (Mock Artifact) ekle (GAN resmini okumaya gerek yok!)
                    img = img_orig.copy()
                    if self.cfg.use_alpha_injection:
                        h, w = img.shape
                        
                        box_w, box_h = random.randint(40, 100), random.randint(40, 100)
                        x = random.randint(int(w * 0.1), int(w * 0.8) - box_w)
                        y = random.randint(int(h * 0.2), int(h * 0.7) - box_h)
                        
                        patch = img[y:y+box_h, x:x+box_w].copy()
                        patch_altered = cv2.GaussianBlur(patch, (5, 5), 0)
                        
                        alpha = random.uniform(self.cfg.alpha_min, self.cfg.alpha_max)
                        img[y:y+box_h, x:x+box_w] = cv2.addWeighted(patch_altered, alpha, patch, 1.0 - alpha, 0)

            else:
                raise ValueError(f"Bilinmeyen girdi tipi: {self.input_type}")

            # --- SON İŞLEMLER ---
            # Gri -> RGB (Model beklentisi)
            img = gray_to_rgb(img)
            
            # Transform (Normalizasyon ve Tensor dönüşümü)
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
        # PhD Dokunuşu: Dropout katmanı eklendi
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        # PhD Dokunuşu: Dropout katmanı eklendi
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    return model


# ============================================================
# 6. CALIBRATION METRICS
# ============================================================

def compute_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(y_true)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if np.sum(mask) == 0:
            continue

        bin_acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int))
        bin_conf = np.mean(y_prob[mask])
        gap = abs(bin_acc - bin_conf)

        ece += (np.sum(mask) / total) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, n_bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs = []
    confs = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if np.sum(mask) == 0:
            continue

        pred = (y_prob[mask] >= 0.5).astype(int)
        acc = np.mean(pred == y_true[mask])
        conf = np.mean(y_prob[mask])

        accs.append(acc)
        confs.append(conf)

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


# ============================================================
# 7. THRESHOLD-BASED MEDICAL METRICS
# ============================================================

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

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "youden_j": float(youden_j),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, ece_bins: int = 15):
    out = compute_confusion_stats(y_true, y_prob, threshold=threshold)

    # robust guards for rare edge cases
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    out["brier_score"] = float(np.mean((y_prob - y_true) ** 2))

    try:
        out["nll"] = float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1]))
    except Exception:
        out["nll"] = float("nan")

    ece, mce = compute_ece_mce(y_true, y_prob, n_bins=ece_bins)
    out["ece"] = ece
    out["mce"] = mce

    return out


def find_best_threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 201):
    thresholds = np.linspace(0.0, 1.0, grid_size)
    rows = []

    best_j = -999.0
    best_threshold = 0.5
    best_metrics = None

    for th in thresholds:
        stats = compute_confusion_stats(y_true, y_prob, threshold=float(th))
        rows.append(stats)

        if stats["youden_j"] > best_j:
            best_j = stats["youden_j"]
            best_threshold = float(th)
            best_metrics = stats

    threshold_df = pd.DataFrame(rows)
    return best_threshold, best_metrics, threshold_df


# ============================================================
# 8. BOOTSTRAP CI
# ============================================================

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    n_boot: int = 1000,
    seed: int = 123,
    threshold: float = 0.5,
    ece_bins: int = 15,
):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]

        if len(np.unique(yt)) < 2 and metric_name in ["roc_auc", "pr_auc"]:
            continue

        metrics = compute_global_metrics(yt, yp, threshold=threshold, ece_bins=ece_bins)
        value = metrics.get(metric_name, np.nan)
        if not np.isnan(value):
            scores.append(value)

    scores = np.array(scores, dtype=np.float64)
    if len(scores) == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(np.percentile(scores, 2.5)),
        "ci_upper": float(np.percentile(scores, 97.5)),
    }


# ============================================================
# 9. PLOTS
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
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
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
    if len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
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
# 10. DATALOADER FACTORY
# ============================================================

def create_loaders(cfg: Config, input_type: str):
    train_df = load_and_prepare_csv(cfg.train_csv, cfg, split_name="train")
    val_df   = load_and_prepare_csv(cfg.val_csv, cfg, split_name="val")
    test_df  = load_and_prepare_csv(cfg.test_csv, cfg, split_name="test")

    train_ds = RSNADownstreamDataset(train_df, cfg, input_type=input_type, train=True)
    val_ds   = RSNADownstreamDataset(val_df, cfg, input_type=input_type, train=False)
    test_ds  = RSNADownstreamDataset(test_df, cfg, input_type=input_type, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


# ============================================================
# 11. TRAIN / EVAL LOOP
# ============================================================

def get_pos_weight_from_train_df(train_df: pd.DataFrame) -> float:
    pos = float((train_df["label_bin"] == 1).sum())
    neg = float((train_df["label_bin"] == 0).sum())
    return neg / (pos + 1e-12)


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    amp: bool = True,
    scaler=None,
    train: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_ids = []
    all_paths = []

    # Validasyon sırasında gradyan hesaplamasını kapatıyoruz (PhD Standartı)
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).view(-1, 1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            # Mixed Precision (AMP) ile ileri besleme
            # Mixed Precision (AMP) ile ileri besleme
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                # Scaler kullanımı (Sadece eğitimde!)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # --- DİKKAT: BURADA BAŞKA ELSE OLMAMALI ---
            
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            labs = y.detach().cpu().numpy().reshape(-1)

            running_loss += loss.item() * x.size(0)
            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())
            all_ids.extend(batch["id"])
            
            # Path bilgisi varsa ekle (Sanity Check ve Raporlama için)
            if "path" in batch:
                all_paths.extend(batch["path"])
            else:
                all_paths.extend([None] * x.size(0))

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Global metrikleri hesapla (AUC, ECE vb.)
    epoch_metrics = compute_global_metrics(
        all_labels, all_probs,
        threshold=0.5,
        ece_bins=15
    )

    pred_df = pd.DataFrame({
        "id": all_ids,
        "path": all_paths,
        "y_true": all_labels.astype(int),
        "y_prob": all_probs,
    })

    return epoch_loss, epoch_metrics, pred_df

# ============================================================
# 12. EXPERIMENT RUNNER
# ============================================================

def train_and_evaluate_single_experiment(cfg: Config, input_type: str, model_name: str):
    exp_name = f"{model_name}_{input_type}"
    exp_dir = os.path.join(cfg.output_root, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    pred_dir = os.path.join(exp_dir, "predictions")
    plot_dir = os.path.join(exp_dir, "plots")
    report_dir = os.path.join(exp_dir, "reports")

    for p in [exp_dir, ckpt_dir, pred_dir, plot_dir, report_dir]:
        ensure_dir(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 100)

    train_loader, val_loader, test_loader, train_df, _, _ = create_loaders(cfg, input_type=input_type)
    model = build_model(model_name=model_name, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)

    if cfg.positive_class_weight is None:
        pos_weight_value = get_pos_weight_from_train_df(train_df)
    else:
        pos_weight_value = float(cfg.positive_class_weight)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    history_rows = []
    best_val_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(cfg.epochs):
        start_t = time.time()

        train_loss, train_metrics, train_pred_df = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=(cfg.amp and device.type == "cuda"),
            scaler=scaler,
            train=True,
        )

        val_loss, val_metrics, val_pred_df = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=(cfg.amp and device.type == "cuda"),
            scaler=None,
            train=False,
        )

        val_auc_for_scheduler = val_metrics["roc_auc"]
        if np.isnan(val_auc_for_scheduler):
            val_auc_for_scheduler = 0.0
        scheduler.step(val_auc_for_scheduler)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_ppv": val_metrics["ppv"],
            "val_npv": val_metrics["npv"],
            "val_brier_score": val_metrics["brier_score"],
            "val_nll": val_metrics["nll"],
            "val_ece": val_metrics["ece"],
            "val_mce": val_metrics["mce"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start_t,
        }
        history_rows.append(row)

        print(
            f"[{epoch+1:03d}/{cfg.epochs:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
        )

        pd.DataFrame(history_rows).to_csv(os.path.join(exp_dir, "history.csv"), index=False)

        current_val_auc = val_metrics["roc_auc"]
        if np.isnan(current_val_auc):
            current_val_auc = -1.0

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "config": asdict(cfg),
                "model_name": model_name,
                "input_type": input_type,
            }
            torch.save(best_state, os.path.join(ckpt_dir, "best.pt"))

            train_pred_df.to_csv(os.path.join(pred_dir, "train_predictions_best_epoch.csv"), index=False)
            val_pred_df.to_csv(os.path.join(pred_dir, "val_predictions_best_epoch.csv"), index=False)
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    best_ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss, val_metrics, val_pred_df = run_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp=(cfg.amp and device.type == "cuda"),
        scaler=None,
        train=False,
    )
    val_pred_df.to_csv(os.path.join(pred_dir, "val_predictions_final_best_model.csv"), index=False)

    val_y = val_pred_df["y_true"].values.astype(int)
    val_prob = val_pred_df["y_prob"].values.astype(float)

    best_threshold, _, threshold_df = find_best_threshold_by_youden(
        val_y, val_prob, grid_size=cfg.threshold_grid_size
    )
    threshold_df.to_csv(os.path.join(report_dir, "val_threshold_sweep.csv"), index=False)
    plot_threshold_sweep(threshold_df, os.path.join(plot_dir, "val_threshold_sweep.png"))

    test_loss, _, test_pred_df = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp=(cfg.amp and device.type == "cuda"),
        scaler=None,
        train=False,
    )
    test_pred_df.to_csv(os.path.join(pred_dir, "test_predictions.csv"), index=False)

    test_y = test_pred_df["y_true"].values.astype(int)
    test_prob = test_pred_df["y_prob"].values.astype(float)

    test_metrics_05 = compute_global_metrics(
        test_y, test_prob,
        threshold=cfg.default_threshold,
        ece_bins=cfg.ece_bins,
    )
    test_metrics_best_th = compute_global_metrics(
        test_y, test_prob,
        threshold=best_threshold,
        ece_bins=cfg.ece_bins,
    )

    cm_05 = confusion_matrix(test_y, (test_prob >= cfg.default_threshold).astype(int), labels=[0, 1])
    cm_best = confusion_matrix(test_y, (test_prob >= best_threshold).astype(int), labels=[0, 1])
    plot_confusion_matrix(cm_05, os.path.join(plot_dir, "confusion_matrix_threshold_0_5.png"))
    plot_confusion_matrix(cm_best, os.path.join(plot_dir, "confusion_matrix_best_threshold.png"))

    plot_roc_curve(test_y, test_prob, os.path.join(plot_dir, "roc_curve.png"))
    plot_pr_curve(test_y, test_prob, os.path.join(plot_dir, "pr_curve.png"))
    plot_reliability_diagram(
        test_y, test_prob,
        os.path.join(plot_dir, "reliability_diagram.png"),
        n_bins=cfg.ece_bins,
    )

    ci_metrics = {}
    for metric_name in [
        "roc_auc", "pr_auc", "accuracy", "f1", "sensitivity",
        "specificity", "ppv", "npv", "balanced_accuracy", "mcc"
    ]:
        ci_metrics[metric_name] = bootstrap_metric_ci(
            test_y,
            test_prob,
            metric_name=metric_name,
            n_boot=cfg.bootstrap_samples,
            seed=cfg.bootstrap_seed,
            threshold=best_threshold,
            ece_bins=cfg.ece_bins,
        )

    save_json(ci_metrics, os.path.join(report_dir, "bootstrap_confidence_intervals.json"))

    final_summary = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "input_type": input_type,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_best_threshold_by_youden": float(best_threshold),
        "test_default_threshold_0_5": test_metrics_05,
        "test_best_threshold_from_val": test_metrics_best_th,
        "bootstrap_ci": ci_metrics,
    }
    save_json(final_summary, os.path.join(report_dir, "final_summary.json"))

    flat_row = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "input_type": input_type,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "val_best_threshold_by_youden": float(best_threshold),

        "test_accuracy": float(test_metrics_best_th["accuracy"]),
        "test_precision": float(test_metrics_best_th["precision"]),
        "test_recall": float(test_metrics_best_th["recall"]),
        "test_f1": float(test_metrics_best_th["f1"]),
        "test_roc_auc": float(test_metrics_best_th["roc_auc"]),
        "test_pr_auc": float(test_metrics_best_th["pr_auc"]),
        "test_sensitivity": float(test_metrics_best_th["sensitivity"]),
        "test_specificity": float(test_metrics_best_th["specificity"]),
        "test_ppv": float(test_metrics_best_th["ppv"]),
        "test_npv": float(test_metrics_best_th["npv"]),
        "test_balanced_accuracy": float(test_metrics_best_th["balanced_accuracy"]),
        "test_mcc": float(test_metrics_best_th["mcc"]),
        "test_youden_j": float(test_metrics_best_th["youden_j"]),
        "test_brier_score": float(test_metrics_best_th["brier_score"]),
        "test_nll": float(test_metrics_best_th["nll"]),
        "test_ece": float(test_metrics_best_th["ece"]),
        "test_mce": float(test_metrics_best_th["mce"]),
        "tn": int(test_metrics_best_th["tn"]),
        "fp": int(test_metrics_best_th["fp"]),
        "fn": int(test_metrics_best_th["fn"]),
        "tp": int(test_metrics_best_th["tp"]),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return flat_row


# ============================================================
# 13. MASTER RUNNER
# ============================================================

def run_full_pipeline(cfg: Config):
    ensure_dir(cfg.output_root)
    save_json(asdict(cfg), os.path.join(cfg.output_root, "config.json"))

    all_rows = []

    for model_name in cfg.model_names:
        for input_type in cfg.input_types:
            row = train_and_evaluate_single_experiment(
                cfg,
                input_type=input_type,
                model_name=model_name,
            )
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(cfg.output_root, "all_results.csv"), index=False)

    ranked_df = results_df.sort_values(
        by=["test_roc_auc", "test_pr_auc", "test_f1", "test_balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    ranked_df.to_csv(os.path.join(cfg.output_root, "all_results_ranked.csv"), index=False)

    for model_name in cfg.model_names:
        sub = ranked_df[ranked_df["model_name"] == model_name].copy()
        sub.to_csv(os.path.join(cfg.output_root, f"comparison_{model_name}.csv"), index=False)

    print("\n" + "=" * 100)
    print("FINAL RANKED RESULTS")
    print("=" * 100)
    print(ranked_df)

    return ranked_df


# ============================================================
# 14. SANITY CHECK
# ============================================================

def sanity_check(cfg: Config, max_samples: int = 20):
    print("=" * 100)
    print("SANITY CHECK (Ph.D. Final Validation)")
    print("=" * 100)

    # Önce CSV'lerin fiziksel varlığını kontrol edelim
    for csv_p in [cfg.train_csv, cfg.val_csv, cfg.test_csv]:
        if not os.path.exists(csv_p):
            raise FileNotFoundError(f"Kritik Hata: CSV dosyası bulunamadı -> {csv_p}")

    # Örnek veriyi yükle
    df = load_and_prepare_csv(cfg.train_csv, cfg, split_name="train").head(max_samples).copy()

    original_ok = 0
    gan_full_ok = 0
    gan_blended_ok = 0
    bad_full = []
    bad_blended = []

    for _, row in df.iterrows():
        # 1. Orijinal Resim Kontrolü (Hata düzeltildi: Artık sayıyor)
        if os.path.exists(row["resolved_image_path"]):
            original_ok += 1

        sample_id = row["sample_id"]
        split_name = row["split_name"]

        # 2. GAN Full (v4_seamless) Kontrolü
        full_p = os.path.join(cfg.gan_roots["gan_full"], split_name, cfg.gan_variant_map["gan_full"], f"{sample_id}.png")
        if os.path.exists(full_p): 
            gan_full_ok += 1
        else: 
            bad_full.append(sample_id)

        # 3. GAN Blended (v5_poisson) Kontrolü
        blend_p = os.path.join(cfg.gan_roots["gan_blended"], split_name, cfg.gan_variant_map["gan_blended"], f"{sample_id}.png")
        if os.path.exists(blend_p): 
            gan_blended_ok += 1
        else: 
            bad_blended.append(sample_id)

    # --- RAPORLAMA ---
    print(f"Original image found      : {original_ok}/{len(df)}")
    print(f"GAN full image found      : {gan_full_ok}/{len(df)}")
    print(f"GAN blended image found   : {gan_blended_ok}/{len(df)}")
    print("-" * 50)
    print("Varyant Eşleşmeleri:")
    print(f"  > gan_full    -> {cfg.gan_roots['gan_full']}")
    print(f"  > gan_blended -> {cfg.gan_roots['gan_blended']}")
    print("-" * 50)
    print("Train CSV                 :", cfg.train_csv)
    print("Output Root               :", cfg.output_root)
    print("=" * 100)

    # Hata Detayları
    if bad_full:
        print(f"\n⚠️ Eksik gan_full (İlk 5): {bad_full[:5]}")
    if bad_blended:
        print(f"\n⚠️ Eksik gan_blended (İlk 5): {bad_blended[:5]}")

    # Eğer her şey tamamsa büyük maratona hazırız
    if original_ok == len(df) and gan_full_ok == len(df) and gan_blended_ok == len(df):
        print("✅ SANITY CHECK BAŞARILI: Veri yolları doğrulanmıştır. Maraton başlayabilir!")
    else:
        print("❌ SANITY CHECK HATALI: Lütfen eksik dosyaları kontrol et aşkım.")

# ============================================================
# 15. ENTRY
# ============================================================

if __name__ == "__main__":
    print("=" * 100)
    print("DOWNSTREAM CLASSIFIER EVALUATION")
    print("=" * 100)
    print(json.dumps(asdict(CFG), indent=2, ensure_ascii=False))

    sanity_check(CFG, max_samples=20)
    run_full_pipeline(CFG)