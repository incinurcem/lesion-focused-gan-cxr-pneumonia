# -*- coding: utf-8 -*-
"""
run_downstream_classifier_evaluation.py  (LEAK-FREE)

Downstream classifier — pipeline TÜM örneklere aynı şekilde uygulanır.
- if label == 1 / else dalı YOKTUR.
- Alpha injection / mock artifact YOKTUR.
- gan_blended klasörü label-bağımsız export'tan gelir (v6_label_free).
"""

import os
import gc
import cv2
import json
import time
import copy
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


# ============================================================
# 1. CONFIG
# ============================================================
@dataclass
class Config:
    train_csv: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/train_preprocessed.csv"
    val_csv:   str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/val_preprocessed.csv"
    test_csv:  str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"
    original_image_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/images_png"
    output_root: str = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v8_leakfree"

    gan_roots: Dict[str, str] = None
    gan_variant_map: Dict[str, str] = None
    positive_class_weight: Optional[float] = None

    # Hangi deneyleri çalıştıracağız
    input_types: Tuple[str, ...] = ("original", "global", "gan_full", "gan_blended")
    model_names: Tuple[str, ...] = ("resnet50", "efficientnet_b0")

    # CLAHE
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    unsharp_amount: float = 0.5
    unsharp_sigma: float = 1.0

    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 5e-4
    dropout: float = 0.3
    amp: bool = True
    pretrained: bool = True
    early_stopping_patience: int = 15

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    bootstrap_samples: int = 1000
    bootstrap_seed: int = 123
    ece_bins: int = 15
    default_threshold: float = 0.5
    threshold_grid_size: int = 201

    image_col_candidates: Tuple[str, ...] = ("image_path_png", "path", "image_path")
    label_col_candidates: Tuple[str, ...] = ("target", "label", "class")
    id_col_candidates:    Tuple[str, ...] = ("patientId", "patient_id", "id")

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        self.gan_roots = {
            "gan_full":    "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v3",
            "gan_blended": "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v6_label_free",
        }
        self.gan_variant_map = {
            "gan_full":    "enhanced_full",
            "gan_blended": "enhanced_blended",
        }

CFG = Config()


# ============================================================
# 2. UTILS
# ============================================================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def find_first_existing_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns: return c
    if required:
        raise ValueError(f"Sütun bulunamadı: {candidates} | mevcut: {list(df.columns)}")
    return None

def infer_binary_label(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    mapping = {"0":0,"1":1,"false":0,"true":1,"negative":0,"positive":1,
               "normal":0,"pneumonia":1,"no":0,"yes":1}
    out = []
    for x in series.astype(str).str.strip().str.lower():
        if x not in mapping: raise ValueError(f"Bilinmeyen etiket: {x}")
        out.append(mapping[x])
    return pd.Series(out, index=series.index, dtype=np.int64)

def resolve_image_path(raw_path, image_id, root):
    candidates = []
    if raw_path is not None and str(raw_path).strip().lower() not in ("", "nan"):
        rp = str(raw_path).strip()
        candidates.append(rp)
        if not os.path.isabs(rp): candidates.append(os.path.join(root, rp))
        candidates.append(os.path.join(root, os.path.basename(rp)))
    if image_id is not None and str(image_id).strip().lower() not in ("", "nan"):
        stem = os.path.splitext(str(image_id).strip())[0]
        candidates.extend([
            os.path.join(root, f"{stem}.png"),
            os.path.join(root, "images_png", f"{stem}.png"),
            os.path.join(root, "train", f"{stem}.png"),
            os.path.join(root, "val",   f"{stem}.png"),
            os.path.join(root, "test",  f"{stem}.png"),
        ])
    seen = set()
    for p in candidates:
        if p in seen: continue
        seen.add(p)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Görüntü bulunamadı: id={image_id} raw={raw_path} root={root}")

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    return img

def gray_to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def apply_global_enhancement(img, use_clahe=True, clahe_clip_limit=2.0,
                             clahe_tile_grid_size=8, unsharp_amount=1.0, unsharp_sigma=1.0):
    out = img.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
        out = clahe.apply(out)
    if unsharp_amount > 0:
        blur = cv2.GaussianBlur(out, (0,0), unsharp_sigma)
        out = cv2.addWeighted(out, 1.0+unsharp_amount, blur, -unsharp_amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# 3. DATA PREP
# ============================================================
def load_and_prepare_csv(csv_path, cfg, split_name):
    df = pd.read_csv(csv_path).copy()
    image_col = find_first_existing_column(df, cfg.image_col_candidates, required=False)
    label_col = find_first_existing_column(df, cfg.label_col_candidates, required=True)
    id_col    = find_first_existing_column(df, cfg.id_col_candidates,    required=False)

    df["label_bin"] = infer_binary_label(df[label_col])
    raw_paths = df[image_col].astype(str) if image_col else pd.Series([None]*len(df), index=df.index)
    image_ids = df[id_col].astype(str)    if id_col    else pd.Series([None]*len(df), index=df.index)

    resolved = []; sample_ids = []
    for idx in df.index:
        rp = raw_paths.loc[idx] if image_col else None
        ii = image_ids.loc[idx] if id_col    else None
        p = resolve_image_path(rp, ii, cfg.original_image_root)
        resolved.append(p)
        if ii is not None and str(ii).strip().lower() not in ("","nan"):
            sample_ids.append(str(ii))
        else:
            sample_ids.append(os.path.splitext(os.path.basename(p))[0])

    df["resolved_image_path"] = resolved
    df["sample_id"] = sample_ids
    df["split_name"] = split_name
    return df


# ============================================================
# 4. DATASET (LEAK-FREE)
# ============================================================
def build_transforms(image_size, mean, std, train=True):
    """
    Makul augmentation. RandomErasing ve agresif affine kaldırıldı çünkü
    leakage giderildikten sonra modeli yapay olarak engellemenin anlamı yok.
    """
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    """
    KRİTİK: Bu dataset sınıfı, label'a BAKMAZ.
    Tüm örnekler aynı pipeline'dan geçer.
    """
    def __init__(self, df, cfg, input_type, train):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.input_type = input_type
        self.transform = build_transforms(cfg.image_size, cfg.mean, cfg.std, train=train)
        assert input_type in ["original", "global", "gan_full", "gan_blended"]

    def __len__(self): return len(self.df)

    def _find_gan_path(self, sample_id, split_name, input_type):
        root = self.cfg.gan_roots[input_type]
        sub  = self.cfg.gan_variant_map[input_type]
        p = os.path.join(root, split_name, sub, f"{sample_id}.png")
        if os.path.exists(p): return p
        for ext in [".jpg", ".jpeg"]:
            alt = p.replace(".png", ext)
            if os.path.exists(alt): return alt
        raise FileNotFoundError(f"[KRİTİK] {input_type} görüntüsü yok: {p}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        label = int(row["label_bin"])
        split_name = row["split_name"]
        orig_path = row["resolved_image_path"]

        if self.input_type == "original":
            img_path = orig_path
            img = read_gray(img_path)

        elif self.input_type == "global":
            img_path = orig_path
            img = read_gray(img_path)
            img = apply_global_enhancement(
                img,
                use_clahe=self.cfg.use_clahe,
                clahe_clip_limit=self.cfg.clahe_clip_limit,
                clahe_tile_grid_size=self.cfg.clahe_tile_grid_size,
                unsharp_amount=self.cfg.unsharp_amount,
                unsharp_sigma=self.cfg.unsharp_sigma,
            )

        elif self.input_type == "gan_full":
            img_path = self._find_gan_path(sample_id, split_name, "gan_full")
            img = read_gray(img_path)

        elif self.input_type == "gan_blended":
            # ETİKETE BAKMAZ. Pos ve neg aynı klasörden okunur.
            img_path = self._find_gan_path(sample_id, split_name, "gan_blended")
            img = read_gray(img_path)

        else:
            raise ValueError(self.input_type)

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
def build_model(model_name, pretrained, dropout):
    model_name = model_name.lower()
    print(f"[INFO] Model: {model_name} (dropout={dropout})")
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    else:
        raise ValueError(model_name)
    return model


# ============================================================
# 6. METRICS
# ============================================================
def compute_ece_mce(y_true, y_prob, n_bins=15):
    edges = np.linspace(0,1,n_bins+1); ece=0.0; mce=0.0; total=len(y_true)
    for i in range(n_bins):
        lo,hi = edges[i], edges[i+1]
        mask = (y_prob>=lo) & (y_prob<=hi if i==n_bins-1 else y_prob<hi)
        if mask.sum()==0: continue
        bin_acc = np.mean(y_true[mask] == (y_prob[mask]>=0.5).astype(int))
        bin_conf = np.mean(y_prob[mask])
        gap = abs(bin_acc-bin_conf)
        ece += (mask.sum()/total)*gap; mce = max(mce,gap)
    return float(ece), float(mce)

def compute_confusion_stats(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    sens = tp/(tp+fn+1e-12); spec = tn/(tn+fp+1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "precision": float(precision_score(y_true,y_pred,zero_division=0)),
        "recall": float(recall_score(y_true,y_pred,zero_division=0)),
        "f1": float(f1_score(y_true,y_pred,zero_division=0)),
        "sensitivity": float(sens), "specificity": float(spec),
        "ppv": float(tp/(tp+fp+1e-12)), "npv": float(tn/(tn+fn+1e-12)),
        "balanced_accuracy": float((sens+spec)/2),
        "mcc": float(matthews_corrcoef(y_true,y_pred)) if len(np.unique(y_pred))>1 else 0.0,
        "youden_j": float(sens+spec-1),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }

def compute_global_metrics(y_true, y_prob, threshold=0.5, ece_bins=15):
    out = compute_confusion_stats(y_true, y_prob, threshold)
    if len(np.unique(y_true))>=2:
        out["roc_auc"] = float(roc_auc_score(y_true,y_prob))
        out["pr_auc"]  = float(average_precision_score(y_true,y_prob))
    else:
        out["roc_auc"]=float("nan"); out["pr_auc"]=float("nan")
    out["brier_score"] = float(np.mean((y_prob-y_true)**2))
    try: out["nll"] = float(log_loss(y_true, np.vstack([1-y_prob,y_prob]).T, labels=[0,1]))
    except: out["nll"]=float("nan")
    e,m = compute_ece_mce(y_true,y_prob,n_bins=ece_bins)
    out["ece"]=e; out["mce"]=m
    return out

def find_best_threshold_by_youden(y_true, y_prob, grid_size=201):
    ths = np.linspace(0,1,grid_size); rows=[]; best_j=-999; best_th=0.5; best=None
    for th in ths:
        s = compute_confusion_stats(y_true,y_prob,float(th))
        rows.append(s)
        if s["youden_j"]>best_j:
            best_j=s["youden_j"]; best_th=float(th); best=s
    return best_th, best, pd.DataFrame(rows)

def bootstrap_metric_ci(y_true, y_prob, metric_name, n_boot=1000, seed=123,
                        threshold=0.5, ece_bins=15):
    rng = np.random.default_rng(seed); n=len(y_true); scores=[]
    for _ in range(n_boot):
        idx = rng.integers(0,n,n); yt=y_true[idx]; yp=y_prob[idx]
        if len(np.unique(yt))<2 and metric_name in ["roc_auc","pr_auc"]: continue
        m = compute_global_metrics(yt,yp,threshold,ece_bins)
        v = m.get(metric_name,np.nan)
        if not np.isnan(v): scores.append(v)
    scores = np.array(scores)
    if len(scores)==0: return {"mean":np.nan,"ci_lower":np.nan,"ci_upper":np.nan}
    return {"mean":float(np.mean(scores)),
            "ci_lower":float(np.percentile(scores,2.5)),
            "ci_upper":float(np.percentile(scores,97.5))}


# ============================================================
# 7. PLOTS
# ============================================================
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(5,4)); plt.imshow(cm,interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    th = cm.max()/2 if cm.max()>0 else 0.5
    for i in range(2):
        for j in range(2):
            plt.text(j,i,format(cm[i,j],"d"),ha="center",va="center",
                     color="white" if cm[i,j]>th else "black")
    plt.tight_layout(); plt.savefig(save_path,dpi=200); plt.close()

def plot_roc_curve(y_true,y_prob,save_path):
    if len(np.unique(y_true))<2: return
    fpr,tpr,_ = roc_curve(y_true,y_prob); a = roc_auc_score(y_true,y_prob)
    plt.figure(figsize=(6,5)); plt.plot(fpr,tpr,label=f"AUC={a:.4f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC"); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=200); plt.close()

def plot_pr_curve(y_true,y_prob,save_path):
    if len(np.unique(y_true))<2: return
    p,r,_ = precision_recall_curve(y_true,y_prob); a = auc(r,p)
    plt.figure(figsize=(6,5)); plt.plot(r,p,label=f"AUC={a:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=200); plt.close()

def plot_threshold_sweep(df, save_path):
    plt.figure(figsize=(8,5))
    for col in ["sensitivity","specificity","f1","balanced_accuracy"]:
        plt.plot(df["threshold"], df[col], label=col)
    plt.xlabel("Threshold"); plt.ylabel("Metric")
    plt.title("Threshold Sweep"); plt.legend(); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=200); plt.close()

def plot_reliability_diagram(y_true,y_prob,save_path,n_bins=15):
    edges = np.linspace(0,1,n_bins+1); accs=[]; confs=[]
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask = (y_prob>=lo) & (y_prob<=hi if i==n_bins-1 else y_prob<hi)
        if mask.sum()==0: continue
        accs.append(np.mean((y_prob[mask]>=0.5).astype(int)==y_true[mask]))
        confs.append(np.mean(y_prob[mask]))
    plt.figure(figsize=(6,6)); plt.plot([0,1],[0,1],"--")
    plt.plot(confs,accs,marker="o"); plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title("Reliability"); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=200); plt.close()


# ============================================================
# 8. LOADERS
# ============================================================
def create_loaders(cfg, input_type):
    train_df = load_and_prepare_csv(cfg.train_csv, cfg, "train")
    val_df   = load_and_prepare_csv(cfg.val_csv,   cfg, "val")
    test_df  = load_and_prepare_csv(cfg.test_csv,  cfg, "test")

    train_ds = RSNADownstreamDataset(train_df, cfg, input_type, train=True)
    val_ds   = RSNADownstreamDataset(val_df,   cfg, input_type, train=False)
    test_ds  = RSNADownstreamDataset(test_df,  cfg, input_type, train=False)

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers>0))
    return mk(train_ds,True), mk(val_ds,False), mk(test_ds,False), train_df, val_df, test_df


# ============================================================
# 9. TRAIN/EVAL LOOP
# ============================================================
def get_pos_weight(train_df):
    pos = float((train_df["label_bin"]==1).sum())
    neg = float((train_df["label_bin"]==0).sum())
    return neg / (pos + 1e-12)

def run_one_epoch(model, loader, criterion, optimizer, device,
                  amp=True, scaler=None, train=True):
    model.train() if train else model.eval()
    running=0.0; all_p=[]; all_y=[]; all_id=[]; all_path=[]
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).view(-1,1)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x); loss = criterion(logits,y)
            if train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            running += loss.item()*x.size(0)
            all_p.extend(probs.tolist())
            all_y.extend(y.detach().cpu().numpy().reshape(-1).tolist())
            all_id.extend(batch["id"])
            all_path.extend(batch.get("path", [None]*x.size(0)))
    epoch_loss = running/len(loader.dataset)
    yp = np.array(all_p); yt = np.array(all_y).astype(int)
    metrics = compute_global_metrics(yt, yp, 0.5, 15)
    pred_df = pd.DataFrame({"id":all_id,"path":all_path,"y_true":yt,"y_prob":yp})
    return epoch_loss, metrics, pred_df


# ============================================================
# 10. EXPERIMENT
# ============================================================
def train_and_evaluate(cfg, input_type, model_name):
    exp = f"{model_name}_{input_type}"
    exp_dir = os.path.join(cfg.output_root, exp)
    ckpt_dir = os.path.join(exp_dir,"checkpoints"); pred_dir = os.path.join(exp_dir,"predictions")
    plot_dir = os.path.join(exp_dir,"plots"); rep_dir = os.path.join(exp_dir,"reports")
    for p in [exp_dir,ckpt_dir,pred_dir,plot_dir,rep_dir]: ensure_dir(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n"+"="*100); print(f"EXPERIMENT: {exp}"); print("="*100)

    train_loader, val_loader, test_loader, train_df, _, _ = create_loaders(cfg, input_type)
    model = build_model(model_name, cfg.pretrained, cfg.dropout).to(device)

    pw = get_pos_weight(train_df) if cfg.positive_class_weight is None else float(cfg.positive_class_weight)
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                     factor=cfg.scheduler_factor,
                                                     patience=cfg.scheduler_patience)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=="cuda"))

    history=[]; best_auc=-1; best_epoch=-1; patience=0
    for epoch in range(cfg.epochs):
        t0=time.time()
        tr_loss, tr_m, tr_pred = run_one_epoch(model, train_loader, criterion, optimizer,
                                               device, cfg.amp and device.type=="cuda",
                                               scaler, True)
        vl_loss, vl_m, vl_pred = run_one_epoch(model, val_loader, criterion, optimizer,
                                               device, cfg.amp and device.type=="cuda",
                                               None, False)
        s_auc = vl_m["roc_auc"]
        if np.isnan(s_auc): s_auc=0.0
        scheduler.step(s_auc)

        row = {"epoch":epoch+1,"train_loss":tr_loss,"val_loss":vl_loss,
               "val_auc":vl_m["roc_auc"],"val_pr_auc":vl_m["pr_auc"],
               "val_f1":vl_m["f1"],"val_sens":vl_m["sensitivity"],"val_spec":vl_m["specificity"],
               "lr":optimizer.param_groups[0]["lr"],"time":time.time()-t0}
        history.append(row)
        print(f"[{epoch+1:03d}/{cfg.epochs:03d}] tr={tr_loss:.4f} vl={vl_loss:.4f} "
              f"vAUC={vl_m['roc_auc']:.4f} vF1={vl_m['f1']:.4f}")
        pd.DataFrame(history).to_csv(os.path.join(exp_dir,"history.csv"),index=False)

        cur = vl_m["roc_auc"] if not np.isnan(vl_m["roc_auc"]) else -1.0
        if cur > best_auc:
            best_auc = cur; best_epoch = epoch+1; patience = 0
            torch.save({"model_state_dict":copy.deepcopy(model.state_dict()),
                        "epoch":best_epoch,"best_val_auc":best_auc,
                        "model_name":model_name,"input_type":input_type},
                       os.path.join(ckpt_dir,"best.pt"))
            vl_pred.to_csv(os.path.join(pred_dir,"val_predictions_best_epoch.csv"),index=False)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stop @ {epoch+1}"); break

    # Best modeli yükle
    bst = torch.load(os.path.join(ckpt_dir,"best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(bst["model_state_dict"])

    _, _, vl_pred = run_one_epoch(model, val_loader, criterion, optimizer, device,
                                  cfg.amp and device.type=="cuda", None, False)
    vy = vl_pred["y_true"].values.astype(int); vp = vl_pred["y_prob"].values.astype(float)
    best_th, _, th_df = find_best_threshold_by_youden(vy, vp, cfg.threshold_grid_size)
    th_df.to_csv(os.path.join(rep_dir,"val_threshold_sweep.csv"),index=False)
    plot_threshold_sweep(th_df, os.path.join(plot_dir,"val_threshold_sweep.png"))

    _, _, test_pred = run_one_epoch(model, test_loader, criterion, optimizer, device,
                                    cfg.amp and device.type=="cuda", None, False)
    test_pred.to_csv(os.path.join(pred_dir,"test_predictions.csv"),index=False)
    ty = test_pred["y_true"].values.astype(int); tp = test_pred["y_prob"].values.astype(float)

    m05  = compute_global_metrics(ty,tp,cfg.default_threshold,cfg.ece_bins)
    mbst = compute_global_metrics(ty,tp,best_th,cfg.ece_bins)

    plot_confusion_matrix(confusion_matrix(ty,(tp>=0.5).astype(int),labels=[0,1]),
                          os.path.join(plot_dir,"cm_0.5.png"))
    plot_confusion_matrix(confusion_matrix(ty,(tp>=best_th).astype(int),labels=[0,1]),
                          os.path.join(plot_dir,"cm_best.png"))
    plot_roc_curve(ty,tp,os.path.join(plot_dir,"roc.png"))
    plot_pr_curve(ty,tp,os.path.join(plot_dir,"pr.png"))
    plot_reliability_diagram(ty,tp,os.path.join(plot_dir,"reliability.png"),cfg.ece_bins)

    ci = {}
    for mname in ["roc_auc","pr_auc","accuracy","f1","sensitivity",
                  "specificity","ppv","npv","balanced_accuracy","mcc"]:
        ci[mname] = bootstrap_metric_ci(ty,tp,mname,cfg.bootstrap_samples,cfg.bootstrap_seed,
                                        best_th,cfg.ece_bins)
    save_json(ci, os.path.join(rep_dir,"bootstrap_ci.json"))

    summary = {
        "experiment_name":exp,"model_name":model_name,"input_type":input_type,
        "best_epoch":int(best_epoch),"best_val_auc":float(best_auc),
        "val_best_threshold_by_youden":float(best_th),
        "test_default_0.5":m05,"test_best_threshold":mbst,"bootstrap_ci":ci,
    }
    save_json(summary, os.path.join(rep_dir,"final_summary.json"))

    flat = {"experiment_name":exp,"model_name":model_name,"input_type":input_type,
            "best_epoch":int(best_epoch),"best_val_auc":float(best_auc),
            "val_best_threshold":float(best_th)}
    for k,v in mbst.items(): flat[f"test_{k}"] = v
    del model; gc.collect(); torch.cuda.empty_cache()
    return flat


def run_full_pipeline(cfg):
    ensure_dir(cfg.output_root)
    save_json(asdict(cfg), os.path.join(cfg.output_root,"config.json"))
    rows=[]
    for mn in cfg.model_names:
        for it in cfg.input_types:
            rows.append(train_and_evaluate(cfg,it,mn))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.output_root,"all_results.csv"),index=False)
    df.sort_values("test_roc_auc",ascending=False).to_csv(
        os.path.join(cfg.output_root,"all_results_ranked.csv"),index=False)
    print("\n"+"="*100); print("FINAL RESULTS"); print("="*100); print(df)
    return df


# ============================================================
# 11. SANITY CHECK
# ============================================================
def sanity_check(cfg, max_samples=20):
    print("="*100); print("SANITY CHECK"); print("="*100)
    for csv_p in [cfg.train_csv,cfg.val_csv,cfg.test_csv]:
        if not os.path.exists(csv_p): raise FileNotFoundError(csv_p)
    df = load_and_prepare_csv(cfg.train_csv,cfg,"train").head(max_samples)
    ok_orig=ok_full=ok_blend=0; bad_full=[]; bad_blend=[]
    for _, row in df.iterrows():
        if os.path.exists(row["resolved_image_path"]): ok_orig += 1
        sid = row["sample_id"]; sp = row["split_name"]
        fp = os.path.join(cfg.gan_roots["gan_full"], sp, cfg.gan_variant_map["gan_full"], f"{sid}.png")
        bp = os.path.join(cfg.gan_roots["gan_blended"], sp, cfg.gan_variant_map["gan_blended"], f"{sid}.png")
        if os.path.exists(fp): ok_full += 1
        else: bad_full.append(sid)
        if os.path.exists(bp): ok_blend += 1
        else: bad_blend.append(sid)
    print(f"original   : {ok_orig}/{len(df)}")
    print(f"gan_full   : {ok_full}/{len(df)}")
    print(f"gan_blended: {ok_blend}/{len(df)}")
    if bad_full:  print(f"⚠ eksik gan_full   (ilk 5): {bad_full[:5]}")
    if bad_blend: print(f"⚠ eksik gan_blended (ilk 5): {bad_blend[:5]}")
    if ok_orig==len(df) and ok_full==len(df) and ok_blend==len(df):
        print("✓ Sanity check OK")
    else:
        print("✗ Eksik dosyalar var, önce export et.")


if __name__ == "__main__":
    print("="*100); print("DOWNSTREAM CLASSIFIER (LEAK-FREE)"); print("="*100)
    print(json.dumps(asdict(CFG), indent=2, ensure_ascii=False))
    sanity_check(CFG, 20)
    run_full_pipeline(CFG)