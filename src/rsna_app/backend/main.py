# -*- coding: utf-8 -*-
"""
backend/main.py — Leak-free deployment

Bu sürüm:
- downstream_v8_leakfree modellerini yükler
- enhanced_images_v6_label_free klasöründen blended okur
- Eğitimle uyumlu dropout=0.3 kullanır
"""

import os, io, base64, threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRIVE_BASE = "/content/drive/MyDrive/Spring Semester/seminar project"

# === LEAK-FREE PATHS ===
MODELS_DIR       = f"{DRIVE_BASE}/outputs/downstream_v8_leakfree"
DATA_DIR         = f"{DRIVE_BASE}/data"
TEST_CSV         = f"{DATA_DIR}/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"
ORIGINAL_ROOT    = f"{DATA_DIR}/preprocessed_rsna_lesion/images_png"
GAN_FULL_ROOT    = f"{DATA_DIR}/enhanced_images_v3"
GAN_BLENDED_ROOT = f"{DATA_DIR}/enhanced_images_v6_label_free"

INPUT_TYPE_TO_ROOT = {
    "original": ORIGINAL_ROOT,
    "global":   ORIGINAL_ROOT,
    "gan_full": GAN_FULL_ROOT,
    "gan_blended": GAN_BLENDED_ROOT,
}
GAN_SUBDIR = {"gan_full": "enhanced_full", "gan_blended": "enhanced_blended"}

# === MODEL CONFIG (training script ile birebir) ===
DROPOUT  = 0.3
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 1))
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(DROPOUT),
                                  nn.Linear(m.classifier[1].in_features, 1))
    return m

ARCH_BUILDERS = {
    "resnet50":         (build_resnet50,        lambda m: m.layer4[-1]),
    "efficientnet_b0":  (build_efficientnet_b0, lambda m: m.features[-1]),
}

_lock = threading.Lock()
_cache: Dict[str, dict] = {}

def discover_experiments():
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        full = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(full):
            continue
        cands = [os.path.join(full, "best.pt"),
                 os.path.join(full, "checkpoints", "best.pt")]
        ckpt = next((c for c in cands if os.path.isfile(c)), None)
        if ckpt is None:
            continue
        arch, it = None, None
        for a in ARCH_BUILDERS:
            if name.startswith(a + "_"):
                arch, it = a, name[len(a)+1:]
                break
        if arch is None or it not in INPUT_TYPE_TO_ROOT:
            continue
        out.append({"id": name, "arch": arch, "input_type": it, "ckpt_path": ckpt})
    return out

EXPERIMENTS = {e["id"]: e for e in discover_experiments()}


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, x):
        self.model.zero_grad()
        out = self.model(x)
        out.backward(torch.ones_like(out))
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (self.activations * w).sum(dim=1).squeeze()
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def load_experiment(exp_id):
    with _lock:
        if exp_id in _cache:
            return _cache[exp_id]
        if exp_id not in EXPERIMENTS:
            raise HTTPException(404, f"Bilinmeyen model: {exp_id}")
        exp = EXPERIMENTS[exp_id]
        builder, target_fn = ARCH_BUILDERS[exp["arch"]]
        m = builder()
        ckpt = torch.load(exp["ckpt_path"], map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        _cache[exp_id] = {**exp, "model": m, "cam": GradCAM(m, target_fn(m))}
        print(f"[INFO] Loaded model: {exp_id}")
        return _cache[exp_id]


def apply_global(g):
    """Eğitimdeki global pipeline ile birebir aynı: CLAHE + unsharp(0.5, sigma=1.0)."""
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    o = c.apply(g)
    b = cv2.GaussianBlur(o, (0, 0), 1.0)
    return np.clip(cv2.addWeighted(o, 1.5, b, -0.5, 0), 0, 255).astype(np.uint8)


def resolve_image_path(sid, split, it, raw):
    if it in ("original", "global"):
        for c in [raw, f"{ORIGINAL_ROOT}/{sid}.png",
                  f"{ORIGINAL_ROOT}/{split}/{sid}.png"]:
            if c and os.path.exists(c):
                return c
    else:
        c = f"{INPUT_TYPE_TO_ROOT[it]}/{split}/{GAN_SUBDIR[it]}/{sid}.png"
        if os.path.exists(c):
            return c
    raise HTTPException(404, f"Görüntü bulunamadı: {sid}/{it}/{split}")


def load_input_for_model(sid, split, it, raw):
    p = resolve_image_path(sid, split, it, raw)
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise HTTPException(500, f"Okunamadı: {p}")
    if it == "global":
        g = apply_global(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), p


_test_df = None
def get_test_df():
    global _test_df
    if _test_df is not None:
        return _test_df
    if not os.path.exists(TEST_CSV):
        _test_df = pd.DataFrame()
        return _test_df
    df = pd.read_csv(TEST_CSV)
    idc = next((c for c in ["patientId","sample_id","id","patient_id"] if c in df.columns), None)
    lc  = next((c for c in ["Target","target","label","class"] if c in df.columns), None)
    pc  = next((c for c in ["image_path_png","path","image_path"] if c in df.columns), None)
    df = df.rename(columns={idc:"sample_id", lc:"label"})
    df["raw_path"]  = df[pc] if pc else None
    df["split"]     = df.get("split_name", "test")
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"]     = df["label"].astype(int)
    _test_df = df[["sample_id","label","raw_path","split"]].reset_index(drop=True)
    return _test_df


def to_b64(rgb):
    _, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_one(exp_id, img_rgb):
    s = load_experiment(exp_id)
    x = preprocess(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(s["model"](x)).item()
    xc = x.clone().detach().requires_grad_(True)
    h = s["cam"].generate(xc)
    h = cv2.resize(h, (IMG_SIZE, IMG_SIZE))
    hc = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET),
                      cv2.COLOR_BGR2RGB)
    iv = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    ov = cv2.addWeighted(iv, 0.6, hc, 0.4, 0)
    return {
        "model_id": exp_id,
        "arch": s["arch"],
        "input_type": s["input_type"],
        "probability": float(prob),
        "prediction": "Lesion (Pozitif)" if prob > 0.5 else "Normal (Negatif)",
        "image_b64":   to_b64(iv),
        "heatmap_b64": to_b64(hc),
        "overlay_b64": to_b64(ov),
    }


@asynccontextmanager
async def lifespan(app):
    print(f"[INFO] device={DEVICE}  models={list(EXPERIMENTS.keys())}  "
          f"csv_exists={os.path.exists(TEST_CSV)}")
    yield


app = FastAPI(lifespan=lifespan, title="RSNA Leak-Free Deploy")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"device": str(DEVICE),
            "models": list(EXPERIMENTS.keys()),
            "test_csv": os.path.exists(TEST_CSV)}


@app.get("/models")
def list_models():
    return [{"id": e["id"], "arch": e["arch"], "input_type": e["input_type"]}
            for e in EXPERIMENTS.values()]


@app.get("/samples")
def list_samples(label: Optional[int] = None, limit: int = 50):
    df = get_test_df()
    if df.empty:
        return []
    if label in (0, 1):
        df = df[df["label"] == label]
    return df.head(limit).to_dict(orient="records")


class PredictRequest(BaseModel):
    sample_id: str
    model_ids: List[str]


@app.post("/predict_sample")
def predict_sample(req: PredictRequest):
    df = get_test_df()
    if df.empty:
        raise HTTPException(500, "Test CSV bulunamadı")
    row = df[df["sample_id"] == req.sample_id]
    if row.empty:
        raise HTTPException(404, f"Sample yok: {req.sample_id}")
    row = row.iloc[0]
    raw   = row["raw_path"] if pd.notna(row.get("raw_path")) else None
    split = str(row.get("split", "test"))
    res = []
    for mid in req.model_ids:
        if mid not in EXPERIMENTS:
            res.append({"model_id": mid, "error": "model not found"})
            continue
        try:
            img, used = load_input_for_model(
                req.sample_id, split, EXPERIMENTS[mid]["input_type"], raw)
            r = run_one(mid, img)
            r["used_path"] = used
            res.append(r)
        except HTTPException as e:
            res.append({"model_id": mid, "error": e.detail})
        except Exception as e:
            res.append({"model_id": mid, "error": str(e)})
    return {"sample_id": req.sample_id, "true_label": int(row["label"]), "results": res}


@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...), model_ids: str = Form(...)):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    g = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise HTTPException(400, "Geçersiz görüntü")
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    res = []
    for mid in ids:
        if mid not in EXPERIMENTS:
            res.append({"model_id": mid, "error": "model not found"})
            continue
        it = EXPERIMENTS[mid]["input_type"]
        img_in = cv2.cvtColor(apply_global(g), cv2.COLOR_GRAY2RGB) if it == "global" else rgb
        try:
            r = run_one(mid, img_in)
            if it in ("gan_full", "gan_blended"):
                r["warning"] = "Bu model GAN-enhanced görüntü ile eğitildi; ham yüklemede distribution shift olabilir."
            res.append(r)
        except Exception as e:
            res.append({"model_id": mid, "error": str(e)})
    return {"results": res}
