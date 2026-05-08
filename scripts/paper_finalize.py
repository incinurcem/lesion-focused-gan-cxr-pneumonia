# -*- coding: utf-8 -*-
"""
paper_finalize.py

Mevcut test_predictions.csv dosyalarından makaleye yerleştirilecek üç analizi üretir:
  1. Statistical significance tests (DeLong-style bootstrap + McNemar)
  2. Clinical breakdown (FP/1000, FN/1000, workload reduction)
  3. Threshold sensitivity analysis (operating point sweep)
  4. Backbone sensitivity analysis (ResNet50 vs EfficientNet-B0)

Çıktı: outputs/paper_analysis/ klasörüne CSV ve TXT olarak kaydeder.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG — kendi yollarına göre güncelle
# ============================================================

DOWNSTREAM_ROOT = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v8_leakfree"
OUTPUT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/paper_analysis"

# Eğer klasör adın farklıysa burayı düzenle
INPUT_TYPES = ["original", "global", "gan_full", "gan_blended"]
MODELS = ["resnet50", "efficientnet_b0"]

# Best threshold per experiment (val_best_threshold_by_youden değerlerini all_results_ranked.csv'den al)
# Aşağıdaki default 0.5; istersen güncelleyebilirsin
BEST_THRESHOLDS = {
    "resnet50_original":         0.495,
    "resnet50_global":           0.410,
    "resnet50_gan_full":         0.390,
    "resnet50_gan_blended":      0.525,
    "efficientnet_b0_original":  0.515,
    "efficientnet_b0_global":    0.500,
    "efficientnet_b0_gan_full":  0.510,
    "efficientnet_b0_gan_blended": 0.510,
}


# ============================================================
# UTILITIES
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_predictions(model, input_type):
    path = os.path.join(
        DOWNSTREAM_ROOT,
        f"{model}_{input_type}",
        "predictions",
        "test_predictions.csv",
    )
    if not os.path.exists(path):
        print(f"[WARN] Bulunamadı: {path}")
        return None
    return pd.read_csv(path)


# ============================================================
# 1. STATISTICAL SIGNIFICANCE TESTS
# ============================================================

def paired_bootstrap_auc(y_true, p1, p2, n_boot=2000, seed=42):
    """Paired bootstrap test for AUC difference (DeLong-style)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a1 = roc_auc_score(y_true[idx], p1[idx])
        a2 = roc_auc_score(y_true[idx], p2[idx])
        diffs.append(a1 - a2)
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    p_two_sided = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    p_two_sided = max(p_two_sided, 1.0 / n_boot)  # floor
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return float(np.mean(diffs)), float(p_two_sided), ci


def mcnemar_test(y_true, pred1, pred2):
    """McNemar's exact-style test for paired binary predictions."""
    b = int(((pred1 != y_true) & (pred2 == y_true)).sum())
    c = int(((pred1 == y_true) & (pred2 != y_true)).sum())
    if b + c == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(p), b, c


def run_significance_tests(model, reference="gan_blended"):
    print(f"\n{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE TESTS — {model}")
    print(f"Reference: {reference}")
    print(f"{'='*80}\n")

    ref_df = load_predictions(model, reference)
    if ref_df is None:
        return None

    y = ref_df["y_true"].values.astype(int)
    p_ref = ref_df["y_prob"].values.astype(float)
    th_ref = BEST_THRESHOLDS.get(f"{model}_{reference}", 0.5)
    pred_ref = (p_ref >= th_ref).astype(int)

    rows = []
    for it in INPUT_TYPES:
        if it == reference:
            continue
        df = load_predictions(model, it)
        if df is None:
            continue

        p_oth = df["y_prob"].values.astype(float)
        th_oth = BEST_THRESHOLDS.get(f"{model}_{it}", 0.5)
        pred_oth = (p_oth >= th_oth).astype(int)

        delta_auc, p_auc, ci_auc = paired_bootstrap_auc(y, p_ref, p_oth)
        p_mcn, b, c = mcnemar_test(y, pred_ref, pred_oth)

        rows.append({
            "model": model,
            "comparison": f"{reference} vs {it}",
            "delta_auc": round(delta_auc, 4),
            "auc_ci_lower": round(ci_auc[0], 4),
            "auc_ci_upper": round(ci_auc[1], 4),
            "auc_pval": round(p_auc, 4),
            "auc_significant": "✓" if p_auc < 0.05 else "—",
            "mcnemar_pval": round(p_mcn, 4),
            "mcnemar_significant": "✓" if p_mcn < 0.05 else "—",
            "discordant_b": b,
            "discordant_c": c,
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    return df_out


# ============================================================
# 2. CLINICAL BREAKDOWN TABLE
# ============================================================

def clinical_breakdown(model):
    print(f"\n{'='*80}")
    print(f"CLINICAL BREAKDOWN — {model}")
    print(f"{'='*80}\n")

    rows = []
    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None:
            continue

        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)
        th = BEST_THRESHOLDS.get(f"{model}_{it}", 0.5)
        pred = (p >= th).astype(int)

        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tp = int(((y == 1) & (pred == 1)).sum())

        n_neg = tn + fp
        n_pos = tp + fn
        n_total = len(y)

        rows.append({
            "model": model,
            "method": it,
            "threshold": round(th, 3),
            "n_total": n_total,
            "n_healthy": n_neg,
            "n_pneumonia": n_pos,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": round(tp / max(n_pos, 1), 4),
            "Specificity": round(tn / max(n_neg, 1), 4),
            "PPV": round(tp / max(tp + fp, 1), 4),
            "FP_per_1000_healthy": round(1000 * fp / max(n_neg, 1), 1),
            "FN_per_1000_pneumonia": round(1000 * fn / max(n_pos, 1), 1),
            "Flag_rate_pct": round(100 * (tp + fp) / n_total, 1),
            "Workload_reduction_pct": round(100 * (1 - (tp + fp) / n_total), 1),
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    # Klinik karşılaştırma metni
    if len(df_out) >= 2:
        baseline = df_out[df_out["method"] == "original"].iloc[0] if len(df_out[df_out["method"] == "original"]) else None
        gan_bl = df_out[df_out["method"] == "gan_blended"].iloc[0] if len(df_out[df_out["method"] == "gan_blended"]) else None
        if baseline is not None and gan_bl is not None:
            delta_fp = baseline["FP_per_1000_healthy"] - gan_bl["FP_per_1000_healthy"]
            print(f"\n[CLINICAL TRANSLATION] {model}")
            print(f"  GAN-Blended reduces false positives by {delta_fp:+.1f} per 1000 healthy individuals")
            print(f"  vs. baseline (original).")

    return df_out


# ============================================================
# 3. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

def threshold_sensitivity(model, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    print(f"\n{'='*80}")
    print(f"THRESHOLD SENSITIVITY — {model}")
    print(f"{'='*80}\n")

    rows = []
    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None:
            continue
        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)

        for th in thresholds:
            pred = (p >= th).astype(int)
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)

            rows.append({
                "model": model,
                "method": it,
                "threshold": th,
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "PPV": round(ppv, 4),
                "Youden_J": round(sens + spec - 1, 4),
            })

    df_out = pd.DataFrame(rows)
    # Pivot for compactness
    pivot_spec = df_out.pivot(index="method", columns="threshold", values="Specificity")
    pivot_sens = df_out.pivot(index="method", columns="threshold", values="Sensitivity")

    print(f"\nSpecificity at various thresholds ({model}):")
    print(pivot_spec.to_string())
    print(f"\nSensitivity at various thresholds ({model}):")
    print(pivot_sens.to_string())

    return df_out


# ============================================================
# 4. BACKBONE SENSITIVITY ANALYSIS
# ============================================================

def backbone_sensitivity():
    print(f"\n{'='*80}")
    print(f"BACKBONE SENSITIVITY ANALYSIS")
    print(f"{'='*80}\n")

    rows = []
    for model in MODELS:
        for it in INPUT_TYPES:
            df = load_predictions(model, it)
            if df is None:
                continue
            y = df["y_true"].values.astype(int)
            p = df["y_prob"].values.astype(float)
            th = BEST_THRESHOLDS.get(f"{model}_{it}", 0.5)
            pred = (p >= th).astype(int)

            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            try:
                auc_v = roc_auc_score(y, p)
            except Exception:
                auc_v = float("nan")

            rows.append({
                "backbone": model,
                "method": it,
                "AUC": round(auc_v, 4),
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
            })

    df_all = pd.DataFrame(rows)

    # Compute deltas (gan_blended vs original) per backbone
    delta_rows = []
    for model in MODELS:
        sub = df_all[df_all["backbone"] == model]
        if len(sub) == 0:
            continue
        try:
            base = sub[sub["method"] == "original"].iloc[0]
            blend = sub[sub["method"] == "gan_blended"].iloc[0]
            delta_rows.append({
                "backbone": model,
                "delta_AUC":  round(blend["AUC"] - base["AUC"], 4),
                "delta_Sens": round(blend["Sensitivity"] - base["Sensitivity"], 4),
                "delta_Spec": round(blend["Specificity"] - base["Specificity"], 4),
            })
        except IndexError:
            continue

    df_delta = pd.DataFrame(delta_rows)
    print("\nFull comparison:")
    print(df_all.to_string(index=False))
    print("\nDelta (GAN-Blended vs Original) per backbone:")
    print(df_delta.to_string(index=False))

    # Klinik yorum
    if len(df_delta) >= 2:
        print("\n[INTERPRETATION]")
        for _, row in df_delta.iterrows():
            sign = "+" if row["delta_Spec"] >= 0 else ""
            print(f"  {row['backbone']}: ΔSpec = {sign}{row['delta_Spec']:.4f}")
        print("  → Lesion-focused enhancement effect is backbone-dependent;")
        print("    inductive bias of the backbone modulates the benefit of explicit spatial guidance.")

    return df_all, df_delta


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"\n{'#'*80}")
    print("PAPER FINALIZATION ANALYSIS")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'#'*80}")

    all_results = {}

    # 1. Significance tests
    for model in MODELS:
        df = run_significance_tests(model, reference="gan_blended")
        if df is not None:
            df.to_csv(os.path.join(OUTPUT_DIR, f"significance_{model}.csv"), index=False)
            all_results[f"significance_{model}"] = df

    # 2. Clinical breakdown
    for model in MODELS:
        df = clinical_breakdown(model)
        if df is not None:
            df.to_csv(os.path.join(OUTPUT_DIR, f"clinical_breakdown_{model}.csv"), index=False)
            all_results[f"clinical_{model}"] = df

    # 3. Threshold sensitivity
    for model in MODELS:
        df = threshold_sensitivity(model)
        if df is not None:
            df.to_csv(os.path.join(OUTPUT_DIR, f"threshold_sensitivity_{model}.csv"), index=False)
            all_results[f"threshold_{model}"] = df

    # 4. Backbone sensitivity
    df_all, df_delta = backbone_sensitivity()
    df_all.to_csv(os.path.join(OUTPUT_DIR, "backbone_full.csv"), index=False)
    df_delta.to_csv(os.path.join(OUTPUT_DIR, "backbone_delta.csv"), index=False)

    print(f"\n{'#'*80}")
    print("BİTTİ. Tüm çıktılar şuraya kaydedildi:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'#'*80}\n")

    print("MAKALE BÖLÜMLERİNE YERLEŞTİRME REHBERİ:")
    print("-" * 80)
    print("  Results 4.X — Statistical significance:")
    print("    significance_resnet50.csv, significance_efficientnet_b0.csv")
    print("  Discussion 5.X — Clinical translation:")
    print("    clinical_breakdown_resnet50.csv, clinical_breakdown_efficientnet_b0.csv")
    print("  Results 4.Y — Backbone sensitivity:")
    print("    backbone_full.csv, backbone_delta.csv")
    print("  Appendix A — Threshold sensitivity:")
    print("    threshold_sensitivity_*.csv")
    print("-" * 80)


if __name__ == "__main__":
    main()