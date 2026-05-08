# -*- coding: utf-8 -*-
"""
paper_extras.py

Mevcut test_predictions.csv dosyalarından dört ek analiz üretir:
  1. Decision Curve Analysis (DCA) — clinical net benefit
  2. Subgroup analysis — confidence-based difficulty stratification
  3. Calibration deep-dive — reliability bins + Integrated Calibration Index (ICI)
  4. Error overlap analysis — pairwise agreement matrices

Çıktı: outputs/paper_analysis_extras/ klasörüne CSV ve PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


DOWNSTREAM_ROOT = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v8_leakfree"
OUTPUT_DIR = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/paper_analysis_extras"

INPUT_TYPES = ["original", "global", "gan_full", "gan_blended"]
MODELS = ["resnet50", "efficientnet_b0"]

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


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def load_predictions(model, input_type):
    path = os.path.join(DOWNSTREAM_ROOT, f"{model}_{input_type}",
                        "predictions", "test_predictions.csv")
    if not os.path.exists(path):
        print(f"[WARN] {path}")
        return None
    return pd.read_csv(path)


# ============================================================
# 1. DECISION CURVE ANALYSIS (DCA)
# ============================================================

def net_benefit(y_true, y_prob, threshold_prob):
    """
    Net Benefit at threshold probability p_t:
    NB = TP/n - FP/n * (p_t / (1 - p_t))
    """
    pred = (y_prob >= threshold_prob).astype(int)
    n = len(y_true)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    if threshold_prob >= 1.0:
        return 0.0
    return tp / n - (fp / n) * (threshold_prob / (1 - threshold_prob))


def treat_all_net_benefit(y_true, threshold_prob):
    """If we treat everyone as positive."""
    n = len(y_true)
    prevalence = (y_true == 1).mean()
    if threshold_prob >= 1.0:
        return 0.0
    return prevalence - (1 - prevalence) * (threshold_prob / (1 - threshold_prob))


def run_dca(model, threshold_range=np.linspace(0.05, 0.7, 30)):
    print(f"\n{'='*80}")
    print(f"DECISION CURVE ANALYSIS — {model}")
    print(f"{'='*80}")

    plt.figure(figsize=(9, 6))
    rows = []

    # Treat-all and treat-none baselines
    first_df = load_predictions(model, INPUT_TYPES[0])
    if first_df is None:
        return None
    y_true = first_df["y_true"].values.astype(int)

    nb_treat_all = [treat_all_net_benefit(y_true, p) for p in threshold_range]
    nb_treat_none = [0.0] * len(threshold_range)

    plt.plot(threshold_range, nb_treat_all, "k--", label="Treat all", alpha=0.6, linewidth=1)
    plt.plot(threshold_range, nb_treat_none, "k:", label="Treat none", alpha=0.6, linewidth=1)

    colors = {"original": "#888", "global": "#2ca02c", "gan_full": "#ff7f0e", "gan_blended": "#d62728"}

    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None: continue
        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)
        nb_curve = [net_benefit(y, p, t) for t in threshold_range]

        plt.plot(threshold_range, nb_curve, label=it, color=colors.get(it), linewidth=2)

        for t, nb in zip(threshold_range, nb_curve):
            rows.append({"model": model, "method": it, "threshold_prob": round(t, 3),
                         "net_benefit": round(nb, 5)})

    plt.xlabel("Threshold probability $p_t$")
    plt.ylabel("Net Benefit")
    plt.title(f"Decision Curve Analysis — {model}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"dca_{model}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  Saved: {out_png}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUT_DIR, f"dca_{model}.csv"), index=False)

    # Summary at clinically reasonable thresholds
    print(f"\nNet Benefit at p_t = 0.20 (screening) and p_t = 0.40 (triage):")
    for it in INPUT_TYPES:
        sub = df_out[df_out["method"] == it]
        if len(sub) == 0: continue
        nb_20 = sub.iloc[(sub["threshold_prob"] - 0.20).abs().argmin()]["net_benefit"]
        nb_40 = sub.iloc[(sub["threshold_prob"] - 0.40).abs().argmin()]["net_benefit"]
        print(f"  {it:15s}  NB(0.20)={nb_20:.4f}  NB(0.40)={nb_40:.4f}")

    return df_out


# ============================================================
# 2. SUBGROUP ANALYSIS — confidence-based difficulty stratification
# ============================================================

def run_subgroup(model, reference="original"):
    """
    Difficulty proxy = predicted probability extremity from reference model.
    Easy:   |p - 0.5| > 0.4  (very confident)
    Medium: 0.2 < |p - 0.5| <= 0.4
    Hard:   |p - 0.5| <= 0.2 (uncertain, near decision boundary)
    """
    print(f"\n{'='*80}")
    print(f"SUBGROUP (DIFFICULTY) ANALYSIS — {model}")
    print(f"Reference for difficulty: {reference}")
    print(f"{'='*80}")

    ref_df = load_predictions(model, reference)
    if ref_df is None: return None

    p_ref = ref_df["y_prob"].values.astype(float)
    confidence = np.abs(p_ref - 0.5)

    easy_mask = confidence > 0.4
    medium_mask = (confidence > 0.2) & (confidence <= 0.4)
    hard_mask = confidence <= 0.2

    print(f"  Easy (|p-0.5|>0.4):    n={easy_mask.sum()}")
    print(f"  Medium (0.2<|p-0.5|<=0.4): n={medium_mask.sum()}")
    print(f"  Hard (|p-0.5|<=0.2):   n={hard_mask.sum()}")

    rows = []
    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None: continue
        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)
        th = BEST_THRESHOLDS.get(f"{model}_{it}", 0.5)
        pred = (p >= th).astype(int)

        for name, mask in [("easy", easy_mask), ("medium", medium_mask), ("hard", hard_mask)]:
            if mask.sum() == 0: continue
            y_sub = y[mask]
            p_sub = p[mask]
            pred_sub = pred[mask]
            try:
                auc_v = roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else float("nan")
            except Exception:
                auc_v = float("nan")
            acc = (pred_sub == y_sub).mean()
            rows.append({
                "model": model,
                "method": it,
                "subgroup": name,
                "n": int(mask.sum()),
                "n_pos": int(y_sub.sum()),
                "AUC": round(auc_v, 4) if not np.isnan(auc_v) else "—",
                "Accuracy": round(acc, 4),
            })

    df_out = pd.DataFrame(rows)
    print("\n" + df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUTPUT_DIR, f"subgroup_{model}.csv"), index=False)

    # Hard subgroup spotlight
    print(f"\n[KEY INSIGHT] Hard subgroup (|p-0.5|<=0.2) — borderline cases:")
    hard = df_out[df_out["subgroup"] == "hard"]
    if len(hard) > 0:
        print(hard[["method", "n", "AUC", "Accuracy"]].to_string(index=False))

    return df_out


# ============================================================
# 3. CALIBRATION DEEP-DIVE
# ============================================================

def integrated_calibration_index(y_true, y_prob, n_bins=20):
    """
    ICI = mean absolute difference between predicted prob and observed frequency
    """
    bins = np.linspace(0, 1, n_bins + 1)
    total_diff = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins-1 else y_prob < hi)
        if mask.sum() == 0: continue
        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        total_diff += (mask.sum() / n) * abs(predicted - observed)
    return float(total_diff)


def run_calibration(model):
    print(f"\n{'='*80}")
    print(f"CALIBRATION DEEP-DIVE — {model}")
    print(f"{'='*80}")

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    rows = []
    colors = {"original": "#888", "global": "#2ca02c", "gan_full": "#ff7f0e", "gan_blended": "#d62728"}

    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None: continue
        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)

        # Reliability curve
        n_bins = 15
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, observed_freq = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (p >= lo) & (p <= hi if i == n_bins-1 else p < hi)
            if mask.sum() == 0: continue
            bin_centers.append(p[mask].mean())
            observed_freq.append(y[mask].mean())

        plt.plot(bin_centers, observed_freq, marker="o", label=it,
                 color=colors.get(it), linewidth=2, markersize=6)

        ici = integrated_calibration_index(y, p, n_bins=20)
        brier = float(((p - y) ** 2).mean())
        rows.append({
            "model": model, "method": it,
            "Brier": round(brier, 4),
            "ICI": round(ici, 4),
            "ICI_per_1000_decisions": round(1000 * ici, 1),
        })

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability Diagram — {model}")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"calibration_{model}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  Saved: {out_png}")

    df_out = pd.DataFrame(rows)
    print("\n" + df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUTPUT_DIR, f"calibration_{model}.csv"), index=False)
    return df_out


# ============================================================
# 4. ERROR OVERLAP ANALYSIS
# ============================================================

def run_error_overlap(model):
    print(f"\n{'='*80}")
    print(f"ERROR OVERLAP — {model}")
    print(f"{'='*80}")

    error_masks = {}
    for it in INPUT_TYPES:
        df = load_predictions(model, it)
        if df is None: continue
        y = df["y_true"].values.astype(int)
        p = df["y_prob"].values.astype(float)
        th = BEST_THRESHOLDS.get(f"{model}_{it}", 0.5)
        pred = (p >= th).astype(int)
        error_masks[it] = (pred != y)

    # Pairwise overlap matrix (Jaccard)
    methods = list(error_masks.keys())
    n_methods = len(methods)
    jaccard = np.zeros((n_methods, n_methods))
    overlap_counts = np.zeros((n_methods, n_methods), dtype=int)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            e1 = error_masks[m1]
            e2 = error_masks[m2]
            inter = (e1 & e2).sum()
            union = (e1 | e2).sum()
            jaccard[i, j] = inter / max(union, 1)
            overlap_counts[i, j] = inter

    df_jaccard = pd.DataFrame(jaccard, index=methods, columns=methods).round(3)
    df_counts = pd.DataFrame(overlap_counts, index=methods, columns=methods)

    print(f"\nError overlap (Jaccard similarity):")
    print(df_jaccard.to_string())
    print(f"\nCommon error counts:")
    print(df_counts.to_string())

    # Errors unique to each method
    print(f"\nUnique errors per method (errors made by this method but not by any other):")
    rows = []
    for m in methods:
        own = error_masks[m]
        others = np.zeros_like(own)
        for o in methods:
            if o != m:
                others = others | error_masks[o]
        unique = own & ~others
        rows.append({
            "model": model,
            "method": m,
            "total_errors": int(own.sum()),
            "unique_errors": int(unique.sum()),
            "shared_with_all": int((error_masks[methods[0]] & error_masks[methods[1]] &
                                     error_masks[methods[2]] & error_masks[methods[3]]).sum()),
        })
    df_unique = pd.DataFrame(rows)
    print("\n" + df_unique.to_string(index=False))

    # Save
    df_jaccard.to_csv(os.path.join(OUTPUT_DIR, f"error_jaccard_{model}.csv"))
    df_counts.to_csv(os.path.join(OUTPUT_DIR, f"error_counts_{model}.csv"))
    df_unique.to_csv(os.path.join(OUTPUT_DIR, f"error_unique_{model}.csv"), index=False)

    # Heatmap visualization
    plt.figure(figsize=(7, 6))
    plt.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(label="Jaccard similarity")
    plt.xticks(range(n_methods), methods, rotation=45, ha="right")
    plt.yticks(range(n_methods), methods)
    plt.title(f"Error Overlap (Jaccard) — {model}")
    for i in range(n_methods):
        for j in range(n_methods):
            plt.text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center",
                     color="white" if jaccard[i,j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"error_overlap_{model}.png"), dpi=200)
    plt.close()

    return df_jaccard, df_unique


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"\n{'#'*80}")
    print(f"PAPER EXTRAS ANALYSIS — Output: {OUTPUT_DIR}")
    print(f"{'#'*80}")

    for model in MODELS:
        run_dca(model)
        run_subgroup(model, reference="original")
        run_calibration(model)
        run_error_overlap(model)

    print(f"\n{'#'*80}")
    print("BİTTİ. Tüm çıktılar:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()