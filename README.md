<div align="center">

# 🫁 Lesion-Focused GAN Enhancement for Chest X-Ray Pneumonia Classification

**A leak-free, mask-supervised generative pipeline for pneumonia detection on the RSNA dataset**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%202018-success)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

[**Paper**](#) · [**Dataset**](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) · [**Demo**](#deployment-demo)

</div>

---

## 📋 Overview

This repository contains the implementation of a **lesion-focused, spatially-guided GAN-based image enhancement pipeline** for chest radiograph pneumonia classification. Unlike most generative medical imaging studies that focus on synthetic data augmentation or full-image enhancement, our approach **explicitly targets pathology-relevant regions** using bounding-box-derived supervision while preserving healthy anatomical structures.

### Key Contributions

- 🎯 **Lesion-focused generative enhancement** with explicit bbox supervision (rather than implicitly learned attention)
- 🔬 **Label-independent inference pipeline** — no annotation required at deployment time
- ✅ **Pipeline integrity validation** via shuffled-label sanity testing — a critical methodological safeguard rarely reported in medical imaging literature
- 📊 **Direct comparison** of lesion-focused vs. full-image GAN enhancement under matched protocols
- 🏥 **Clinically-grounded evaluation** — Decision Curve Analysis, calibration (ICI), error overlap analysis

---

## 🏗️ Project Structure

```
seminar_project/
├── scripts/                          # Core training and analysis scripts
│   ├── gan_model_rsna.py             # Lesion-focused generator + PatchGAN discriminator
│   ├── gan_losses_rsna.py            # Adversarial + lesion + edge consistency losses
│   ├── train_gan_rsna.py             # GAN training entry point
│   ├── validate_gan_rsna.py          # GAN validation (L/B ratio, edge score)
│   ├── export_blended_label_free.py  # Label-independent enhanced image export
│   ├── run_downstream_classifier_evaluation.py   # Full classifier benchmark
│   ├── leakage_sanity_check.py       # Shuffled-label integrity test
│   ├── paper_finalize.py             # Statistical tests + clinical breakdown
│   └── paper_extras.py               # DCA, calibration, error overlap, subgroup analysis
├── src/                              # Deployment app
│   ├── backend/main.py               # FastAPI inference server
│   └── frontend/app.py               # Streamlit comparison UI
├── configs/                          # Hyperparameter and path configs
├── main.py                           # Top-level entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/USERNAME/lesion-focused-gan-cxr-pneumonia.git
cd lesion-focused-gan-cxr-pneumonia
pip install -r requirements.txt
```

### 2. Dataset

Download the **RSNA Pneumonia Detection Challenge** dataset from [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and preprocess into:

```
data/preprocessed_rsna_lesion/
├── images_png/{patientId}.png
├── masks_png/{patientId}.png
└── metadata/
    ├── train_preprocessed.csv
    ├── val_preprocessed.csv
    └── test_preprocessed.csv
```

CSV columns: `patientId`, `image_path_png`, `mask_path_png`, `target`, `split_name`

### 3. Reproduce Results

```bash
# Step 1 — Train the lesion-focused GAN
python scripts/train_gan_rsna.py

# Step 2 — Export label-free blended images
python scripts/export_blended_label_free.py \
  --checkpoint outputs/gan_rsna/checkpoints/best.pt \
  --train_csv data/.../train_preprocessed.csv \
  --val_csv   data/.../val_preprocessed.csv \
  --test_csv  data/.../test_preprocessed.csv \
  --image_root data/.../images_png \
  --output_root data/enhanced_images_v6_label_free

# Step 3 — Validate pipeline integrity (CRITICAL)
python scripts/leakage_sanity_check.py

# Step 4 — Run downstream classifier benchmark (4 inputs × 2 backbones)
python scripts/run_downstream_classifier_evaluation.py

# Step 5 — Statistical and clinical analysis
python scripts/paper_finalize.py
python scripts/paper_extras.py
```

---

## 📊 Results

### Test Set Performance (n=2,668)

| Backbone        | Input         | ROC-AUC | Specificity | PPV   | Brier ↓ |
|-----------------|---------------|---------|-------------|-------|---------|
| ResNet50        | Original      | 0.892   | 0.782       | 0.527 | 0.145   |
| ResNet50        | Global (CLAHE)| 0.887   | 0.766       | 0.508 | 0.125   |
| ResNet50        | GAN-Full      | 0.872   | 0.796       | 0.528 | 0.121   |
| **ResNet50**    | **GAN-Blended** | 0.872 | **0.818**   | **0.545** | 0.137 |
| EfficientNet-B0 | Original      | 0.883   | 0.809       | 0.548 | 0.136   |
| EfficientNet-B0 | Global        | 0.889   | 0.773       | 0.524 | 0.137   |
| EfficientNet-B0 | GAN-Full      | 0.868   | 0.771       | 0.508 | 0.155   |
| EfficientNet-B0 | GAN-Blended   | 0.872   | 0.775       | 0.511 | 0.153   |

### Pipeline Integrity Validation (Shuffled-Label Sanity Test)

All four input modalities passed the leakage check (max validation AUC under randomized labels ∈ [0.484, 0.509]):

| Input          | Max Val AUC (shuffled) | Status |
|----------------|------------------------|--------|
| Original       | 0.489                  | ✓ Clean |
| Global         | 0.484                  | ✓ Clean |
| GAN-Full       | 0.509                  | ✓ Clean |
| GAN-Blended    | 0.494                  | ✓ Clean |

---

## 🧪 Key Findings

1. **Backbone-dependent operating-point shift.** Lesion-focused enhancement induces a sensitivity–specificity trade-off that is architecture-dependent: ResNet50 gains specificity (+3.5 pp) at the cost of sensitivity (−8.3 pp); EfficientNet-B0 shows the opposite pattern.

2. **Hard-case accuracy advantage.** On borderline predictions (|p − 0.5| ≤ 0.2), GAN-Blended achieves the highest classification accuracy among all methods on ResNet50 (+2.7 pp over baseline).

3. **Complementary error profiles.** Jaccard similarity of 0.56–0.65 between GAN-enhanced and baseline error sets indicates **distinct failure modes**, suggesting potential for ensemble strategies.

4. **Methodological warning.** Initial experiments produced inflated AUC > 0.96 due to label-conditional preprocessing. We identified and corrected this via shuffled-label sanity testing — a protocol we recommend as **standard practice** for medical image enhancement pipelines.

---

## 🖥️ Deployment Demo

An interactive multi-model comparison tool (FastAPI + Streamlit) is included in `src/`:

```bash
# Start backend
cd src/backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
cd src/frontend && streamlit run app.py --server.port 8501
```

Features:
- Compare predictions across 8 models (4 inputs × 2 backbones)
- Grad-CAM visualizations for each prediction
- Test-set sample browsing or custom image upload

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{YOURNAME2026LesionGAN,
  author       = {Your Name},
  title        = {Lesion-Focused GAN Enhancement for Chest X-Ray Pneumonia Classification:
                  A Leak-Free Pipeline with Backbone Sensitivity Analysis},
  year         = {2026},
  howpublished = {\url{https://github.com/USERNAME/lesion-focused-gan-cxr-pneumonia}},
  note         = {Seminar Project, [Your University]}
}
```

---

## 📚 References

1. Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*.
2. Trevethan, R. (2017). Sensitivity, specificity, and predictive values. *Frontiers in Public Health*.
3. Vickers, A. J. & Elkin, E. B. (2006). Decision curve analysis. *Medical Decision Making*.
4. Pisano, E. D. et al. (1998). CLAHE in medical imaging. *Journal of Digital Imaging*.
5. Yi, X. et al. (2019). Generative adversarial networks in medical imaging. *Medical Image Analysis*.
6. Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) file for details.

The RSNA Pneumonia Detection Challenge dataset is subject to its own [terms of use](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules).

---

## 🙏 Acknowledgments

- **Dataset:** Radiological Society of North America (RSNA), Society of Thoracic Radiology (STR), National Institutes of Health (NIH)
- **Pretrained models:** PyTorch torchvision (ImageNet weights)
- **Advisor:** [Hocan adı] — [Üniversiten]

---

<div align="center">

*This project was developed as a Spring Semester Seminar Project.*

⭐ If you find this useful, please consider starring the repository!

</div>
