import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. KONFİGÜRASYON VE YOLLAR
# ==============================================================================
CHECKPOINT_PATH = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/downstream_v5_FINAL_MARATHON/resnet50_gan_blended/checkpoints/best.pt"
TEST_CSV_PATH   = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion/metadata/test_preprocessed.csv"
GAN_IMAGE_ROOT  = "/content/drive/MyDrive/Spring Semester/seminar project/data/enhanced_images_v5_poisson"
OUTPUT_DIR      = "/content/drive/MyDrive/Spring Semester/seminar project/outputs/analysis_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_DROPOUT_PASSES = 10 # Uncertainty için testin kaç kez tekrarlanacağı

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. YARDIMCI FONKSİYONLAR VE VERİ SETİ
# ==============================================================================
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"⚠️ Dosya bulunamadı: {path}")
    return img

def gray_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

class AnalysisDataset(Dataset):
    def __init__(self, csv_path, gan_root):
        self.df = pd.read_csv(csv_path)
        self.gan_root = gan_root

        self.id_col = "patientId" if "patientId" in self.df.columns else "sample_id"
        self.label_col = "Target" if "Target" in self.df.columns else "target"
        
        if self.id_col not in self.df.columns:
            for c in ["id", "patient_id"]:
                if c in self.df.columns: self.id_col = c

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])
        label = int(row[self.label_col])
        split = row["split_name"] if "split_name" in self.df.columns else "test"

        img_path = os.path.join(self.gan_root, split, "enhanced_blended", f"{sample_id}.png")

        img = read_gray(img_path)
        img_rgb = gray_to_rgb(img)
        return self.transform(img_rgb), label, sample_id, img_rgb

# ==============================================================================
# 3. GRAD-CAM MEKANİZMASI
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward(torch.ones_like(output))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        weighted_activations = self.activations * pooled_gradients

        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# ==============================================================================
# 4. DEĞERLENDİRME VE METRİK FONKSİYONLARI
# ==============================================================================
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prob_in_bin = in_bin.mean()
        if prob_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prob_in_bin
    return ece

def enable_dropout(model):
    """Monte Carlo Dropout için sadece Dropout katmanlarını eğitime açar"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_evaluation_and_uncertainty(model, dataloader):
    model.eval()
    all_labels, all_probs, all_ids = [], [], []
    
    print("🔹 Standart Çıkarım Yapılıyor...")
    with torch.no_grad():
        for inputs, labels, ids, _ in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            probs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f"🔹 Monte Carlo Dropout ile Belirsizlik (Uncertainty) Hesaplanıyor ({MC_DROPOUT_PASSES} tur)...")
    enable_dropout(model)
    mc_probs = np.zeros((MC_DROPOUT_PASSES, len(dataloader.dataset)))
    
    with torch.no_grad():
        for pass_idx in range(MC_DROPOUT_PASSES):
            idx = 0
            for inputs, _, _, _ in tqdm(dataloader, leave=False, desc=f"Pass {pass_idx+1}/{MC_DROPOUT_PASSES}"):
                inputs = inputs.to(DEVICE)
                probs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                mc_probs[pass_idx, idx:idx+len(probs)] = probs
                idx += len(probs)
                
    mc_mean = mc_probs.mean(axis=0)
    mc_std = mc_probs.std(axis=0) # Epistemic Uncertainty
    
    model.eval() # Orijinal haline döndür
    return all_labels, all_probs, all_ids, mc_mean, mc_std

# ==============================================================================
# 5. GÖRSELLEŞTİRME VE KAYDETME ADIMLARI
# ==============================================================================
def plot_classification_metrics(y_true, y_prob, output_dir):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

    # Confusion Matrix
    y_pred_class = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_calibration(y_true, y_prob, output_dir):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = calculate_ece(y_true, y_prob)
    
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Model (ECE = {ece:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()

def plot_uncertainty(y_true, y_prob, mc_std, output_dir):
    y_pred_class = (y_prob > 0.5).astype(int)
    correct_mask = (y_true == y_pred_class)
    
    plt.figure(figsize=(8,5))
    sns.histplot(mc_std[correct_mask], color='green', label='Correct Predictions', stat="density", alpha=0.5, bins=30)
    sns.histplot(mc_std[~correct_mask], color='red', label='Incorrect Predictions', stat="density", alpha=0.5, bins=30)
    plt.xlabel('Epistemic Uncertainty (Std Dev over MC Passes)')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'))
    plt.close()

def plot_gradcam_batch(dataset, indices, cam_extractor, title_prefix, save_name, output_dir):
    if len(indices) == 0: return
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    if len(indices) == 1: axes = [axes] # Handle single row
    
    for i, idx in enumerate(indices):
        img_tensor, label, s_id, raw_img = dataset[idx]
        heatmap = cam_extractor.generate_heatmap(img_tensor.unsqueeze(0).to(DEVICE))

        heatmap_res = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

        img_vis = cv2.resize(raw_img, (224, 224)) / 255.0
        overlay = cv2.addWeighted(np.float32(img_vis), 0.6, np.float32(heatmap_color), 0.4, 0)

        axes[i][0].imshow(img_vis); axes[i][0].set_title(f"{title_prefix} - ID: {s_id}"); axes[i][0].axis('off')
        axes[i][1].imshow(heatmap_res, cmap='jet'); axes[i][1].set_title("Grad-CAM Heatmap"); axes[i][1].axis('off')
        axes[i][2].imshow(overlay); axes[i][2].set_title("Odak Analizi Overlay"); axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{save_name}.png'))
    plt.close()

# ==============================================================================
# 6. ANA YÜRÜTME BLOĞU (MAIN)
# ==============================================================================
if __name__ == "__main__":
    print(f"🔄 Checkpoint yükleniyor...")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.45), nn.Linear(model.fc.in_features, 1))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)

    # Dataset ve Dataloader
    dataset = AnalysisDataset(TEST_CSV_PATH, GAN_IMAGE_ROOT)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 1-3. Çıkarım, Metrikler ve Belirsizlik Hesaplamaları
    y_true, y_prob, ids, mc_mean, mc_std = run_evaluation_and_uncertainty(model, dataloader)
    
    print("📈 Grafikler çiziliyor ve kaydediliyor...")
    plot_classification_metrics(y_true, y_prob, OUTPUT_DIR)
    plot_calibration(y_true, y_prob, OUTPUT_DIR)
    plot_uncertainty(y_true, y_prob, mc_std, OUTPUT_DIR)

    # 4. Hata Analizi ve Grad-CAM
    print("🔍 Hata Analizi (Error Analysis) için Grad-CAM haritaları çıkarılıyor...")
    y_pred_class = (y_prob > 0.5).astype(int)
    
    # İndeksleri bulalım
    tp_indices = np.where((y_true == 1) & (y_pred_class == 1))[0].tolist()
    fp_indices = np.where((y_true == 0) & (y_pred_class == 1))[0].tolist()
    fn_indices = np.where((y_true == 1) & (y_pred_class == 0))[0].tolist()

    # Çizim için rastgele seç (max 5)
    sample_tp = random.sample(tp_indices, min(5, len(tp_indices)))
    sample_fp = random.sample(fp_indices, min(5, len(fp_indices)))
    sample_fn = random.sample(fn_indices, min(5, len(fn_indices)))

    cam = GradCAM(model, model.layer4[-1])
    
    plot_gradcam_batch(dataset, sample_tp, cam, "True Positive", "gradcam_true_positives", OUTPUT_DIR)
    plot_gradcam_batch(dataset, sample_fp, cam, "False Positive", "gradcam_error_false_positives", OUTPUT_DIR)
    plot_gradcam_batch(dataset, sample_fn, cam, "False Negative", "gradcam_error_false_negatives", OUTPUT_DIR)
    
    cam.remove_hooks()

    print(f"\n✅ Tüm analizler başarıyla tamamlandı!\nSonuçlar kaydedildi: {OUTPUT_DIR}")