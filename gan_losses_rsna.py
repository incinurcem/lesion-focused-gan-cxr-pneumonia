import torch
import torch.nn as nn
import torch.nn.functional as F

"gan_losses_rsna.py"

class GANLoss(nn.Module):
    """
    LSGAN loss (MSE tabanlı)
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target_is_real: bool):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Belirli bir maske altındaki değerlerin ortalamasını hesaplar.
    x    : [B, 1, H, W]
    mask : [B, 1, H, W]
    """
    num = torch.sum(x * mask, dim=(1, 2, 3))
    den = torch.sum(mask, dim=(1, 2, 3)).clamp_min(eps)
    return torch.mean(num / den)


def lesion_change_loss(
    enhanced: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    target_margin: float = 0.20,
):
    """
    ROI içinde modelin aktif olmasını sağlar.
    Değişim target_margin (şu an %20) altında kalırsa ağır ceza verir.
    """
    diff = torch.abs(enhanced - image)
    roi_change = masked_mean(diff, mask)
    loss = F.relu(target_margin - roi_change)
    return loss


def background_preservation_loss(enhanced: torch.Tensor, image: torch.Tensor, mask: torch.Tensor):
    """
    ROI dışındaki sağlıklı dokuların korunmasını sağlar.
    """
    bg_mask = 1.0 - mask
    diff = torch.abs(enhanced - image)
    return masked_mean(diff, bg_mask)


def identity_reconstruction_loss(enhanced: torch.Tensor, image: torch.Tensor):
    """
    Global sapmayı engeller. Agresif modda ağırlığı düşük tutulmalıdır.
    """
    return F.l1_loss(enhanced, image)


def total_variation_loss(x: torch.Tensor):
    """
    Görüntüdeki gürültüyü azaltır, pikseller arası geçişi yumuşatır.
    """
    loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss_h + loss_w


def sobel_edges(x: torch.Tensor):
    """
    Görüntüdeki kenar yapılarını (anatomi) tespit eder.
    """
    device = x.device
    channels = x.shape[1]

    kernel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1],
         [0,  0,  0],
         [1,  2,  1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    edge_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
    edge_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
    return edge


def edge_consistency_loss(enhanced: torch.Tensor, image: torch.Tensor, mask: torch.Tensor):
    """
    Sağlıklı dokudaki anatomik kenarların korunmasını sağlar.
    """
    bg_mask = 1.0 - mask
    edge_enh = sobel_edges(enhanced)
    edge_img = sobel_edges(image)
    return masked_mean(torch.abs(edge_enh - edge_img), bg_mask)


def delta_outside_roi_loss(delta: torch.Tensor, mask: torch.Tensor):
    """
    Generator'ın ROI dışında residual (delta) üretmesini engeller.
    """
    bg_mask = 1.0 - mask
    return masked_mean(torch.abs(delta), bg_mask)


def lesion_contrast_boost_loss(
    enhanced: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    ROI ile arka plan arasındaki kontrastın azalmasını cezalandırır.
    """
    bg_mask = 1.0 - mask

    enh_roi = torch.sum(enhanced * mask, dim=(1, 2, 3)) / torch.sum(mask, dim=(1, 2, 3)).clamp_min(eps)
    enh_bg = torch.sum(enhanced * bg_mask, dim=(1, 2, 3)) / torch.sum(bg_mask, dim=(1, 2, 3)).clamp_min(eps)

    img_roi = torch.sum(image * mask, dim=(1, 2, 3)) / torch.sum(mask, dim=(1, 2, 3)).clamp_min(eps)
    img_bg = torch.sum(image * bg_mask, dim=(1, 2, 3)) / torch.sum(bg_mask, dim=(1, 2, 3)).clamp_min(eps)

    enh_contrast = torch.abs(enh_roi - enh_bg)
    img_contrast = torch.abs(img_roi - img_bg)

    return torch.mean(F.relu(img_contrast - enh_contrast))


def compute_generator_loss(
    pred_fake,
    enhanced,
    image,
    mask,
    delta,
    gan_loss_fn,
    lambda_adv=1.0,
    lambda_lesion=150.0,      # AGRESİF: Model artık pasif kalamaz
    lambda_bg=2.0,           # GEVŞETİLDİ: Arka plana biraz izin veriyoruz
    lambda_id=0.1,           # GEVŞETİLDİ
    lambda_tv=0.02,
    lambda_edge=0.5,         # GEVŞETİLDİ: Anatomik katılık esnetildi
    lambda_roi_recon=0.0,
    lambda_delta_bg=2.0,     # GEVŞETİLDİ
    lesion_target_margin=0.20, # YÜKSELTİLDİ: %20 belirginlik hedefi
    lambda_contrast=10.0,     # YÜKSELTİLDİ: Görünürlük ön planda
):
    l_adv = gan_loss_fn(pred_fake, True)
    l_lesion = lesion_change_loss(
        enhanced=enhanced,
        image=image,
        mask=mask,
        target_margin=lesion_target_margin,
    )
    l_bg = background_preservation_loss(enhanced, image, mask)
    l_id = identity_reconstruction_loss(enhanced, image)
    l_tv = total_variation_loss(enhanced)
    l_edge = edge_consistency_loss(enhanced, image, mask)
    l_delta_bg = delta_outside_roi_loss(delta, mask)
    l_contrast = lesion_contrast_boost_loss(enhanced, image, mask)

    total = (
        lambda_adv * l_adv
        + lambda_lesion * l_lesion
        + lambda_bg * l_bg
        + lambda_id * l_id
        + lambda_tv * l_tv
        + lambda_edge * l_edge
        + lambda_delta_bg * l_delta_bg
        + lambda_contrast * l_contrast
    )

    return {
        "g_total": total,
        "g_adv": l_adv,
        "g_lesion": l_lesion,
        "g_bg": l_bg,
        "g_id": l_id,
        "g_tv": l_tv,
        "g_edge": l_edge,
        "g_roi_recon": torch.zeros_like(l_adv),
        "g_delta_bg": l_delta_bg,
        "g_contrast": l_contrast,
    }


def compute_discriminator_loss(pred_real, pred_fake, gan_loss_fn):
    l_real = gan_loss_fn(pred_real, True)
    l_fake = gan_loss_fn(pred_fake, False)
    total = 0.5 * (l_real + l_fake)

    return {
        "d_total": total,
        "d_real": l_real,
        "d_fake": l_fake,
    }