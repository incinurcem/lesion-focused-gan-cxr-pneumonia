import torch
import torch.nn as nn
import torch.nn.functional as F
"""
gan_model_rsna.py
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, norm=norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class LesionFocusedGenerator(nn.Module):
    """
    Mask-Guided Inference-Free Generator
    
    Eskiden: image + mask -> cat -> generator
    Şimdi  : image -> generator -> delta
    
    Artık maske bir GİRİŞ (input) değil, eğitim sırasında bir REHBER'dir.
    Model, lezyonlu bölgeleri kendi bulup parlatmayı öğrenir.
    """
    def __init__(self, in_channels=1, base_channels=64, out_channels=1, use_tanh=True):
        super().__init__()

        # in_channels artık 1 (sadece röntgen görüntüsü)
        self.down1 = DownBlock(in_channels, base_channels, norm=False)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ResidualRefineBlock(base_channels * 16),
            ResidualRefineBlock(base_channels * 16),
        )

        self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1)
        self.use_tanh = use_tanh

    def forward(self, image):
        # image girişi: [B, 1, 256, 256]
        x = image

        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        delta = self.final(x)

        if self.use_tanh:
            delta = torch.tanh(delta)

        # ARTIK MASKELİ ÇARPMA YAPMIYORUZ (Sızıntıyı önlemek için)
        # Generator, loss fonksiyonu sayesinde delta'yı sadece lezyon bölgesinde 
        # üretmeyi kendisi öğrenmiştir.
        enhanced = image + delta
        enhanced = torch.clamp(enhanced, -1.0, 1.0)

        return {
            "delta": delta,
            "enhanced": enhanced,
        }


class PatchDiscriminator(nn.Module):
    """
    Discriminator görüntünün 'gerçekçi bir iyileştirme' olup olmadığını denetler.
    Giriş: İyileştirilmiş görüntü [B, 1, H, W]
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, norm=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)