#h
# Generative Deep Learning Models in Medical Image Analysis

Bu proje, medikal görüntüleme alanında üretici derin öğrenme modellerinin kullanımını incelemek için hazırlanmış seminer/proje altyapısıdır. Yapı özellikle göğüs röntgeni (CXR) gibi görüntüler üzerinde şu başlıklara odaklanmak üzere tasarlanmıştır:

- GAN tabanlı görüntü üretimi / rekonstrüksiyonu
- VAE tabanlı latent temsil öğrenme
- Diffusion tabanlı daha kararlı üretim mekanizmaları
- Reconstruction residual map ile anomali bölgesi çıkarımı
- Heatmap ve overlay tabanlı görselleştirme
- Deney, checkpoint ve konfigürasyon yönetimi

---

## Projenin amacı

Bu repo'nun temel amacı şudur:

1. Üretici modellerin medikal görüntülerde nasıl çalıştığını sistematik biçimde incelemek  
2. Farklı generative model ailelerini aynı proje çatısı altında kıyaslamak  
3. Özellikle rekonstrüksiyon temelli anomali lokalizasyonu senaryoları için residual map üretmek  
4. Seminar sunumu, rapor ve deney çıktıları için düzenli bir altyapı sağlamak  

---

## Hedef kullanım senaryoları

Bu yapı şu tür projelere uygundur:

- CXR üzerinde normal/anormal ayrımı için reconstruction-based anomaly detection
- Gerçek ve üretilen görüntülerin kıyaslanması
- Bbox veya maske üstüne heatmap bindirme
- Eğitim/validasyon metriklerinin kayıt altına alınması
- FID, SSIM, PSNR, reconstruction error gibi ölçümlerin raporlanması

---

## Klasör yapısı

```text
project_root/
│
├── configs/
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── logger.py
│   └── seed.py
│
├── visualization/
│   ├── __init__.py
│   ├── figure_builder.py
│   ├── overlays.py
│   └── residual_maps.py
│
├── tests/
│   ├── test_config_files_exist.py
│   └── test_project_layout.py
│
├── README.md
└── requirements.txt