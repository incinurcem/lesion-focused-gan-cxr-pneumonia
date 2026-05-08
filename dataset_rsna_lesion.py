import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm import tqdm


try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise ImportError(
        "albumentations kurulu değil. Kurmak için:\n"
        "pip install albumentations"
    )


@dataclass
class RSNADatasetConfig:
    image_size: int = 256
    image_mean: float = 0.5
    image_std: float = 0.5
    soft_mask: bool = True
    soft_mask_kernel: int = 21
    use_augmentations: bool = True
    task_mode: str = "gan"
    return_original_mask: bool = True


def read_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return img


def normalize_to_01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def apply_soft_mask(mask_01: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask_blur = cv2.GaussianBlur(mask_01, (kernel_size, kernel_size), 0)
    return np.clip(mask_blur, 0.0, 1.0).astype(np.float32)


def build_train_transforms(image_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=7,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.3
        ),
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=1.0),
        ToTensorV2(transpose_mask=False),
    ])


def build_eval_transforms(image_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=1.0),
        ToTensorV2(transpose_mask=False),
    ])


class RSNALesionDataset(Dataset):
    REQUIRED_COLUMNS = [
        "patientId",
        "split",
        "target",
        "class",
        "class_index",
        "image_path_png",
        "mask_path_png",
    ]

    def __init__(
        self,
        csv_path: str,
        config: RSNADatasetConfig,
        split: str,
        transforms=None,
        filter_missing: bool = True,
    ):
        self.csv_path = csv_path
        self.config = config
        self.split = split
        self.transforms = transforms

        self.df = pd.read_csv(csv_path)

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV içinde eksik kolonlar var: {missing_cols}\n"
                f"CSV: {csv_path}"
            )

        self.df["patientId"] = self.df["patientId"].astype(str).str.strip()
        self.df["split"] = self.df["split"].astype(str).str.strip()
        self.df["class"] = self.df["class"].astype(str).str.strip()
        self.df["image_path_png"] = self.df["image_path_png"].astype(str).str.strip()
        self.df["mask_path_png"] = self.df["mask_path_png"].astype(str).str.strip()

        self.df = self.df[self.df["split"] == split].copy()


        if filter_missing:
            print(f"[INFO] Dosya kontrolü başlıyor: {split} seti ({len(self.df)} örnek)")

            valid_rows = []

            for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
                if os.path.exists(row["image_path_png"]) and os.path.exists(row["mask_path_png"]):
                    valid_rows.append(True)
                else:
                    valid_rows.append(False)

            self.df = self.df[valid_rows].copy()

        self.df = self.df.reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"Dataset boş kaldı. CSV={csv_path}, split={split}")

    def __len__(self):
        return len(self.df)

    def _prepare_mask(self, mask_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_mask_01 = normalize_to_01(mask_uint8)
        used_mask = original_mask_01.copy()

        if self.config.soft_mask:
            used_mask = apply_soft_mask(
                used_mask,
                kernel_size=self.config.soft_mask_kernel
            )

        return original_mask_01.astype(np.float32), used_mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        patient_id = row["patientId"]
        image_path = row["image_path_png"]
        mask_path = row["mask_path_png"]
        target = int(row["target"])
        class_name = row["class"]
        class_index = int(row["class_index"])

        image_uint8 = read_grayscale_image(image_path)
        mask_uint8 = read_grayscale_image(mask_path)

        image_01 = normalize_to_01(image_uint8)
        original_mask_01, used_mask_01 = self._prepare_mask(mask_uint8)

        if self.transforms is not None:
            transformed = self.transforms(image=image_01, mask=used_mask_01)
            image_tensor = transformed["image"].float()   # [1,H,W], normalized to ~[-1,1]
            mask_tensor = transformed["mask"].float()     # [H,W] or [1,H,W]
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image_01).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(used_mask_01).unsqueeze(0).float()

        if self.config.return_original_mask:
            original_mask_tensor = torch.from_numpy(original_mask_01).unsqueeze(0).float()
        else:
            original_mask_tensor = torch.empty(0)

        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "input_tensor": input_tensor,
            "target": torch.tensor(target, dtype=torch.long),
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "class_name": class_name,
            "patient_id": patient_id,
            "image_path": image_path,
            "mask_path": mask_path,
        }

        if self.config.return_original_mask:
            sample["original_mask"] = original_mask_tensor

        return sample


def build_weighted_sampler_from_dataframe(df: pd.DataFrame) -> WeightedRandomSampler:
    if "target" not in df.columns:
        raise ValueError("Sampler için dataframe içinde 'target' kolonu gerekli.")

    class_counts = df["target"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise ValueError("Weighted sampler için en az iki sınıf gerekli.")

    weights = []
    for _, row in df.iterrows():
        y = int(row["target"])
        weights.append(1.0 / class_counts[y])

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def create_rsna_datasets(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    config: RSNADatasetConfig,
):
    train_tfms = build_train_transforms(
        image_size=config.image_size,
        mean=config.image_mean,
        std=config.image_std
    ) if config.use_augmentations else build_eval_transforms(
        image_size=config.image_size,
        mean=config.image_mean,
        std=config.image_std
    )

    eval_tfms = build_eval_transforms(
        image_size=config.image_size,
        mean=config.image_mean,
        std=config.image_std
    )

    train_ds = RSNALesionDataset(
        csv_path=train_csv,
        config=config,
        split="train",
        transforms=train_tfms,
    )

    val_ds = RSNALesionDataset(
        csv_path=val_csv,
        config=config,
        split="val",
        transforms=eval_tfms,
    )

    test_ds = RSNALesionDataset(
        csv_path=test_csv,
        config=config,
        split="test",
        transforms=eval_tfms,
    )

    return train_ds, val_ds, test_ds


def create_rsna_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    config: RSNADatasetConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
):
    train_ds, val_ds, test_ds = create_rsna_datasets(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        config=config,
    )

    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler:
        train_sampler = build_weighted_sampler_from_dataframe(train_ds.df)
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def sanity_check_loader(loader, max_batches: int = 1):
    print("=" * 80)
    print("DATALOADER SANITY CHECK")
    print("=" * 80)

    for b_idx, batch in enumerate(loader):
        print(f"Batch index         : {b_idx}")
        print(f"image shape         : {batch['image'].shape}")
        print(f"mask shape          : {batch['mask'].shape}")
        print(f"input_tensor shape  : {batch['input_tensor'].shape}")
        print(f"target shape        : {batch['target'].shape}")
        print(f"class_index shape   : {batch['class_index'].shape}")
        print(f"image dtype         : {batch['image'].dtype}")
        print(f"mask dtype          : {batch['mask'].dtype}")
        print(f"image min/max       : {batch['image'].min().item():.4f} / {batch['image'].max().item():.4f}")
        print(f"mask min/max        : {batch['mask'].min().item():.4f} / {batch['mask'].max().item():.4f}")
        print(f"patient ids example : {batch['patient_id'][:4]}")
        print(f"targets example     : {batch['target'][:8].tolist()}")

        if b_idx + 1 >= max_batches:
            break

    print("=" * 80)
    print("SANITY CHECK TAMAM")
    print("=" * 80)