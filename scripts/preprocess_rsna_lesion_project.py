import os
import json
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom kurulu değil. Kurmak için: pip install pydicom")


# =========================================================
# USER PATHS
# =========================================================

MASTER_CSV = "/content/drive/MyDrive/Spring Semester/dataset/processed_metadata/rsna_master_metadata.csv"
OUTPUT_ROOT = "/content/drive/MyDrive/Spring Semester/seminar project/data/preprocessed_rsna_lesion"

IMAGE_SIZE = 256
SAVE_OVERLAYS = True

# True olursa mevcut image/mask/overlay dosyalarını tekrar üretmez.
# Eksik olanları tamamlar.
SKIP_EXISTING_FILES = True

BAD_FILENAMES = {
    "c90ba168-4d65-4205-90e5-4f96d693d54a (1)",
    "c90ba168-4d65-4205-90e5-4f96d693d54a (1).dcm",
}


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def apply_mono_fix(ds, img: np.ndarray) -> np.ndarray:
    try:
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img
    except Exception:
        pass
    return img


def read_dicom_as_uint8(dicom_path: str) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img = apply_mono_fix(ds, img)
    img = minmax_normalize(img)
    return img


def resize_image(img: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def parse_bbox_list(bbox_raw):
    if bbox_raw is None or (isinstance(bbox_raw, float) and np.isnan(bbox_raw)):
        return []

    if isinstance(bbox_raw, list):
        return bbox_raw

    if isinstance(bbox_raw, str):
        s = bbox_raw.strip()
        if s == "" or s == "[]":
            return []
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return []

    return []


def build_union_mask_from_bboxes(bbox_list, orig_h: int, orig_w: int, out_size: int) -> np.ndarray:
    mask = np.zeros((out_size, out_size), dtype=np.uint8)

    if orig_h <= 0 or orig_w <= 0:
        return mask

    scale_x = out_size / float(orig_w)
    scale_y = out_size / float(orig_h)

    for bbox in bbox_list:
        x = safe_float(bbox.get("x", np.nan))
        y = safe_float(bbox.get("y", np.nan))
        w = safe_float(bbox.get("width", np.nan))
        h = safe_float(bbox.get("height", np.nan))

        if any(np.isnan(v) for v in [x, y, w, h]):
            continue
        if w <= 0 or h <= 0:
            continue

        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + w) * scale_x))
        y2 = int(round((y + h) * scale_y))

        x1 = max(0, min(out_size - 1, x1))
        y1 = max(0, min(out_size - 1, y1))
        x2 = max(0, min(out_size, x2))
        y2 = max(0, min(out_size, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        mask[y1:y2, x1:x2] = 255

    return mask


def create_overlay(image_uint8: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    overlay = image_bgr.copy()

    red_layer = np.zeros_like(image_bgr)
    red_layer[:, :, 2] = 255

    mask_bool = mask_uint8 > 0
    blended = cv2.addWeighted(image_bgr, 0.45, red_layer, 0.55, 0)
    overlay[mask_bool] = blended[mask_bool]
    return overlay


def class_to_index(class_name: str) -> int:
    mapping = {
        "Normal": 0,
        "No Lung Opacity / Not Normal": 1,
        "Lung Opacity": 2,
    }
    return mapping.get(class_name, -1)


def valid_split(split: str) -> bool:
    return split in {"train", "val", "test"}


def read_png_gray(path: str):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def infer_image_shape_from_existing_png(image_png_path: str):
    img = read_png_gray(image_png_path)
    if img is None:
        return None
    return img.shape[:2]


# =========================================================
# Main Preprocessor
# =========================================================

class RSNAPreprocessor:
    def __init__(
        self,
        master_csv: str,
        output_root: str,
        image_size: int = 256,
        save_overlays: bool = True,
        skip_existing_files: bool = True
    ):
        self.master_csv = master_csv
        self.output_root = Path(output_root)
        self.image_size = image_size
        self.save_overlays = save_overlays
        self.skip_existing_files = skip_existing_files

        self.images_root = self.output_root / "images_png"
        self.masks_root = self.output_root / "masks_png"
        self.overlays_root = self.output_root / "overlays"
        self.meta_root = self.output_root / "metadata"

        for root in [self.images_root, self.masks_root, self.meta_root]:
            ensure_dir(str(root))

        if self.save_overlays:
            ensure_dir(str(self.overlays_root))

        for split in ["train", "val", "test"]:
            ensure_dir(str(self.images_root / split))
            ensure_dir(str(self.masks_root / split))
            if self.save_overlays:
                ensure_dir(str(self.overlays_root / split))

        self.processed_rows = []
        self.failed_cases = []

    def load_master_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.master_csv):
            raise FileNotFoundError(f"Master CSV bulunamadı: {self.master_csv}")

        df = pd.read_csv(self.master_csv)

        required_cols = [
            "patientId",
            "image_path",
            "split",
            "target",
            "class",
            "has_bbox",
            "bbox_count",
            "bbox_list",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Master CSV eksik kolon içeriyor: {missing}")

        df["patientId"] = df["patientId"].astype(str).str.strip()
        df["split"] = df["split"].astype(str).str.strip()
        df["class"] = df["class"].astype(str).str.strip()
        df["image_path"] = df["image_path"].astype(str).str.strip()

        df = df[df["split"].apply(valid_split)].copy()

        df = df[~df["patientId"].isin(BAD_FILENAMES)].copy()
        df = df[~df["image_path"].apply(lambda x: any(bad in x for bad in BAD_FILENAMES))].copy()

        print(f"[INFO] image_path varlık kontrolü başlıyor ({len(df)} satır)")
        valid_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking image paths"):
            valid_rows.append(os.path.exists(row["image_path"]))
        df = df[valid_rows].copy()

        if df["patientId"].duplicated().any():
            dup_ids = df.loc[df["patientId"].duplicated(), "patientId"].tolist()[:10]
            raise ValueError(f"Master CSV içinde duplicate patientId var. İlk örnekler: {dup_ids}")

        df = df.reset_index(drop=True)

        print("\n" + "=" * 90)
        print("MASTER CSV HAZIR")
        print("=" * 90)
        print(f"Toplam satır: {len(df)}")
        print("\nSplit dağılımı:")
        print(df["split"].value_counts())
        print("\nTarget dağılımı:")
        print(df["target"].value_counts())

        return df

    def process_one_row(self, row: pd.Series):
        patient_id = str(row["patientId"]).strip()
        image_path = row["image_path"]
        split = row["split"]
        target = int(row["target"])
        class_name = row["class"]
        has_bbox = int(row["has_bbox"])
        bbox_count = int(row["bbox_count"])
        bbox_list = parse_bbox_list(row["bbox_list"])

        image_out_path = str(self.images_root / split / f"{patient_id}.png")
        mask_out_path = str(self.masks_root / split / f"{patient_id}.png")
        overlay_out_path = str(self.overlays_root / split / f"{patient_id}.png") if self.save_overlays else ""

        try:
            image_exists = os.path.exists(image_out_path)
            mask_exists = os.path.exists(mask_out_path)
            overlay_exists = (os.path.exists(overlay_out_path) if self.save_overlays else True)

            need_image = not image_exists
            need_mask = not mask_exists
            need_overlay = self.save_overlays and (not overlay_exists)

            # -------------------------------------------------
            # Eğer her şey varsa tekrar üretme, sadece metadata yaz
            # -------------------------------------------------
            if self.skip_existing_files and (not need_image) and (not need_mask) and (not need_overlay):
                existing_img = read_png_gray(image_out_path)
                existing_mask = read_png_gray(mask_out_path)

                if existing_img is None:
                    raise IOError(f"Mevcut image okunamadı: {image_out_path}")
                if existing_mask is None:
                    raise IOError(f"Mevcut mask okunamadı: {mask_out_path}")

                proc_h, proc_w = existing_img.shape[:2]
                if proc_h != self.image_size or proc_w != self.image_size:
                    print(f"[WARN] {patient_id}: mevcut image size {proc_w}x{proc_h}, beklenen {self.image_size}x{self.image_size}")

                mask_area = int((existing_mask > 0).sum())
                lesion_ratio = float(mask_area / float(existing_mask.shape[0] * existing_mask.shape[1]))

                self.processed_rows.append({
                    "patientId": patient_id,
                    "split": split,
                    "target": target,
                    "class": class_name,
                    "class_index": class_to_index(class_name),
                    "has_bbox": has_bbox,
                    "bbox_count": bbox_count,
                    "image_path_dicom": image_path,
                    "image_path_png": image_out_path,
                    "mask_path_png": mask_out_path,
                    "overlay_path_png": overlay_out_path if self.save_overlays else "",
                    "orig_height": np.nan,
                    "orig_width": np.nan,
                    "processed_size": proc_h,
                    "mask_area_pixels": mask_area,
                    "lesion_area_ratio": lesion_ratio,
                    "file_status": "reused_existing_files"
                })
                return

            # -------------------------------------------------
            # Eksik bir şey varsa DICOM'dan yükle
            # -------------------------------------------------
            img_uint8 = read_dicom_as_uint8(image_path)
            orig_h, orig_w = img_uint8.shape[:2]
            resized_img = resize_image(img_uint8, self.image_size)

            if target == 1 and has_bbox == 1 and bbox_count > 0:
                mask = build_union_mask_from_bboxes(
                    bbox_list=bbox_list,
                    orig_h=orig_h,
                    orig_w=orig_w,
                    out_size=self.image_size,
                )
            else:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

            # Sadece eksikleri yaz
            if need_image or (not self.skip_existing_files):
                ok_img = cv2.imwrite(image_out_path, resized_img)
                if not ok_img:
                    raise IOError(f"PNG kaydedilemedi: {image_out_path}")

            if need_mask or (not self.skip_existing_files):
                ok_mask = cv2.imwrite(mask_out_path, mask)
                if not ok_mask:
                    raise IOError(f"Mask kaydedilemedi: {mask_out_path}")

            if self.save_overlays:
                if need_overlay or (not self.skip_existing_files):
                    overlay = create_overlay(resized_img, mask)
                    ok_overlay = cv2.imwrite(overlay_out_path, overlay)
                    if not ok_overlay:
                        raise IOError(f"Overlay kaydedilemedi: {overlay_out_path}")

            # Son halini diskten tekrar oku ki metadata gerçeği yansıtsın
            final_img = read_png_gray(image_out_path)
            final_mask = read_png_gray(mask_out_path)

            if final_img is None:
                raise IOError(f"Kaydedilen image tekrar okunamadı: {image_out_path}")
            if final_mask is None:
                raise IOError(f"Kaydedilen mask tekrar okunamadı: {mask_out_path}")

            mask_area = int((final_mask > 0).sum())
            lesion_ratio = float(mask_area / float(final_mask.shape[0] * final_mask.shape[1]))

            self.processed_rows.append({
                "patientId": patient_id,
                "split": split,
                "target": target,
                "class": class_name,
                "class_index": class_to_index(class_name),
                "has_bbox": has_bbox,
                "bbox_count": bbox_count,
                "image_path_dicom": image_path,
                "image_path_png": image_out_path,
                "mask_path_png": mask_out_path,
                "overlay_path_png": overlay_out_path if self.save_overlays else "",
                "orig_height": orig_h,
                "orig_width": orig_w,
                "processed_size": final_img.shape[0],
                "mask_area_pixels": mask_area,
                "lesion_area_ratio": lesion_ratio,
                "file_status": "generated_missing_files"
            })

        except Exception as e:
            self.failed_cases.append({
                "patientId": patient_id,
                "image_path": image_path,
                "split": split,
                "error": str(e),
            })

    def save_outputs(self):
        processed_df = pd.DataFrame(self.processed_rows)
        failed_df = pd.DataFrame(self.failed_cases)

        processed_csv = self.meta_root / "processed_master_with_outputs.csv"
        train_csv = self.meta_root / "train_preprocessed.csv"
        val_csv = self.meta_root / "val_preprocessed.csv"
        test_csv = self.meta_root / "test_preprocessed.csv"
        failed_csv = self.meta_root / "failed_cases.csv"
        summary_json = self.meta_root / "preprocessing_summary.json"

        if len(processed_df) == 0:
            raise RuntimeError("Hiç processed kayıt oluşmadı. Metadata yazılamaz.")

        processed_df.to_csv(processed_csv, index=False)
        processed_df[processed_df["split"] == "train"].to_csv(train_csv, index=False)
        processed_df[processed_df["split"] == "val"].to_csv(val_csv, index=False)
        processed_df[processed_df["split"] == "test"].to_csv(test_csv, index=False)

        if len(failed_df) > 0:
            failed_df.to_csv(failed_csv, index=False)
        else:
            pd.DataFrame(columns=["patientId", "image_path", "split", "error"]).to_csv(failed_csv, index=False)

        summary = {
            "master_csv": self.master_csv,
            "output_root": str(self.output_root),
            "image_size": self.image_size,
            "save_overlays": bool(self.save_overlays),
            "skip_existing_files": bool(self.skip_existing_files),
            "total_processed": int(len(processed_df)),
            "total_failed": int(len(failed_df)),
            "split_counts": processed_df["split"].value_counts().to_dict(),
            "target_counts": processed_df["target"].value_counts().to_dict(),
            "class_counts": processed_df["class"].value_counts().to_dict(),
            "positive_with_nonzero_mask": int(((processed_df["target"] == 1) & (processed_df["mask_area_pixels"] > 0)).sum()),
            "negative_with_nonzero_mask": int(((processed_df["target"] == 0) & (processed_df["mask_area_pixels"] > 0)).sum()),
            "reused_existing_files_count": int((processed_df["file_status"] == "reused_existing_files").sum()),
            "generated_missing_files_count": int((processed_df["file_status"] == "generated_missing_files").sum()),
        }

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 90)
        print("PREPROCESSING / METADATA REBUILD TAMAMLANDI")
        print("=" * 90)
        print(f"Processed CSV : {processed_csv}")
        print(f"Train CSV     : {train_csv}")
        print(f"Val CSV       : {val_csv}")
        print(f"Test CSV      : {test_csv}")
        print(f"Failed CSV    : {failed_csv}")
        print(f"Summary JSON  : {summary_json}")
        print("-" * 90)
        print(f"Toplam işlenen             : {len(processed_df)}")
        print(f"Hatalı                     : {len(failed_df)}")
        print(f"Mevcut dosyadan kullanılan : {(processed_df['file_status'] == 'reused_existing_files').sum()}")
        print(f"Eksik üretip tamamlanan    : {(processed_df['file_status'] == 'generated_missing_files').sum()}")

        print("\nSplit dağılımı:")
        print(processed_df["split"].value_counts())

        print("\nTarget dağılımı:")
        print(processed_df["target"].value_counts())

        print("\nClass dağılımı:")
        print(processed_df["class"].value_counts())

    def run(self):
        df = self.load_master_csv()

        print(f"\n[INFO] Eksik tamamlama + metadata rebuild başlıyor ({len(df)} örnek)")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing/Rebuild"):
            self.process_one_row(row)

        self.save_outputs()


def main():
    processor = RSNAPreprocessor(
        master_csv=MASTER_CSV,
        output_root=OUTPUT_ROOT,
        image_size=IMAGE_SIZE,
        save_overlays=SAVE_OVERLAYS,
        skip_existing_files=SKIP_EXISTING_FILES,
    )
    processor.run()


if __name__ == "__main__":
    main()