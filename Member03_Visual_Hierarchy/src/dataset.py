
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

SHAPE_R, SHAPE_C = 288, 384

def find_by_stem(folder, stem):
    folder = Path(folder)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        hit = folder / f"{stem}{ext}"
        if hit.exists():
            return hit
    hits = list(folder.glob(stem + ".*"))
    if not hits:
        raise FileNotFoundError(f"{stem} not in {folder}")
    return hits[0]

def resize_pad(img, interpolation=cv2.INTER_LINEAR, pad_value=0):
    """Same geometry as TranSalNet utils/data_process.py preprocess_img."""
    orig_h, orig_w = img.shape[:2]
    if img.ndim == 2:
        padded = np.full((SHAPE_R, SHAPE_C), pad_value, dtype=img.dtype)
    else:
        padded = np.full((SHAPE_R, SHAPE_C, img.shape[2]), pad_value, dtype=img.dtype)
    rows_rate = orig_h / SHAPE_R
    cols_rate = orig_w / SHAPE_C
    if rows_rate > cols_rate:
        new_cols = (orig_w * SHAPE_R) // orig_h
        resized = cv2.resize(img, (new_cols, SHAPE_R), interpolation=interpolation)
        if new_cols > SHAPE_C:
            resized = cv2.resize(img, (SHAPE_C, SHAPE_R), interpolation=interpolation)
            new_cols = SHAPE_C
        start = (SHAPE_C - new_cols) // 2
        padded[:, start:start + new_cols] = resized
    else:
        new_rows = (orig_h * SHAPE_C) // orig_w
        resized = cv2.resize(img, (SHAPE_C, new_rows), interpolation=interpolation)
        if new_rows > SHAPE_R:
            resized = cv2.resize(img, (SHAPE_C, SHAPE_R), interpolation=interpolation)
            new_rows = SHAPE_R
        start = (SHAPE_R - new_rows) // 2
        padded[start:start + new_rows, :] = resized
    return padded

def light_augment(img):
    """Colour only. img is uint8 BGR. Layout is unchanged."""
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.uniform(-15, 15)
    out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
    out = np.clip(out + np.random.normal(0, 5, out.shape), 0, 255).astype(np.uint8)
    quality = int(np.random.randint(75, 96))
    ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out

class UEyesWebDataset(Dataset):
    def __init__(self, root, ids, augment=False):
        self.root = Path(root)
        self.ids = list(ids)
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        image = cv2.imread(str(find_by_stem(self.root / "images", stem)))
        sal = cv2.imread(str(find_by_stem(self.root / "saliency_maps", stem)), cv2.IMREAD_GRAYSCALE)
        fix = cv2.imread(str(find_by_stem(self.root / "fixations", stem)), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            image = light_augment(image)

        image = resize_pad(image, cv2.INTER_LINEAR, pad_value=1)
        sal = resize_pad(sal, cv2.INTER_LINEAR, pad_value=0).astype(np.float32)
        fix = resize_pad(fix, cv2.INTER_NEAREST, pad_value=0)

        image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        sal_01 = sal / (sal.max() + 1e-8)
        sal_dist = sal_01 / (sal_01.sum() + 1e-8)
        fix = (fix > 0).astype(np.float32)

        return {
            "id": stem,
            "image": torch.from_numpy(image),
            "saliency_01": torch.from_numpy(sal_01).unsqueeze(0),
            "saliency_dist": torch.from_numpy(sal_dist).unsqueeze(0),
            "fixation": torch.from_numpy(fix).unsqueeze(0),
        }
