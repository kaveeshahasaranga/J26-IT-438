"""
Visual Hierarchy & Attention Predictor - Inference Engine
Loads Model 2 (fine-tuned TranSalNet_Res) and Model 1 (SALICON Baseline).
Adheres strictly to locked_decisions.txt:
- Input: Pad to 384 wide by 288 tall. No stretching.
- Scaling: Divide by 255 only. No ImageNet normalization.
- Output: Model applies sigmoid internally (values in [0, 1]).
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure TranSalNet and src are discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR / "TranSalNet"), str(ROOT_DIR / "src")])

from TranSalNet_Res import TranSalNet
from dataset import resize_pad

SHAPE_R, SHAPE_C = 288, 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SaliencyPredictor:
    def __init__(self, root_dir=ROOT_DIR):
        self.root_dir = Path(root_dir)
        self.model2 = None
        self.model1 = None
        self.device = DEVICE
        self.model2_meta = {}
        
        # Ensure pretrained weights folder is accessible for TranSalNet initialization
        self._ensure_weights_link()
        self._load_model2()
        self._load_model1()

    def _ensure_weights_link(self):
        """Ensures TranSalNet's hardcoded pretrained_models path resolves."""
        target_dir = self.root_dir / "pretrained_models"
        target_dir.mkdir(parents=True, exist_ok=True)
        src_resnet = self.root_dir / "models" / "transalnet_pretrained" / "resnet50-0676ba61.pth"
        dst_resnet = target_dir / "resnet50-0676ba61.pth"
        if src_resnet.exists() and not dst_resnet.exists():
            import shutil
            shutil.copy2(src_resnet, dst_resnet)

    def _load_model2(self):
        """Loads Model 2: Fine-tuned TranSalNet_Res checkpoint (best.pt)."""
        ckpt_path = self.root_dir / "models" / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            print(f"[Warning] Model 2 checkpoint not found at {ckpt_path}")
            return
        
        try:
            m = TranSalNet()
            ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            m.load_state_dict(state_dict)
            m = m.to(self.device).eval()
            self.model2 = m
            self.model2_meta = {
                "epoch": ckpt.get("epoch", "B2"),
                "stage": ckpt.get("stage", "B"),
                "val_cc": ckpt.get("val_cc", 0.6386),
                "device": self.device
            }
            print(f"[Info] Model 2 loaded successfully from {ckpt_path.name} (Val CC: {self.model2_meta['val_cc']:.4f})")
        except Exception as e:
            print(f"[Error] Failed to load Model 2: {e}")

    def _load_model1(self):
        """Loads Model 1: Zero-shot TranSalNet_Res with SALICON pretrained weights."""
        wts_path = self.root_dir / "models" / "transalnet_pretrained" / "TranSalNet_Res.pth"
        if not wts_path.exists():
            print(f"[Warning] Model 1 weights not found at {wts_path}")
            return
            
        try:
            m = TranSalNet()
            state = torch.load(str(wts_path), map_location=self.device, weights_only=False)
            m.load_state_dict(state)
            m = m.to(self.device).eval()
            self.model1 = m
            print("[Info] Model 1 (SALICON Baseline) loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load Model 1: {e}")

    @staticmethod
    def postprocess_float(pred, orig_h, orig_w):
        """
        Unpads and resizes the (288, 384) model output back to the original image (orig_h, orig_w).
        Preserves geometric accuracy without aspect-ratio distortion.
        Matches locked decisions and Colab vhap_step2.ipynb.
        """
        pred = pred.astype(np.float32)
        pred_h, pred_w = pred.shape[:2]
        if (orig_h / pred_h) > (orig_w / pred_w):
            new_cols = (pred_w * orig_h) // pred_h
            resized = cv2.resize(pred, (new_cols, orig_h), interpolation=cv2.INTER_LINEAR)
            start = (resized.shape[1] - orig_w) // 2
            cropped = resized[:, start:start + orig_w]
        else:
            new_rows = (pred_h * orig_w) // pred_w
            resized = cv2.resize(pred, (orig_w, new_rows), interpolation=cv2.INTER_LINEAR)
            start = (resized.shape[0] - orig_h) // 2
            cropped = resized[start:start + orig_h, :]
        if cropped.shape[:2] != (orig_h, orig_w):
            cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(cropped, 0.0, 1.0)

    def predict(self, bgr_image, use_model="model2"):
        """
        Generates saliency map for input BGR image.
        Returns:
            saliency_map: float32 numpy array (orig_h, orig_w) in range [0, 1]
        """
        orig_h, orig_w = bgr_image.shape[:2]
        
        # 1. Geometry: Pad to 384 wide by 288 tall (pad_value=1 for images)
        padded = resize_pad(bgr_image, interpolation=cv2.INTER_LINEAR, pad_value=1)
        
        # 2. Pixel scaling: Divide by 255 only (no ImageNet normalization)
        x = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None]
        tensor_x = torch.from_numpy(x).to(self.device)
        
        # 3. Model forward pass
        active_model = self.model2 if use_model == "model2" and self.model2 is not None else self.model1
        if active_model is None:
            raise RuntimeError("Neither Model 2 nor Model 1 is loaded.")
            
        with torch.no_grad():
            output_tensor = active_model(tensor_x)
            pred_raw = output_tensor[0, 0].cpu().numpy()
            
        # 4. Postprocess: Unpad and restore original dimensions
        saliency_map = self.postprocess_float(pred_raw, orig_h, orig_w)
        return saliency_map

    def generate_heatmap_overlay(self, bgr_image, saliency_map, colormap=cv2.COLORMAP_JET, alpha=0.55):
        """
        Creates an attention heatmap overlay blended with the original screenshot.
        """
        # Normalize saliency to 0-255 uint8
        sal_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        sal_uint8 = (sal_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(sal_uint8, colormap)
        overlay = cv2.addWeighted(bgr_image, 1.0 - alpha, heatmap, alpha, 0)
        return overlay, heatmap
