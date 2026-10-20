"""
Visual Hierarchy & Attention Predictor - Web API Backend
FastAPI server serving inference endpoints and the interactive web frontend.
"""

import base64
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add app and root directory to python path
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.extend([str(APP_DIR), str(ROOT_DIR / "TranSalNet"), str(ROOT_DIR / "src")])

from hierarchy import VisualHierarchyAnalyzer
from inference import SaliencyPredictor

app = FastAPI(
    title="Visual Hierarchy & Attention Predictor",
    description="Predict human visual attention and UI visual hierarchy rankings using TranSalNet Model 2",
    version="1.0.0"
)

# Enable CORS for local development and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor & Hierarchy Analyzer
predictor = SaliencyPredictor(root_dir=ROOT_DIR)
analyzer = VisualHierarchyAnalyzer(max_elements=15)

COLORMAP_MAP = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def encode_image_base64(bgr_image, ext=".png", quality=92):
    """Encodes a BGR numpy image into a base64 data URI string."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, bgr_image, params)
    if not success:
        raise ValueError("Failed to encode image to base64")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


@app.get("/api/status")
def get_status():
    """Returns the initialization status of Model 2, Model 1, and device."""
    return {
        "status": "ready" if predictor.model2 is not None else "degraded",
        "device": predictor.device,
        "model2": {
            "loaded": predictor.model2 is not None,
            "name": "TranSalNet_Res (Model 2 - Webpage Fine-Tuned)",
            "checkpoint": "models/checkpoints/best.pt",
            "val_cc": predictor.model2_meta.get("val_cc", 0.6386),
            "stage": predictor.model2_meta.get("stage", "B"),
            "epoch": predictor.model2_meta.get("epoch", "B2"),
        },
        "model1": {
            "loaded": predictor.model1 is not None,
            "name": "TranSalNet_Res (Model 1 - SALICON Baseline)",
            "weights": "models/transalnet_pretrained/TranSalNet_Res.pth",
            "test_cc": 0.4055,
        },
        "locked_decisions": {
            "input_resolution": "384 x 288 (Padded, No Stretch)",
            "scaling": "Divide by 255 (No ImageNet Normalization)",
            "element_metric": "Mean saliency inside bounding box",
        }
    }


@app.get("/api/samples")
def get_samples():
    """Lists available webpage sample images for instant 1-click testing."""
    samples_dir = ROOT_DIR / "data" / "ueyes_web" / "images"
    if not samples_dir.exists():
        return {"samples": []}

    # Pick 8 distinct curated samples from the dataset
    all_images = sorted([f.name for f in samples_dir.glob("*.png") if not f.name.startswith(".")])
    selected = all_images[:10]

    samples = []
    for name in selected:
        stem = Path(name).stem
        samples.append({
            "id": stem,
            "filename": name,
            "title": f"Webpage Design {stem}",
            "thumbnail_url": f"/api/sample/{name}"
        })
    return {"samples": samples}


@app.get("/api/sample/{filename}")
def get_sample_image(filename: str):
    """Serves a sample image file from the dataset."""
    file_path = ROOT_DIR / "data" / "ueyes_web" / "images" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(file_path), media_type="image/png")


@app.post("/api/predict")
async def predict_attention(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    colormap: str = Form("jet"),
    alpha: float = Form(0.55),
    model: str = Form("model2"),
    compare_baseline: bool = Form(False),
):
    """
    Main prediction endpoint:
    Accepts an uploaded screenshot or a sample_id.
    Computes saliency map, heatmap overlays, element detection, and visual hierarchy ranking.
    """
    bgr_image = None

    # Load image from either file upload or sample ID
    if file is not None and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif sample_id:
        img_path = ROOT_DIR / "data" / "ueyes_web" / "images" / f"{sample_id}.png"
        if img_path.exists():
            bgr_image = cv2.imread(str(img_path))

    if bgr_image is None:
        raise HTTPException(status_code=400, detail="No valid image provided via upload or sample_id.")

    orig_h, orig_w = bgr_image.shape[:2]

    # Select colormap
    cv2_colormap = COLORMAP_MAP.get(colormap.lower(), cv2.COLORMAP_JET)

    # 1. Run Primary Model (Model 2: Fine-Tuned)
    try:
        saliency_map = predictor.predict(bgr_image, use_model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 2. Generate Heatmap Overlays
    overlay, heatmap = predictor.generate_heatmap_overlay(
        bgr_image, saliency_map, colormap=cv2_colormap, alpha=float(alpha)
    )

    # 3. Analyze Visual Hierarchy & Detect Ranked Elements
    elements, annotated_hierarchy = analyzer.analyze_hierarchy(bgr_image, saliency_map)

    # 4. Optional: Run Model 1 Baseline for Side-by-Side Comparison
    baseline_overlay_b64 = None
    baseline_stats = None
    if compare_baseline and predictor.model1 is not None:
        try:
            m1_saliency = predictor.predict(bgr_image, use_model="model1")
            m1_overlay, _ = predictor.generate_heatmap_overlay(
                bgr_image, m1_saliency, colormap=cv2_colormap, alpha=float(alpha)
            )
            baseline_overlay_b64 = encode_image_base64(m1_overlay)
            baseline_stats = {
                "mean_attention": round(float(np.mean(m1_saliency)), 4),
                "peak_attention": round(float(np.max(m1_saliency)), 4),
            }
        except Exception as ex:
            print(f"[Warning] Baseline comparison failed: {ex}")

    # Calculate overall image statistics
    mean_att = float(np.mean(saliency_map))
    peak_att = float(np.max(saliency_map))
    top_3_score_sum = sum(e["mean_score"] for e in elements[:3])
    total_score_sum = sum(e["mean_score"] for e in elements) if elements else 1.0
    top_3_share = int(round((top_3_score_sum / max(1e-6, total_score_sum)) * 100))

    # Base64 encodings
    original_b64 = encode_image_base64(bgr_image)
    heatmap_b64 = encode_image_base64(heatmap)
    overlay_b64 = encode_image_base64(overlay)
    hierarchy_b64 = encode_image_base64(annotated_hierarchy)

    return {
        "status": "success",
        "image_size": {"width": orig_w, "height": orig_h},
        "model_used": "Model 2 (Fine-Tuned on Webpages)" if model == "model2" else "Model 1 (SALICON Baseline)",
        "stats": {
            "mean_attention": round(mean_att, 4),
            "peak_attention": round(peak_att, 4),
            "elements_detected": len(elements),
            "top_3_focus_share_pct": min(100, top_3_share),
            "baseline": baseline_stats,
        },
        "images": {
            "original": original_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64,
            "hierarchy": hierarchy_b64,
            "baseline_overlay": baseline_overlay_b64,
        },
        "elements": elements,
    }


# Mount static directory for frontend UI
static_dir = APP_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
