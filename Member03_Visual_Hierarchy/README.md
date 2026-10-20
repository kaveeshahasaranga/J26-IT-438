# Member 03: Visual Hierarchy & Attention Predictor
**Project ID:** J26-IT-438  
**Component:** Human Visual Attention & UI Visual Hierarchy Ranking using Fine-Tuned TranSalNet

---

## 📌 Project Overview
This component predicts human visual attention and visual hierarchy ranking (1 to N) for web interfaces in under a second using deep learning.

- **Baseline Model (Model 1):** Zero-shot `TranSalNet_Res` with SALICON pretrained weights.
- **Proposed Main Model (Model 2):** `TranSalNet_Res` (ResNet-50 Feature Pyramid + Multi-Scale Attention Transformer) fine-tuned on the **UEyes Web Dataset**.
- **Visual Hierarchy Analyzer:** Detects UI bounding regions and ranks them strictly by **Mean Saliency Score** inside each element, categorizing them into Cognitive UX Focus tiers (Dominant, Secondary, Tertiary, Peripheral).

---

## 📊 Benchmark Results (Held-Out Test Set, n=75)
| Metric | Model 1 (Baseline) | Model 2 (Fine-Tuned) | Relative Gain |
|---|---|---|---|
| **Correlation (CC)** | 0.4055 | **0.6386** | **+57.5%** |
| **KL Divergence (Error)** | 1.2137 | **0.7752** | **-36.1% error** |
| **Normalized Scanpath (NSS)** | 0.9971 | **1.7540** | **+75.9%** |

---

## 📁 Repository Structure
```
Member03_Visual_Hierarchy/
├── app/                      # Interactive FastAPI web application
│   ├── main.py               # REST API endpoints (/api/predict, /api/status, /api/samples)
│   ├── inference.py          # Deep learning inference pipeline (384x288 padding, sigmoid)
│   ├── hierarchy.py          # Visual hierarchy ranking & element detection engine
│   └── static/               # Premium glassmorphic web UI (HTML5, Vanilla CSS, JS)
├── data/                     # Webpage attention dataset metadata and curated samples
│   ├── splits.json           # 70/15/15 split by webpage (346 train, 74 val, 75 test)
│   ├── folds.json            # 5-fold cross validation configuration
│   └── ueyes_web/            # UEyes web images, saliency maps, and fixations
├── decisions/
│   └── locked_decisions.txt  # Research protocol locked decisions
├── models/
│   ├── checkpoints/          # Fine-tuned Model 2 checkpoint (best.pt)
│   └── transalnet_pretrained/# Pretrained backbone & baseline weights
├── notebooks/                # Research & training Google Colab notebooks (Step 1 - 4)
├── src/
│   └── dataset.py            # PyTorch Dataset & geometric padding logic
├── TranSalNet/               # TranSalNet model architecture code
├── download_weights.py       # Helper script to download pretrained weights
├── requirements.txt          # Python dependencies
└── run_app.bat               # 1-click Windows desktop launcher
```

---

## 🚀 Setup & Execution

### 1. Clone & Navigate
```bash
git clone https://github.com/kaveeshahasaranga/J26-IT-438.git
cd J26-IT-438/Member03_Visual_Hierarchy
```

### 2. Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Weights Setup
Download the model weights and place them as follows:
- **ResNet-50 Backbone:** Run `python download_weights.py` to auto-download `resnet50-0676ba61.pth` into `models/transalnet_pretrained/`.
- **Baseline Model 1:** Place `TranSalNet_Res.pth` into `models/transalnet_pretrained/`.
- **Fine-Tuned Model 2:** Place `best.pt` into `models/checkpoints/`.

### 4. Run the Web Application
```bash
# Windows 1-click launcher:
run_app.bat

# Or manual command:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Then navigate to: **http://127.0.0.1:8000**
