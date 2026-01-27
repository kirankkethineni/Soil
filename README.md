# SOIL — Privacy-First Framework for Secure On-Device Inference of Leaf Diseases on Mobile

[![Paper](https://img.shields.io/badge/Paper-OCIT%202025-blue)](https://github.com/kirankkethineni/Soil)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.17.0-orange)](https://www.tensorflow.org/js)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Project%20Page-Live-brightgreen)](https://kirankkethineni.github.io/Soil/)

> **Kiran K. Kethineni · Saraju P. Mohanty · Elias Kougianos**
> Department of Computer Science and Engineering, University of North Texas, USA

---

**[Project Page](https://kirankkethineni.github.io/Soil/) | [Paper (OCIT 2025)](#citation) | [Code & Notebooks](#code-guide)**

---

## Overview

Mobile-based plant disease diagnosis apps typically **upload diseased leaf images to remote servers**, creating serious data privacy and security risks. **SOIL** eliminates this problem entirely.

SOIL is a **privacy-first, browser-based framework** that:
- Dynamically loads crop-specific CNN models into the mobile browser
- Runs inference **100% on-device** using TensorFlow.js — images never leave the device
- Requires **no app installation** — works on any modern mobile browser (iOS & Android)
- Delivers real-time predictions across 4 crops and 18 disease classes

---

## The Problem

| Traditional Approach | SOIL |
|---|---|
| Images uploaded to server | All inference on-device |
| Privacy/security risks | No data leaves the device |
| Platform-specific apps needed | Works in any browser |
| Dedicated hardware (Raspberry Pi, Jetson) | Any smartphone or tablet |
| Large models, high compute | Lightweight, edge-optimized CNNs |

---

## Key Contributions

1. **Privacy-First On-Device Execution** — All inference performed locally in the browser; image data never transmitted to any server.
2. **Cross-Platform Compatibility Without Installation** — Runs on iOS and Android via native browser capabilities (no app store).
3. **Dynamic Crop-Specific Model Loading** — Only the model for the selected crop is fetched, minimizing memory footprint.
4. **Chroma-Sense Architecture** — Novel CNN design that processes R, G, B channels independently (serial multi-channel processing) to minimize parameters and memory while preserving disease-relevant texture features.

---

## Architecture: Chroma-Sense

The core model innovation is **Chroma-Sense**, a serial multi-channel CNN architecture optimized for browser execution.

### How it works

```
RGB Image (256×256×3)
       │
  ┌────┴────┐
  │  Split  │  ← Extract R, G, B channels independently (256×256×1 each)
  └────┬────┘
       │
  ┌────▼────────────────────────────────┐
  │   Shared Feature Extractor (×3)     │
  │                                     │
  │  Conv2D(3×3) → 128×128×8           │
  │  SepConv2D(3×3) → 64×64×24         │
  │  SepConv2D(3×3) → 32×32×24         │
  │  SepConv2D(3×3) → 16×16×24         │
  │  SepConv2D(3×3) → 8×8×24           │
  │  SepConv2D(3×3) → 4×4×24           │
  │  SepConv2D(3×3) → 2×2×16           │
  └────────────────────────────────────┘
       │
  ┌────▼────┐
  │  Stack  │  ← Concatenate 3 channel feature maps → (2×2×48)
  └────┬────┘
       │
  Conv2D(1×1) → Point-wise fusion → (2×2×16)
       │
  GlobalMaxPooling2D → Dense(16) → Dense(N, softmax)
       │
  Disease Class Prediction
```

**Why Chroma-Sense?**
- Disease patterns (spots, rings, rust, blight) manifest in individual color channels
- Shared weights across channels dramatically reduce total parameters
- Point-wise convolution restores inter-channel correlation after fusion
- Result: compact, accurate models suited for browser execution

### Model Conversion Pipeline

```
TensorFlow/Keras Model (Python)
           │
  tensorflowjs_converter
           │
    ┌──────┴──────┐
    │  model.json │  ← Architecture, layer topology, metadata
    │  .bin files │  ← Compressed, sharded weight parameters
    └──────┬──────┘
           │
     Hosted on Server
           │
     Browser fetches + caches
           │
  TF.js inference (WebGL → WASM → CPU fallback)
```

---

## Browser Execution with TensorFlow.js

SOIL uses TF.js to run inference natively in the browser with three automatic backend fallbacks:

| Backend | Trigger | Performance |
|---------|---------|-------------|
| **WebGL** | GPU available | Near real-time (~16ms on iPhone 14 Pro) |
| **WebAssembly (WASM)** | No GPU / GPU disabled | Efficient CPU execution |
| **Pure JavaScript** | Legacy/minimal browsers | Functional, higher latency |

---

## Inference Workflow

```
User opens browser → Selects crop type
         │
Fetch model.json + .bin files from server
         │
Cache locally (reused across sessions)
         │
Initialize TF.js (auto-select backend)
         │
Capture leaf image (camera) or upload from gallery
         │
Preprocess: resize → normalize → convert to tensor
         │
Run inference entirely on-device
         │
Display prediction + confidence score
         │
Dispose tensors (memory management)
```

**All steps occur within the browser. No image data ever leaves the device.**

---

## Dataset

Curated subset of the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset:

| Crop | Disease Classes | Approx. Images |
|------|----------------|----------------|
| Apple | Black Rot, Black Scab, Cedar Rust, Mosaic, Septoria Spot, Healthy | ~4,500 |
| Tomato | Mold, Curls, Blight, Mosaic, Septoria Spot, Healthy | ~4,500 |
| Grape | Black Rot, Esca, Blight, Healthy | ~3,000 |
| Corn | Rust, Blight, Leaf Spots, Healthy | ~3,000 |
| **Total** | **18 classes** | **~18,000** |

- Image resolution: **256×256** pixels
- Train/Validation split: **80:20**

---

## Results

### F1 Scores by Crop

| Crop | Disease | F1 Score |
|------|---------|----------|
| **Apple** (Overall: 94%) | Healthy | 0.95 |
| | Black Rot | 0.95 |
| | Black Scab | 0.94 |
| | Cedar Rust | 0.92 |
| | Mosaic | 0.93 |
| | Septoria Spot | 0.91 |
| **Tomato** (Overall: 91%) | Healthy | 0.92 |
| | Mold | 0.90 |
| | Curls | 0.91 |
| | Blight | 0.90 |
| | Mosaic | 0.93 |
| | Septoria Spot | 0.91 |
| **Grape** (Overall: 96%) | Healthy | 0.98 |
| | Black Rot | 0.96 |
| | Esca | 0.96 |
| | Blight | 0.95 |
| **Corn** (Overall: 95%) | Healthy | 0.96 |
| | Rust | 0.94 |
| | Blight | 0.94 |
| | Leaf Spots | 0.96 |

### Inference Latency Across Devices

| Device | Platform | Hardware | Inference Time |
|--------|----------|----------|---------------|
| Windows Laptop | Windows 10 | NVIDIA RTX 3060 | **13.5 ms** |
| Apple iPhone 14 Pro Max | iOS | Apple GPU | **~16 ms** |
| Samsung A16 | Android | Mid-range | **94 ms** |
| LG Stylo 4 | Android | Entry-level | **400 ms** |

All inference executed **entirely on-device** across all tested hardware.

---

## Code Guide

### Repository Structure

```
Soil/
├── Soil.ipynb                    # Model training, evaluation, Grad-CAM, export
├── templates/
│   ├── index.html                # Production web interface (multi-crop)
│   └── upload.html               # Experimental interface (serial/parallel comparison)
├── static/
│   ├── apple/                    # Apple disease model (model.json + .bin)
│   ├── grape/                    # Grape disease model (model.json + .bin)
│   ├── full_model/               # End-to-end inference model
│   ├── shared_model/             # Shared convolutional encoder
│   └── classifier_model/         # Classification head
├── docs/
│   └── index.html                # GitHub Pages project site
└── README.md
```

### Training (Soil.ipynb)

The notebook covers the complete ML pipeline:

1. **Data loading** — Load images from PlantVillage crop folders, resize to 256×256, normalize to [0,1]
2. **Architecture** — Build `shared_conv_model` (per-channel encoder) + full model with channel fusion
3. **Training** — Adam optimizer, categorical crossentropy, 75 epochs, ReduceLROnPlateau
4. **Evaluation** — Confusion matrix, validation accuracy
5. **Grad-CAM** — Interpretability visualizations showing which image regions drive predictions
6. **Export** — Convert trained model to TF.js format (model.json + .bin shards)

### Running the Web Interface

The web interface is served as a Flask app (or any static file server). The HTML/JS in `templates/index.html` handles all model loading and inference in-browser.

```bash
# Serve locally (example with Python)
python -m http.server 8000

# Or with ngrok for mobile testing
ngrok http 8000
```

Then open on any device browser — select a crop, capture or upload a leaf image, and get instant on-device predictions.

---

## Comparison with Related Work

| Work | Method | Cross-Device | Privacy |
|------|--------|:---:|:---:|
| Khan et al. | Edge-optimized Model | ✗ | ✓ |
| Ahamed et al. | Edge-optimized Model | ✗ | ✓ |
| Akuthota et al. | Edge-optimized Model | ✗ | ✓ |
| Chaitra et al. | Cloud-based inference | ✓ | ✗ |
| Karim et al. | Edge-optimized models | ✗ | ✓ |
| Iftikhar et al. | On-device application | ✗ | ✓ |
| Foysal et al. | On-device application | ✗ | ✓ |
| Zeeshan et al. | IoT-optimized Model | ✗ | ✓ |
| **SOIL (Ours)** | **Browser-based on-device** | **✓** | **✓** |

SOIL is the **only approach** achieving both cross-device compatibility and data privacy simultaneously.

---

## Citation

If you use SOIL in your research, please cite:

```bibtex
@inproceedings{kethineni2025soil,
  title     = {SOIL: A Privacy-First Framework for Secure On-Device Inference of Leaf Diseases on Mobile},
  author    = {Kethineni, Kiran K. and Mohanty, Saraju P. and Kougianos, Elias},
  booktitle = {Proceedings of OCIT 2025},
  year      = {2025},
  institution = {University of North Texas}
}
```

---

## Authors

| Name | Role | Institution |
|------|------|------------|
| Kiran K. Kethineni | Lead Author | Dept. of Computer Science & Engineering, UNT |
| Saraju P. Mohanty | Advisor | Dept. of Computer Science & Engineering, UNT |
| Elias Kougianos | Co-Advisor | Dept. of Electrical Engineering, UNT |

---

*For questions or issues, please open a GitHub Issue.*
