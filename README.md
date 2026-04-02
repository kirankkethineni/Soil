# SOIL — Privacy-First Framework for Secure On-Device Inference of Leaf Diseases on Mobile

[![Paper](https://img.shields.io/badge/Paper-OCIT%202025-blue)](https://github.com/kirankkethineni/Soil)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.17.0-orange)](https://www.tensorflow.org/js)
[![Project Page](https://img.shields.io/badge/Project%20Page-Live-brightgreen)](https://kirankkethineni.github.io/Soil/)

> **Kiran K. Kethineni · Saraju P. Mohanty · Elias Kougianos**
> Department of Computer Science and Engineering, University of North Texas, USA

**[Project Page](https://kirankkethineni.github.io/Soil/) | [Paper (OCIT 2025)](https://ieeexplore.ieee.org/abstract/document/11400270) | [Jump to Code](#code-guide)**

---

## Table of Contents

1. [The Core Problem — and Why It Is Harder Than It Looks](#1-the-core-problem--and-why-it-is-harder-than-it-looks)
2. [The Key Insight Behind SOIL](#2-the-key-insight-behind-soil)
3. [Why Not the Obvious Alternatives?](#3-why-not-the-obvious-alternatives)
4. [The Chroma-Sense Architecture — Intuition First](#4-the-chroma-sense-architecture--intuition-first)
5. [From Training to Browser — The Conversion Pipeline](#5-from-training-to-browser--the-conversion-pipeline)
6. [How TensorFlow.js Runs on Any Phone](#6-how-tensorflowjs-runs-on-any-phone)
7. [The Full Inference Workflow](#7-the-full-inference-workflow)
8. [Dataset](#8-dataset)
9. [Results and What They Tell Us](#9-results-and-what-they-tell-us)
10. [Comparison with Prior Work](#10-comparison-with-prior-work)
11. [Code Guide](#11-code-guide)
12. [Citation](#12-citation)

---

## 1. The Core Problem — and Why It Is Harder Than It Looks

Agriculture depends on early, accurate disease detection. AI has gotten good at this — convolutional networks trained on leaf images can classify dozens of diseases with high accuracy. So far so good.

The problem is *delivery*. How do you get that AI capability into the hands of a farmer standing in a field?

The naive answer is a mobile app. But building a mobile app that runs a CNN model exposes a **fundamental tension** between two properties you want simultaneously:

| Property | What it requires |
|---|---|
| **Privacy** — image data stays on the device | Inference must run locally, not on a server |
| **Cross-platform** — works on any phone | A universal runtime, not platform-specific binaries |

Nearly every existing system sacrifices one of these:

- **Cloud-based apps** (e.g., Chaitra et al.) upload images to a remote server for inference. They work on any phone — but your diseased leaf image travels over the internet to a third-party server. This opens doors to eavesdropping, data tampering, and loss of farmer data ownership. In regions with intermittent connectivity, it simply fails.

- **Edge-deployed models** (e.g., Khan et al., Ahamed et al., Karim et al.) run inference locally — on a Raspberry Pi, Jetson Nano, or similar dedicated hardware. Privacy preserved. But you have now required farmers to purchase and set up specialist hardware they will use for a single task. Impractical at scale.

- **Native mobile apps** (e.g., Iftikhar et al., Foysal et al.) run on the phone itself — but require separate iOS and Android builds, app store approval, installation, and updates. Users are reluctant to install apps they will use infrequently. And maintaining two native codebases is expensive.

**The gap**: no prior work achieves *both* local, private inference *and* frictionless cross-platform deployment without dedicated hardware or app installation. SOIL is designed to close exactly that gap.

---

## 2. The Key Insight Behind SOIL

The insight that makes SOIL possible is deceptively simple:

> **Every smartphone already has a GPU-accelerated execution environment that requires no installation: the web browser.**

Modern mobile browsers — Chrome on Android, Safari on iOS — support two technologies that together enable real neural network inference on the device:

- **WebGL**: A JavaScript API that exposes the phone's GPU for general-purpose computation. Tensor multiplications and convolutions can be mapped directly to GPU shader programs. This is how high-end phones achieve 13–16ms inference.

- **WebAssembly (WASM)**: A portable, near-native binary format that runs in the browser's sandboxed environment without GPU access. Provides efficient CPU inference on devices where WebGL is restricted or unavailable.

TensorFlow.js (TF.js) exploits both. It takes a standard trained neural network and executes it entirely inside the browser — with WebGL when a GPU is available, WASM otherwise, and pure JavaScript as a final fallback. No installation. No platform-specific build. Works on iOS and Android identically.

This means: if you can train a model compact enough to load and run in a browser, you get privacy-preserving, cross-platform, installation-free deployment for free. The browser is the universal runtime that was already in every farmer's pocket.

**SOIL's central contribution is building the full stack around this insight** — from a purpose-built lightweight CNN architecture, through a model conversion and serving pipeline, to a web interface that handles camera capture, preprocessing, and inference end-to-end in the browser.

---

## 3. Why Not the Obvious Alternatives?

Before committing to this approach, it is worth understanding why other reasonable-sounding options were insufficient.

### Why not just convert any existing model to TF.js?

Standard ImageNet-pretrained models (ResNet, EfficientNet, MobileNet) are designed for server or high-end device execution. Their weights alone can be tens to hundreds of megabytes. Loading a 50MB model in a mobile browser means:
- Long initial load time, especially on rural/slow connections
- Significant RAM pressure on low-end devices
- High GPU memory usage during WebGL execution

TF.js can technically run large models, but the user experience degrades severely. The model architecture must be co-designed with the deployment constraint — small enough to be useful on a mid-range or budget phone.

### Why not use MobileNet or similar efficient architectures directly?

MobileNet and related architectures use depthwise separable convolutions to reduce parameters — they were designed for mobile *native* apps, where memory is still measured in hundreds of megabytes. They remain significantly larger than what we need for browser execution, and they were not designed with the specific texture-pattern nature of leaf disease in mind.

### Why not a single model for all crops?

A single multi-crop model would need to distinguish between all 18 disease classes across 4 crops simultaneously. This requires a wider, deeper architecture. More parameters mean larger file sizes, slower loading, and higher memory use during inference. Crucially, the model must be in browser memory for the entire session.

The per-crop design means: the model for apple disease detection never loads if the user is diagnosing corn. Each model is small and specialized. Once loaded, it is cached by the browser — subsequent inferences are instant and require no network access at all.

---

## 4. The Chroma-Sense Architecture — Intuition First

The custom CNN architecture in SOIL is called **Chroma-Sense**. Its design is driven by two questions: what is special about plant disease images, and what are the memory/compute constraints of browser execution?

### The disease pattern observation

Plant diseases are fundamentally *textural and chromatic* phenomena. Black rot produces dark, circular lesions. Cedar rust creates orange pustules. Septoria spot manifests as small, circular spots with dark borders. Blight causes rapidly spreading brown necrosis.

These patterns do two things consistently:
1. They appear in **specific regions** of the leaf — localized patches, not diffuse
2. They produce **characteristic texture signatures within individual color channels** — rust changes the red channel strongly, chlorosis alters the green channel, necrosis shows in all three but with different intensities

This suggests that processing each color channel independently, with a feature extractor that learns texture and shape, could capture disease-discriminating information effectively — possibly *more* effectively than jointly processing all three channels from the start, because joint processing forces the model to simultaneously model both within-channel texture and between-channel interaction, requiring more capacity.

### The shared encoder trick

The key efficiency move in Chroma-Sense is using **one shared encoder** for all three channels, applied sequentially:

```
R channel → [encoder] → feature map R   (2×2×16)
G channel → [encoder] → feature map G   (2×2×16)   ← same weights
B channel → [encoder] → feature map B   (2×2×16)   ← same weights
```

This is fundamentally different from having three separate encoders. The encoder learns a universal set of texture/shape detectors — spot detectors, edge detectors, patch detectors — and applies them to each channel. The same parameters serve all three channels. This reduces the parameter count of the feature extraction stage by approximately 3× compared to three independent encoders.

The intuition is that what constitutes a "disease texture" is the same concept whether you are looking at the red channel, the green channel, or the blue channel. Learning one good texture extractor and reusing it is more efficient than learning three redundant ones.

### The architecture in detail

Each channel passes through the shared encoder:

```
Input: 256×256×1 (single channel)
│
Conv2D(8, 3×3, stride=2)            → 128×128×8    ← standard conv for initial feature extraction
SeparableConv2D(24, 3×3, stride=2)  →  64×64×24   ┐
SeparableConv2D(24, 3×3, stride=2)  →  32×32×24   │
SeparableConv2D(24, 3×3, stride=2)  →  16×16×24   │ depthwise separable convs
SeparableConv2D(24, 3×3, stride=2)  →   8×8×24    │ for parameter efficiency
SeparableConv2D(24, 3×3, stride=2)  →   4×4×24    │
SeparableConv2D(16, 3×3, stride=2)  →   2×2×16    ┘

Output per channel: 2×2×16 feature map
```

**Why SeparableConv2D instead of standard Conv2D?**

A standard Conv2D with kernel size 3×3, C_in input channels, and C_out output channels has `3×3×C_in×C_out` parameters per layer. Depthwise separable convolution factors this into two operations: a depthwise conv (`3×3×C_in`) that filters each channel independently, followed by a pointwise conv (`1×1×C_in×C_out`) that combines them. Total: `3×3×C_in + C_in×C_out` — roughly 8-9× fewer parameters for typical channel counts. For browser execution where total model size directly affects load time and GPU memory use, this reduction is not cosmetic — it is what makes the architecture viable on mid-range and budget phones.

**Why stride=2 convolutions instead of pooling?**

Strided convolutions let the network learn *which* spatial information to discard as it downsamples, rather than always discarding it via a fixed max or average operation. This gives the model more flexibility in learning the right spatial abstraction for disease detection, at no additional parameter cost.

### Channel fusion: recovering inter-channel relationships

After processing each channel independently, the three feature maps are concatenated:

```
[feat_R | feat_G | feat_B]  →  2×2×48
```

But now the channel relationships that were discarded at the input (by splitting R, G, B apart) need to be restored. A **1×1 convolution** (pointwise convolution) does exactly this:

```
Conv2D(16, kernel=1×1)  →  2×2×16
```

A 1×1 conv learns a learned linear combination across the 48 stacked channels. It is essentially asking: given what the encoder found in R, G, and B independently, what combination of these features is most useful for disease classification? This restores inter-channel correlation in a data-driven way, after the independent per-channel processing has extracted structured, texture-specific features from each.

### GlobalMaxPooling2D as the spatial aggregation

After the 1×1 fusion, the feature map is 2×2×16. A `GlobalMaxPooling2D` layer reduces this to a 16-dimensional vector by taking the maximum activation across all spatial positions for each channel.

Why max rather than average? For disease detection, you want to know whether the *most activated* region of the feature map strongly signals a disease — not the average activation across the whole image. A single strong disease lesion in one corner of the image should produce a confident classification. Max pooling captures the peak signal; average pooling would dilute it with background.

### Summary of the architecture design philosophy

| Design choice | The constraint it serves | The intuition behind it |
|---|---|---|
| Process channels independently | Browser memory / parameter budget | Disease textures live in individual channels; shared encoder is 3× more efficient |
| Shared encoder weights | Parameter count reduction | Texture detectors are channel-agnostic |
| SeparableConv2D | File size, GPU memory, load time | 8-9× parameter reduction with minimal accuracy loss |
| 1×1 pointwise fusion | Restore inter-channel context | Let the network learn which channel combinations matter |
| GlobalMaxPooling2D | Spatial invariance | Disease may appear anywhere; peak signal matters more than average |
| Per-crop models | Browser session memory | Only load what you need; cached after first use |

---

## 5. From Training to Browser — The Conversion Pipeline

Training happens in Python with TensorFlow/Keras on the full PlantVillage dataset. The trained model is then converted for browser deployment:

```
TensorFlow SavedModel
        │
        ▼
tensorflowjs_converter
        │
        ├──  model.json      ← layer topology, operator types, input/output shapes, metadata
        └──  *.bin shards    ← model weights in compressed, partitioned binary format
```

**Why sharded .bin files?**

Large weight matrices are split into multiple smaller binary files (shards). This has two practical benefits:

1. **Progressive loading**: the browser can begin loading model weights in parallel with page initialization. TF.js can start preparing the WebGL context while remaining shards arrive.

2. **Browser caching**: HTTP caching works at the file level. If a model has 5 shards and the user returns the next day, the browser serves all 5 from cache — zero network traffic. For rural users on metered or slow connections, this matters significantly. The model is essentially installed implicitly the first time it is used.

The model.json file encodes the full graph topology and acts as the index. The .bin files are looked up by the names referenced in model.json. TF.js loads model.json first, then fetches the weight shards in parallel.

---

## 6. How TensorFlow.js Runs on Any Phone

TF.js automatically selects the best available execution backend for the current device and browser:

### WebGL (primary — GPU-accelerated)

WebGL is a browser standard that exposes the device GPU for 2D/3D rendering — but its shader programs can be repurposed for general matrix computation. TF.js maps tensor operations (matrix multiply, convolution, activation functions) onto WebGL shader programs that execute in parallel on GPU cores.

This is how the iPhone 14 Pro Max achieves 16ms inference: the same GPU that renders games and video is executing the CNN forward pass entirely within the browser sandbox. No special access, no native code, no installation.

### WebAssembly (fallback — efficient CPU)

When the GPU is unavailable (disabled by browser policy, insufficient VRAM, or older device), TF.js falls back to WebAssembly. WASM is a compiled binary format — TF.js ships pre-compiled computational kernels for common operations. These run at near-native speed on the CPU, without JIT compilation overhead.

This is how the Samsung A16 achieves 94ms inference — CPU-only, but efficient because WASM kernels avoid the overhead of interpreted JavaScript.

### Pure JavaScript (final fallback)

On legacy browsers or devices without WASM support, TF.js falls back to pure JavaScript. Inference is slower (the LG Stylo 4 takes 400ms) but fully functional. The key point: even the oldest device in the test set can run plant disease diagnosis in the browser without any special support.

This three-tier fallback is what gives SOIL its "works on any device" guarantee. The same codebase — the same HTML file, the same model files — runs on an iPhone 14, a mid-range Android, and a 2018 budget phone.

---

## 7. The Full Inference Workflow

End-to-end, the process from browser open to disease prediction:

```
1.  User opens the web interface in their mobile browser
2.  User selects crop type (Apple / Tomato / Grape / Corn)
3.  Browser fetches model.json + .bin shards from server
        └─ If previously visited: served from browser cache (no network)
4.  TF.js initializes, auto-selects backend (WebGL → WASM → JS)
5.  User captures leaf photo (camera) or uploads from gallery
        └─ Image stays in browser memory. Never transmitted anywhere.
6.  Preprocessing (client-side):
        resize to 256×256  →  normalize pixels to [0, 1]  →  tf.browser.fromPixels()
7.  Inference: preprocessed tensor → TF.js model.predict() → probability vector
8.  Top class + confidence score displayed in browser UI
9.  Tensors explicitly disposed (tf.dispose()) to free GPU/CPU memory
```

The critical privacy property holds at every step: between steps 5 and 9, all computation occurs inside the browser process on the user's device. The server is never contacted after model loading.

---

## 8. Dataset

SOIL uses a curated subset of the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset, partitioned by crop to support per-crop model loading.

| Crop | Disease Classes | Images |
|------|----------------|--------|
| Apple | Healthy, Black Rot, Black Scab, Cedar Rust, Mosaic, Septoria Spot | ~4,500 |
| Tomato | Healthy, Mold, Curls, Blight, Mosaic, Septoria Spot | ~4,500 |
| Grape | Healthy, Black Rot, Esca, Blight | ~3,000 |
| Corn | Healthy, Rust, Blight, Leaf Spots | ~3,000 |
| **Total** | **18 classes** | **~18,000** |

All images are resized to **256×256** pixels. Train/validation split: **80:20**.

One deliberate data handling decision: disease-specific semantics (the particular textures of rings, spots, and patches that distinguish diseases) are preserved across the split. This means train and validation sets contain similar *pattern diversity*, which pushes the model to learn generalizable features rather than memorizing the particular backgrounds or lighting of specific training images.

---

## 9. Results and What They Tell Us

### F1 Scores

| Crop | Disease | F1 | | Crop | Disease | F1 |
|---|---|---|---|---|---|---|
| **Apple** (94%) | Healthy | 0.95 | | **Grape** (96%) | Healthy | 0.98 |
| | Black Rot | 0.95 | | | Black Rot | 0.96 |
| | Black Scab | 0.94 | | | Esca | 0.96 |
| | Cedar Rust | 0.92 | | | Blight | 0.95 |
| | Mosaic | 0.93 | | **Corn** (95%) | Healthy | 0.96 |
| | Septoria Spot | 0.91 | | | Rust | 0.94 |
| **Tomato** (91%) | Healthy | 0.92 | | | Blight | 0.94 |
| | Mold | 0.90 | | | Leaf Spots | 0.96 |
| | Curls | 0.91 | | | | |
| | Blight | 0.90 | | | | |
| | Mosaic | 0.93 | | | | |
| | Septoria Spot | 0.91 | | | | |

**What these numbers tell us**: Grape achieves the highest overall F1 (96%) — its disease classes have visually distinct textures (Esca's internal leaf discoloration, Black Rot's circular lesions, Blight's rapidly spreading necrosis). Tomato's 91% overall is slightly lower, likely because tomato disease classes like Mold and Blight can present with overlapping visual characteristics at certain disease stages. Nonetheless, all four crops achieve strong performance (91–96%) from a model small enough to fit in a browser.

### Inference Latency

| Device | Platform | Hardware | Latency | Backend used |
|--------|----------|----------|---------|-------------|
| Windows Laptop | Windows 10 | NVIDIA RTX 3060 | **13.5 ms** | WebGL |
| iPhone 14 Pro Max | iOS | Apple GPU | **~16 ms** | WebGL |
| Samsung A16 | Android | Mid-range CPU | **94 ms** | WASM |
| LG Stylo 4 | Android | Entry-level CPU | **400 ms** | JavaScript |

**What these numbers tell us**: The latency spread is about 30× between best and worst device, but the critical observation is that *every device is functional*. At 400ms, the LG Stylo 4 is not real-time, but for a farmer in a field who captures a photo and waits less than half a second for a disease prediction, this is entirely acceptable. The framework degrades gracefully rather than failing.

The 13.5ms on a desktop GPU and 16ms on an iPhone confirm that the WebGL backend achieves near-real-time interactive diagnosis on any mid-to-high device — without a server, without an app, and with complete data privacy.

### Grad-CAM: What the Model Actually Learned

Beyond accuracy numbers, the paper uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of the input image most influenced the model's prediction. The heat maps confirm the model is looking at the right things:

- For **Cedar Rust** on apple: attention concentrated on the orange-yellow pustule clusters
- For **Black Scab** on apple: attention on the dark, scabby lesions
- For **Blight** on grape: attention on the brown, spreading necrotic regions
- For **Rust** on corn: attention on the elongated rust pustule rows on the leaf surface

This is important because a model can achieve good accuracy on a benchmark by picking up on spurious correlations — background color, image artifacts, dataset-specific cues — rather than actual disease features. The Grad-CAM evidence shows that Chroma-Sense is learning the right visual signatures, which matters for generalization to real-world field conditions.

---

## 10. Comparison with Prior Work

The literature on plant disease detection on mobile/edge devices is substantial, but the specific combination of constraints SOIL targets — both privacy-preserving AND cross-platform — has not been addressed before.

| Work | Approach | Cross-Device Compatible | Privacy |
|------|----------|:---:|:---:|
| Khan et al. | Edge-optimized Model | ✗ | ✓ |
| Ahamed et al. | Edge-optimized Model | ✗ | ✓ |
| Akuthota et al. | Edge-optimized Model | ✗ | ✓ |
| Chaitra et al. | Cloud-based inference | ✓ | ✗ |
| Karim et al. | Edge-optimized models | ✗ | ✓ |
| Iftikhar et al. | On-device native app | ✗ | ✓ |
| Foysal et al. | On-device native app | ✗ | ✓ |
| Zeeshan et al. | IoT-optimized model | ✗ | ✓ |
| **SOIL (Ours)** | **Browser-based on-device** | **✓** | **✓** |

The pattern is clear: cloud-based approaches get cross-device compatibility by delegating compute to a server, but lose privacy. Edge and native-app approaches preserve privacy but are tied to specific hardware or platforms. SOIL achieves the combination by recognizing that the browser is already a universal, GPU-capable, installation-free runtime.

---

## 11. Code Guide

### Repository Structure

```
Soil/
├── Soil.ipynb              ← Complete ML pipeline: train → evaluate → Grad-CAM → export
├── templates/
│   ├── index.html          ← Production web interface (multi-crop, TF.js inference)
│   └── upload.html         ← Experimental interface (serial vs parallel channel comparison)
├── static/
│   ├── apple/              ← Apple model: model.json + group1-shard1of1.bin + labels.json
│   ├── grape/              ← Grape model: model.json + group1-shard1of1.bin + labels.json
│   ├── full_model/         ← End-to-end RGB model (single-file inference path)
│   ├── shared_model/       ← Shared channel encoder (for decomposed inference)
│   └── classifier_model/   ← Classification head (for decomposed inference)
├── docs/
│   └── index.html          ← GitHub Pages project site
└── README.md
```

### Notebook Walkthrough — `Soil.ipynb`

The notebook is the complete paper implementation, in sequence:

**Step 1 — Data loading and preprocessing**
Images are loaded from crop-organized PlantVillage folders, resized to 256×256, normalized to [0, 1] float32, and split 80/20. Labels are integer-encoded then one-hot encoded for categorical crossentropy.

**Step 2 — Architecture definition**
`build_shared_conv_model()` defines the per-channel encoder. The full model wires three calls to this encoder (one per RGB channel), concatenates outputs, applies the 1×1 fusion conv, GlobalMaxPooling2D, and the Dense classification head.

**Step 3 — Training**
Adam optimizer with `ReduceLROnPlateau` (halves learning rate when validation loss plateaus). 75 epochs. The learning rate schedule helps avoid local minima as the model fine-tunes its channel-specific texture detectors.

**Step 4 — Evaluation**
Confusion matrix and per-class accuracy on the 20% validation split. The confusion matrix reveals which disease pairs are most frequently confused — typically those with visually similar presentations (e.g., early-stage blight vs. mold on tomato).

**Step 5 — Grad-CAM visualization**
For randomly sampled test images, Grad-CAM computes the gradient of the class score with respect to the last convolutional feature map, producing a spatial heat map of model attention. Overlaid on the original image, this confirms whether the model has learned the right visual signatures.

**Step 6 — Export**
```python
model.export("saved_model_dir")
# Then in terminal:
# tensorflowjs_converter --input_format=tf_saved_model saved_model_dir static/apple/
```
This produces the `model.json` + `.bin` files hosted in `static/`.

### Running the Web Interface

```bash
# Serve locally
python -m http.server 8000

# Expose to mobile devices for real-device testing
ngrok http 8000
# Open the ngrok HTTPS URL on any phone browser
```

Open the URL, select a crop, take or upload a photo of a diseased leaf. Prediction and confidence appear within milliseconds to seconds depending on device hardware — all on-device.

---

## 12. Citation

```bibtex
@inproceedings{kethineni2025soil,
  title     = {SOIL: A Privacy-First Framework for Secure On-Device Inference of Leaf Diseases on Mobile},
  author    = {Kethineni, Kiran K. and Mohanty, Saraju P. and Kougianos, Elias},
  booktitle = {Proceedings of OCIT 2025},
  year      = {2025},
  institution = {University of North Texas}
}
```

**Authors**: Kiran K. Kethineni · Saraju P. Mohanty · Elias Kougianos
**Institution**: University of North Texas, USA
**Contact**: kirankumar.kethineni@unt.edu
