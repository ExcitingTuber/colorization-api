---
title: Image Colorization & HD Upscale
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🎨 Image Colorization & HD Restore

**AI-powered image restoration suite — colorize black & white photos and upscale to HD with state-of-the-art deep learning models.**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/VatsalRaina01/colorization-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)](https://gradio.app)

[Live Demo](https://huggingface.co/spaces/VatsalRaina01/colorization-api) · [Report Bug](https://github.com/VatsalRaina01/colorization-api/issues)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🎨 Colorize** | Convert grayscale / B&W photographs to full color using a deep neural network trained on millions of images |
| **🔍 HD Restore** | Upscale any image to 2× or 4× resolution with Real-ESRGAN — reconstructs textures, removes compression artifacts, and recovers fine detail |
| **🧑 Face Enhancement** | Optional GFPGAN-based face restoration that fixes blurry, low-res, or damaged faces with remarkable fidelity |
| **🚀 One-Click Pipeline** | Colorize *and* upscale a B&W photo in a single step — no manual chaining required |
| **📥 Download** | Download the full-resolution result as PNG directly from the UI |
| **🔀 Before / After Slider** | Interactive comparison slider: Bicubic (naïve) vs Real-ESRGAN (AI) so you can see the difference at a glance |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────┐
│                   Gradio UI (app.py)               │
│   ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│   │ Colorize │  │ HD Restore│  │ Colorize + HD │  │
│   └────┬─────┘  └─────┬─────┘  └───────┬───────┘  │
└────────┼───────────────┼────────────────┼──────────┘
         │               │                │
         ▼               │                ▼
   ┌───────────┐         │        ┌──────────────┐
   │colorize.py│         │        │  colorize.py  │
   │ Zhang '16 │         │        │  → upscale.py │
   └───────────┘         ▼        └──────────────┘
                  ┌─────────────┐
                  │ upscale.py  │
                  │ Real-ESRGAN │
                  │  + GFPGAN   │
                  └─────────────┘
```

---

## 🧠 Models

| Component | Model | Paper / Source | Role |
|-----------|-------|----------------|------|
| Colorization | OpenCV DNN (Caffe) | [Zhang et al., ECCV 2016](http://richzhang.github.io/colorization/) | Predicts chrominance (a, b) channels from luminance (L) using a CNN trained on 1.3M ImageNet images |
| Super-Resolution | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) x4plus | Blind real-world SR via RRDB architecture | Reconstructs high-frequency textures and removes compression artifacts — dramatically outperforms bicubic interpolation |
| Face Restoration | [GFPGAN](https://github.com/TencentARC/GFPGAN) v1.4 | GAN-based face restoration | Restores facial features with spatial feature transform layers and uses Real-ESRGAN as the background upsampler |

All model weights are **auto-downloaded on first run** — no manual setup required.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/VatsalRaina01/colorization-api.git
cd colorization-api

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

The Gradio UI will open at `http://localhost:7860`. Model weights (~200 MB) download automatically on the first run.

### Run on Hugging Face Spaces

The app is deployed and available at:
**[huggingface.co/spaces/VatsalRaina01/colorization-api](https://huggingface.co/spaces/VatsalRaina01/colorization-api)**

---

## 📂 Project Structure

```
colorization-api/
├── app.py              # Gradio UI — three-tab interface with slider comparisons
├── colorize.py         # Colorization model loader & inference (Zhang et al.)
├── upscale.py          # Real-ESRGAN + GFPGAN upscaling & face restoration
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes model weights & cache from version control
└── README.md
```

---

## ⚙️ How It Works

### Colorization Pipeline
1. Input RGB image is converted to **LAB** color space
2. The **L** (lightness) channel is extracted, resized to 224×224, and passed through the Caffe network
3. The network predicts the **a** and **b** chrominance channels
4. Predicted a, b channels are resized back to the original resolution and merged with the original L channel
5. LAB → BGR conversion produces the final colorized image

### HD Restore Pipeline
1. Input image is fed to **Real-ESRGAN x4plus** (RRDB architecture with 23 residual blocks)
2. Tiling (400×400 tiles) prevents GPU OOM on large images
3. If face enhancement is enabled, **GFPGAN v1.4** detects and restores faces, using Real-ESRGAN as the background upsampler
4. An interactive slider lets users compare the AI output against naïve bicubic upscaling

---

## 🛠️ Tech Stack

- **[Gradio](https://gradio.app)** — Web UI framework with image slider, tabs, and file download
- **[OpenCV](https://opencv.org)** — Image I/O and DNN inference (Caffe backend)
- **[PyTorch](https://pytorch.org)** — Backend for Real-ESRGAN and GFPGAN
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** — State-of-the-art blind super-resolution
- **[GFPGAN](https://github.com/TencentARC/GFPGAN)** — GAN-based face restoration

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/VatsalRaina01">Vatsal Raina</a></sub>
</div>
