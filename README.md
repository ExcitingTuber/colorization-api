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

# Image Colorization & HD Upscale

Upload a black & white image and colorize it using a deep learning model, then upscale it to HD resolution with AI super-resolution.

## Features

- **🎨 Colorize** — Convert B&W photos to color using Zhang et al. (2016)
- **🔍 HD Upscale** — Enhance any image to 2x or 4x resolution using EDSR super-resolution
- **🚀 Colorize + HD** — One-click pipeline: colorize a B&W image and upscale to HD in one step

## Models

| Feature | Model | Paper |
|---|---|---|
| Colorization | OpenCV DNN (Caffe) | Zhang et al., 2016 |
| Super-Resolution | EDSR | Lim et al., 2017 |

## Usage

```bash
pip install -r requirements.txt
python app.py
```
