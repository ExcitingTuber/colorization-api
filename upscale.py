import cv2
import numpy as np
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

UPSCALE_MODELS = {
    2: {
        "filename": "EDSR_x2.pb",
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
    },
    4: {
        "filename": "EDSR_x4.pb",
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
    },
}

MAX_OUTPUT_PIXELS = 5_000_000  # ~5 megapixels max output


def ensure_file(local_path, url):
    """Download a file if it doesn't already exist, with error handling."""
    if not os.path.exists(local_path):
        logger.info(f"Downloading {os.path.basename(local_path)} from {url}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            size = os.path.getsize(local_path)
            logger.info(f"Downloaded {os.path.basename(local_path)} ({size:,} bytes)")
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            logger.error(f"Failed to download {os.path.basename(local_path)}: {e}")
            raise RuntimeError(f"Failed to download: {os.path.basename(local_path)}") from e
    else:
        logger.info(f"Found existing {os.path.basename(local_path)}")


_sr_models = {}


def get_sr_model(scale):
    """Load and cache a super-resolution model."""
    global _sr_models
    if scale not in _sr_models:
        info = UPSCALE_MODELS[scale]
        local_path = os.path.join(MODEL_DIR, info["filename"])
        ensure_file(local_path, info["url"])

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(local_path)
        sr.setModel("edsr", scale)
        _sr_models[scale] = sr
        logger.info(f"Loaded EDSR x{scale} model")
    return _sr_models[scale]


def upscale_image(img, scale=4):
    """
    Upscale an image using EDSR super-resolution.

    Args:
        img: BGR image (numpy array, uint8)
        scale: 2 or 4

    Returns:
        Upscaled BGR image (numpy array, uint8)
    """
    if scale not in UPSCALE_MODELS:
        raise ValueError(f"Unsupported scale: {scale}. Use 2 or 4.")

    h, w = img.shape[:2]
    output_pixels = h * w * (scale ** 2)

    # If output would be too large, resize input first to prevent OOM
    if output_pixels > MAX_OUTPUT_PIXELS:
        max_input_pixels = MAX_OUTPUT_PIXELS / (scale ** 2)
        ratio = (max_input_pixels / (h * w)) ** 0.5
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized input from {w}x{h} to {new_w}x{new_h} before {scale}x upscale")
        h, w = new_h, new_w

    sr = get_sr_model(scale)
    result = sr.upsample(img)

    out_h, out_w = result.shape[:2]
    logger.info(f"Upscaled: {w}x{h} -> {out_w}x{out_h}")
    return result
