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


def enhance_image(img):
    """
    Post-processing pipeline to sharpen and enhance the upscaled image.
    Applies edge-preserving denoising, unsharp masking, and adaptive contrast.
    """
    # Step 1: Bilateral filter — removes noise while keeping edges crisp
    denoised = cv2.bilateralFilter(img, 5, 50, 50)

    # Step 2: Unsharp masking — sharpens edges and details
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.8, gaussian, -0.8, 0)

    # Step 3: CLAHE on L channel — adaptive contrast enhancement
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return np.clip(result, 0, 255).astype("uint8")


def _upscale_single_pass(img, scale):
    """Run a single EDSR upscale pass."""
    sr = get_sr_model(scale)
    return sr.upsample(img)


def upscale_image(img, scale=4):
    """
    Upscale an image using EDSR super-resolution with enhancement.

    For 4x: runs two passes of x2 (produces significantly sharper results
    than a single x4 pass because the model refines details incrementally).

    Finishes with a post-processing enhancement pipeline (denoise + sharpen + contrast).

    Args:
        img: BGR image (numpy array, uint8)
        scale: 2 or 4

    Returns:
        Upscaled and enhanced BGR image (numpy array, uint8)
    """
    if scale not in (2, 4):
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

    if scale == 4:
        # Two-pass x2 produces much sharper results than single x4
        logger.info(f"Upscaling {w}x{h} with two-pass x2 (effective x4)...")
        img = _upscale_single_pass(img, 2)
        mid_h, mid_w = img.shape[:2]
        logger.info(f"  Pass 1: {w}x{h} -> {mid_w}x{mid_h}")
        img = _upscale_single_pass(img, 2)
        out_h, out_w = img.shape[:2]
        logger.info(f"  Pass 2: {mid_w}x{mid_h} -> {out_w}x{out_h}")
    else:
        img = _upscale_single_pass(img, 2)
        out_h, out_w = img.shape[:2]
        logger.info(f"Upscaled: {w}x{h} -> {out_w}x{out_h}")

    # Post-processing enhancement
    logger.info("Applying enhancement pipeline (denoise + sharpen + contrast)...")
    result = enhance_image(img)

    return result
