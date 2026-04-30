import cv2
import numpy as np
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Fix basicsr/torchvision compatibility ---
# basicsr 1.4.2 imports from torchvision.transforms.functional_tensor
# which was removed in newer torchvision. Patch it before importing.
import torchvision.transforms.functional as _F
if not hasattr(sys.modules.get("torchvision.transforms", None), "functional_tensor"):
    import types
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
    logger.info("Patched torchvision.transforms.functional_tensor shim")

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Real-ESRGAN will use device: {DEVICE}")

# --- Lazy-loaded singletons ---
_upsampler = None
_face_enhancer = None


def get_upsampler():
    """Load Real-ESRGAN x4plus model (lazy, cached)."""
    global _upsampler
    if _upsampler is None:
        logger.info("Loading Real-ESRGAN x4plus model...")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.1.0/RealESRGAN_x4plus.pth"
        )
        _upsampler = RealESRGANer(
            scale=4,
            model_path=model_url,
            model=model,
            tile=400,       # tile to avoid OOM on large images
            tile_pad=10,
            pre_pad=0,
            half=False,     # fp32 for CPU compatibility
            device=DEVICE,
        )
        logger.info("Real-ESRGAN x4plus loaded.")
    return _upsampler


def get_face_enhancer():
    """Load GFPGAN v1.4 face enhancer (lazy, cached)."""
    global _face_enhancer
    if _face_enhancer is None:
        from gfpgan import GFPGANer

        logger.info("Loading GFPGAN v1.4 face enhancer...")
        model_url = (
            "https://github.com/TencentARC/GFPGAN/releases/"
            "download/v1.3.0/GFPGANv1.4.pth"
        )
        _face_enhancer = GFPGANer(
            model_path=model_url,
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=get_upsampler(),
        )
        logger.info("GFPGAN v1.4 loaded.")
    return _face_enhancer


def upscale_image(img, scale=4, enhance_faces=False):
    """
    Upscale and restore an image using Real-ESRGAN.

    Unlike simple interpolation or old SR models, Real-ESRGAN reconstructs
    textures, removes compression artifacts, and produces dramatically
    clearer output.

    Args:
        img: BGR image (numpy array, uint8)
        scale: output scale factor (2 or 4)
        enhance_faces: if True, also restore faces with GFPGAN

    Returns:
        Enhanced BGR image (numpy array, uint8)
    """
    if enhance_faces:
        logger.info(f"Enhancing with GFPGAN + Real-ESRGAN (x{scale})...")
        enhancer = get_face_enhancer()
        # GFPGAN returns: cropped_faces, restored_faces, restored_img
        _, _, output = enhancer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        logger.info(f"Enhancing with Real-ESRGAN (x{scale})...")
        upsampler = get_upsampler()
        output, _ = upsampler.enhance(img, outscale=scale)

    h_in, w_in = img.shape[:2]
    h_out, w_out = output.shape[:2]
    logger.info(f"Enhanced: {w_in}x{h_in} -> {w_out}x{h_out}")
    return output
