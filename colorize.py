import cv2
import numpy as np
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the script's directory as the base for model files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "prototxt": ("colorization_deploy_v2.prototxt",
                 "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt"),
    "caffemodel": ("colorization_release_v2.caffemodel",
                   "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"),
    "pts": ("pts_in_hull.npy",
            "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"),
}

def ensure_file(local_path, url):
    """Download a file if it doesn't already exist, with error handling."""
    if not os.path.exists(local_path):
        logger.info(f"Downloading {os.path.basename(local_path)} from {url}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            logger.info(f"Downloaded {os.path.basename(local_path)} ({os.path.getsize(local_path)} bytes)")
        except Exception as e:
            # Clean up partial downloads
            if os.path.exists(local_path):
                os.remove(local_path)
            logger.error(f"Failed to download {os.path.basename(local_path)}: {e}")
            raise RuntimeError(f"Failed to download model file: {os.path.basename(local_path)}") from e
    else:
        logger.info(f"Found existing {os.path.basename(local_path)}")

def load_model():
    """Load the colorization model, downloading files if needed."""
    for _, (fname, url) in FILES.items():
        local_path = os.path.join(MODEL_DIR, fname)
        ensure_file(local_path, url)

    prototxt_path = os.path.join(MODEL_DIR, FILES["prototxt"][0])
    caffemodel_path = os.path.join(MODEL_DIR, FILES["caffemodel"][0])
    pts_path = os.path.join(MODEL_DIR, FILES["pts"][0])

    logger.info("Loading Caffe model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    pts = np.load(pts_path)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    logger.info("Model loaded successfully!")
    return net

# Lazy loading: only load when first needed, not at import time
_net = None

def get_model():
    """Get the model, loading it lazily on first call."""
    global _net
    if _net is None:
        _net = load_model()
    return _net

def colorize_image(img):
    net = get_model()
    h, w = img.shape[:2]
    img_float = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = resized[:, :, 0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h))

    L_orig = lab[:, :, 0]
    color_lab = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    color_bgr = cv2.cvtColor(color_lab, cv2.COLOR_LAB2BGR)
    color_bgr = np.clip(color_bgr, 0, 1)

    return (color_bgr * 255).astype("uint8")
