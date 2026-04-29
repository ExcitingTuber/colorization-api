import cv2
import numpy as np
import os
import urllib.request

FILES = {
    "prototxt": ("colorization_deploy_v2.prototxt",
                 "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt"),
    "caffemodel": ("colorization_release_v2.caffemodel",
                   "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"),
    "pts": ("pts_in_hull.npy",
            "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"),
}

def ensure_file(local_name, url):
    if not os.path.exists(local_name):
        urllib.request.urlretrieve(url, local_name)

def load_model():
    for _, (fname, url) in FILES.items():
        ensure_file(fname, url)

    net = cv2.dnn.readNetFromCaffe(FILES["prototxt"][0], FILES["caffemodel"][0])
    pts = np.load(FILES["pts"][0])

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

net = load_model()

def colorize_image(img):
    h, w = img.shape[:2]
    img_float = img.astype("float16") / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (128, 128))
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
    del lab, resized, ab
