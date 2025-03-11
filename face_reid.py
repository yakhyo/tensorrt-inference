from models.arcface_onnx import ArcFace
from models.retinaface_onnx import RetinaFace

import numpy as np
import cv2

image1_path = "assets/1_01.jpg"
image2_path = "assets/2_02.jpg"

model = ArcFace(model_path="weights/w600k_mbf.onnx")
retina = RetinaFace(model_path="weights/retinaface_mv2_static.onnx")

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)


_, kpt1 = retina.detect(img1)
_, kpt2 = retina.detect(img2)


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """Computing Similarity between two faces.

    Args:
        feat1 (np.ndarray): Face features.
        feat2 (np.ndarray): Face features.

    Returns:
        np.float32: Cosine similarity between face features.
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity



feat1 = model(img1, kps=kpt1[:1][0])
feat2 = model(img2, kps=kpt2[:1][0])


print(compute_similarity(feat1, feat2))


