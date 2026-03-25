"""
ArcFace embedding generator using ONNX Runtime (CPU only).
Loads a pre-trained ArcFace ONNX model and produces 512-d embeddings.
"""
import cv2
import numpy as np
import onnxruntime as ort

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ArcFaceONNX:
    """
    Wraps an ArcFace ONNX model (e.g. w600k_r50.onnx from InsightFace).
    Expected input: 112×112 BGR face crop, normalised to [-1, 1].
    Output: 512-d L2-normalised embedding.
    """

    def __init__(self, model_path: str = None):
        model_path = model_path or config.ARCFACE_MODEL_PATH
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at {model_path}.\n"
                f"Download from InsightFace model zoo and place it in the models/ folder."
            )

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        inp = self.session.get_inputs()[0]
        # Model expects (1, 3, 112, 112) or similar
        self.input_shape = tuple(inp.shape[2:4]) if len(inp.shape) == 4 else (112, 112)

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR face crop into model-ready tensor.
        Steps: resize → BGR→RGB → HWC→CHW → normalise → add batch dim.
        """
        img = cv2.resize(face_bgr, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # Normalise to [-1, 1]
        img = (img / 127.5) - 1.0
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dim
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Generate a 512-d L2-normalised embedding from a BGR face crop.
        """
        tensor = self.preprocess(face_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        emb = outputs[0].flatten().astype(np.float32)
        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def get_embeddings_batch(self, faces: list[np.ndarray]) -> list[np.ndarray]:
        """
        Batch inference. Currently sequential for CPU simplicity.
        """
        return [self.get_embedding(f) for f in faces]
