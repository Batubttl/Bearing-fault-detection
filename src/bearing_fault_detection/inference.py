"""Load the ONNX model and classify one vibration window."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .cwru import CLASS_NAMES
from .preprocessing import WINDOW_SIZE, normalize

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "cnn.onnx"


def preprocess(window) -> np.ndarray:
    """Same normalization as in training -> shape (1, 1, WINDOW_SIZE)."""
    return normalize(window).reshape(1, 1, -1)


class BearingClassifier:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def predict(self, window) -> dict:
        logits = self.session.run(None, {"signal": preprocess(window)})[0][0]
        exp = np.exp(logits - logits.max())          # softmax: scores -> probabilities
        probs = exp / exp.sum()
        idx = int(probs.argmax())
        return {
            "class_id": idx,
            "label": CLASS_NAMES[idx],
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        }