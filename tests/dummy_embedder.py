from typing import List

import numpy as np


class ColorHashEmbedder:
    """Deterministic test embedder: hashes crop mean color into a unit vector."""

    def __init__(self, dim: int = 16):
        self.available = True
        self.feature_dim = dim
        self.use_part_embedding = False

    def extract(self, frame: np.ndarray, box: List[float]) -> List[np.ndarray]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return []
        crop = frame[y1:y2, x1:x2]
        mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        vec[:3] = mean / 255.0
        vec[3] = 1.0
        vec /= np.linalg.norm(vec) + 1e-8
        return [vec]
