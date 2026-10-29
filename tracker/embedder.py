"""
embedder.py
-----------
Configurable person appearance feature extractor supporting:
- Pretrained OSNet (Omni-Scale Network, MSMT17-trained)
- Baseline ResNet50 (ImageNet-pretrained)
- Full-body embedding mode or multi-part horizontal strip embedding mode.

Features are guaranteed to be finite and L2-normalized.
Never generates random fallback embeddings. If extraction fails, returns an empty list.
"""

import sys
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker.config import (
    DEVICE,
    NUM_PARTS,
    REID_BACKBONE,
    USE_PART_EMBEDDING,
)
from tracker.utils import l2_normalize_vec
from tracker.osnet import osnet_x0_25, osnet_x1_0


class PartEmbedder:
    """
    Appearance feature extractor for person crops.

    Supports:
    - Backbone: "osnet_x0_25" (default), "osnet_x1_0", "resnet50"
    - Part mode: full-body (use_part_embedding=False) or multi-part horizontal strips.

    Attributes:
        available: True if the model successfully loaded.
        feature_dim: Dimensionality of extracted embeddings (512 for OSNet, 2048 for ResNet50).
    """

    def __init__(
        self,
        device: str = DEVICE,
        num_parts: int = NUM_PARTS,
        backbone: str = REID_BACKBONE,
        use_part_embedding: bool = USE_PART_EMBEDDING,
    ):
        self.device = device
        self.num_parts = num_parts if use_part_embedding else 1
        self.use_part_embedding = use_part_embedding
        self.backbone = backbone
        self.available = False
        self.feature_dim = 512

        try:
            if self.backbone == "resnet50":
                r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # 2048D feature pooled output
                self.model = nn.Sequential(*list(r50.children())[:-1]).to(self.device)
                self.feature_dim = 2048
            elif self.backbone == "osnet_x1_0":
                self.model = osnet_x1_0(pretrained=True).to(self.device)
                self.feature_dim = 512
            elif self.backbone in ("osnet", "osnet_x0_25"):
                self.model = osnet_x0_25(pretrained=True).to(self.device)
                self.feature_dim = 512
            else:
                raise ValueError(f"Unknown backbone: {self.backbone}")

            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((256, 128), antialias=True),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self.available = True
        except Exception as e:
            print(f"[PartEmbedder] Error initializing backbone '{self.backbone}': {e}")
            self.available = False

    def extract(self, frame: np.ndarray, box: List[float]) -> List[np.ndarray]:
        """
        Extract appearance embeddings for one detection box.

        Args:
            frame: Full BGR frame (H x W x 3).
            box:   Detection box [x1, y1, x2, y2].

        Returns:
            List of L2-normalized numpy arrays. Returns an empty list [] on invalid crops
            or extraction failure. Never returns random vectors.
        """
        if not self.available:
            return []

        if frame is None or frame.size == 0:
            return []

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)

            box_w = x2 - x1
            box_h = y2 - y1

            # Discard degenerate or tiny boxes (cannot extract reliable person appearance)
            if box_w < 8 or box_h < 8:
                return []

            crops: List[np.ndarray] = []
            if not self.use_part_embedding or self.num_parts <= 1:
                # Full-body crop
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
            else:
                # Multi-part horizontal strip slicing
                part_h = max(1, box_h // self.num_parts)
                for i in range(self.num_parts):
                    py1 = y1 + i * part_h
                    py2 = (y1 + (i + 1) * part_h) if i < self.num_parts - 1 else y2
                    crop = frame[py1:py2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)

            if not crops:
                return []

            batch_tensors = []
            for c in crops:
                c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(c_rgb))

            input_batch = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                feats = self.model(input_batch)
                feats = feats.view(feats.size(0), -1).cpu().numpy()

            out_embeddings = []
            for feat in feats:
                if feat.size == 0 or not np.isfinite(feat).all():
                    continue
                norm_feat = l2_normalize_vec(feat.astype(np.float32))
                # Validate finite values and correct dimensions
                if np.isfinite(norm_feat).all() and norm_feat.shape[0] == self.feature_dim:
                    out_embeddings.append(norm_feat)

            return out_embeddings

        except Exception as e:
            print(f"[PartEmbedder] extract error: {e}")
            return []

    def compute_weighted_mean(
        self,
        embeds: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute a weighted mean of multiple embeddings and L2-normalize.

        Args:
            embeds:  List of L2-normalized embeddings.
            weights: Optional weights (same length as embeds).

        Returns:
            Single L2-normalized embedding or None if embeds is empty.
        """
        if not embeds:
            return None

        # Verify all embeddings have consistent dimension
        dim = embeds[0].shape[0]
        valid_embeds = [e for e in embeds if e is not None and e.shape[0] == dim and np.isfinite(e).all()]
        if not valid_embeds:
            return None

        arr = np.stack(valid_embeds, axis=0)  # shape: (N, D)

        if weights is not None and len(weights) == len(valid_embeds):
            w = np.array(weights, dtype=np.float32)
            w_sum = w.sum()
            if w_sum > 1e-8:
                w /= w_sum
                mean = np.average(arr, axis=0, weights=w)
            else:
                mean = np.mean(arr, axis=0)
        else:
            mean = np.mean(arr, axis=0)

        if not np.isfinite(mean).all():
            return None

        return l2_normalize_vec(mean.astype(np.float32))
