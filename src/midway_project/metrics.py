from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_gray_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def canny_mse(
    generated_image: Image.Image,
    target_edge_image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> float:
    generated_gray = cv2.cvtColor(np.array(generated_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    generated_edges = cv2.Canny(generated_gray, low_threshold, high_threshold).astype(np.float32) / 255.0
    target_edges = np.array(target_edge_image.convert("L"), dtype=np.float32) / 255.0
    return float(np.mean((generated_edges - target_edges) ** 2))


class ClipSimilarityScorer:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, image_a: Image.Image, image_b: Image.Image) -> float:
        inputs = self.processor(
            images=[image_a.convert("RGB"), image_b.convert("RGB")],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        raw_features = self.model.get_image_features(**inputs)
        if isinstance(raw_features, torch.Tensor):
            features = raw_features
        elif hasattr(raw_features, "image_embeds") and raw_features.image_embeds is not None:
            features = raw_features.image_embeds
        elif hasattr(raw_features, "pooler_output"):
            features = raw_features.pooler_output
        else:
            raise TypeError(f"Unsupported CLIP output type: {type(raw_features)!r}")
        features = torch.nn.functional.normalize(features, dim=-1)
        return float(torch.sum(features[0] * features[1]).item())
