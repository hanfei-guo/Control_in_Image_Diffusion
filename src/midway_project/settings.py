from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = ASSETS_DIR / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

COCO_SUBSET_DIR = DATA_DIR / "coco2017_midway"
COCO_IMAGES_DIR = COCO_SUBSET_DIR / "images"
COCO_EDGES_DIR = COCO_SUBSET_DIR / "edges"
COCO_MANIFEST_PATH = COCO_SUBSET_DIR / "subset_manifest.csv"

MIDWAY_OUTPUT_DIR = OUTPUTS_DIR / "midway_baselines"
METRICS_DIR = MIDWAY_OUTPUT_DIR / "metrics"
COMBINED_OUTPUT_DIR = OUTPUTS_DIR / "combined_experiments"

BASE_MODEL_DIR = MODELS_DIR / "stable-diffusion-v1-5"
CONTROLNET_MODEL_DIR = MODELS_DIR / "control_v11p_sd15_canny"
IP_ADAPTER_MODEL_DIR = MODELS_DIR / "ip-adapter-sd15"
CLIP_MODEL_DIR = MODELS_DIR / "clip-vit-base-patch32"

DEFAULT_SEED = 10623
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DATASET_SIZE = 1000
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, distorted, deformed, cropped, extra limbs"
)
DEFAULT_TAU_CANDIDATES = (0.25, 0.4, 0.5, 0.6, 0.75)
DEFAULT_SMOOTH_TAU_CANDIDATES = (0.25, 0.4, 0.5, 0.6, 0.75)
DEFAULT_SMOOTH_SHARPNESS_CANDIDATES = (8.0, 12.0, 16.0)
DEFAULT_SMOOTH_IP_SCALE_CANDIDATES = (0.6, 0.8, 1.0)
DEFAULT_SMOOTH_CONTROL_SCALE_CANDIDATES = (0.8, 1.0, 1.2)
DEFAULT_SMOOTH_CONTROL_SEGMENTS = 6


@dataclass(slots=True)
class GenerationConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    ip_adapter_scale: float = 0.8
    seed: int = DEFAULT_SEED
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


MODEL_SPECS = {
    "stable-diffusion-v1-5": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "local_dir": BASE_MODEL_DIR,
        "allow_patterns": [
            "model_index.json",
            "feature_extractor/preprocessor_config.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
        ],
    },
    "controlnet-canny": {
        "repo_id": "lllyasviel/control_v11p_sd15_canny",
        "local_dir": CONTROLNET_MODEL_DIR,
        "allow_patterns": [
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    "ip-adapter-sd15": {
        "repo_id": "h94/IP-Adapter",
        "local_dir": IP_ADAPTER_MODEL_DIR,
        "allow_patterns": [
            "models/ip-adapter_sd15.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ],
    },
    "clip-vit-base-patch32": {
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": CLIP_MODEL_DIR,
        "allow_patterns": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    },
}
