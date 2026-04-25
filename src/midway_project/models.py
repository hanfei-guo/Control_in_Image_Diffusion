from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from huggingface_hub import snapshot_download

from .settings import MODEL_SPECS


def stabilize_hf_loading() -> None:
    # Windows + transformers async tensor materialization is intermittently unstable
    # for the IP-Adapter image encoder in fresh processes.
    if os.name == "nt":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def download_required_models() -> dict[str, Path]:
    resolved_paths: dict[str, Path] = {}
    for spec in MODEL_SPECS.values():
        local_dir = Path(spec["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=str(local_dir),
            allow_patterns=spec["allow_patterns"],
        )
        resolved_paths[spec["repo_id"]] = local_dir
    return resolved_paths


def optimize_pipeline(pipe, attention_slicing: bool = True):
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if attention_slicing:
        pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    return pipe


def build_controlnet_pipeline(
    base_model_dir: Path | str,
    controlnet_dir: Path | str,
    device: str,
    dtype: torch.dtype,
):
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_dir),
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = optimize_pipeline(pipe, attention_slicing=True)
    pipe.to(device)
    return pipe


def build_ip_adapter_pipeline(
    base_model_dir: Path | str,
    ip_adapter_dir: Path | str,
    device: str,
    dtype: torch.dtype,
    ip_adapter_scale: float,
):
    stabilize_hf_loading()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = optimize_pipeline(pipe, attention_slicing=False)
    pipe.load_ip_adapter(
        str(ip_adapter_dir),
        subfolder="models",
        weight_name="ip-adapter_sd15.safetensors",
        image_encoder_folder="image_encoder",
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    pipe.to(device)
    return pipe


def build_combined_pipeline(
    base_model_dir: Path | str,
    controlnet_dir: Path | str,
    ip_adapter_dir: Path | str,
    device: str,
    dtype: torch.dtype,
    ip_adapter_scale: float,
):
    stabilize_hf_loading()
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_dir),
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = optimize_pipeline(pipe, attention_slicing=False)
    pipe.load_ip_adapter(
        str(ip_adapter_dir),
        subfolder="models",
        weight_name="ip-adapter_sd15.safetensors",
        image_encoder_folder="image_encoder",
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    pipe.to(device)
    return pipe


def build_smooth_combined_pipeline(
    base_model_dir: Path | str,
    controlnet_dir: Path | str,
    ip_adapter_dir: Path | str,
    device: str,
    dtype: torch.dtype,
    ip_adapter_scale: float,
    control_segments: int,
):
    stabilize_hf_loading()
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
    )
    multi_controlnet = MultiControlNetModel([controlnet] * control_segments)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_dir),
        controlnet=multi_controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = optimize_pipeline(pipe, attention_slicing=False)
    pipe.load_ip_adapter(
        str(ip_adapter_dir),
        subfolder="models",
        weight_name="ip-adapter_sd15.safetensors",
        image_encoder_folder="image_encoder",
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    pipe.to(device)
    return pipe
