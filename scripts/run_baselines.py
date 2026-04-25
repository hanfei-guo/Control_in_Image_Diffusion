from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from midway_project.metrics import ClipSimilarityScorer, canny_mse
from midway_project.models import (
    build_controlnet_pipeline,
    build_ip_adapter_pipeline,
    default_torch_dtype,
    detect_device,
)
from midway_project.reporting import save_metrics
from midway_project.settings import (
    BASE_MODEL_DIR,
    CLIP_MODEL_DIR,
    CONTROLNET_MODEL_DIR,
    IP_ADAPTER_MODEL_DIR,
    METRICS_DIR,
    MIDWAY_OUTPUT_DIR,
    GenerationConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run midway baseline generation and evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=10623)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    return parser


def load_manifest(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    frame["sample_id"] = frame["sample_id"].str.zfill(12)
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def ensure_output_dirs() -> dict[str, Path]:
    control_dir = MIDWAY_OUTPUT_DIR / "controlnet" / "images"
    ip_dir = MIDWAY_OUTPUT_DIR / "ip_adapter" / "images"
    control_dir.mkdir(parents=True, exist_ok=True)
    ip_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return {"controlnet": control_dir, "ip_adapter": ip_dir}


def generator_for_row(device: str, base_seed: int, row_index: int) -> torch.Generator:
    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(base_seed + row_index)
    return torch.Generator().manual_seed(base_seed + row_index)


def generate_controlnet(
    manifest: pd.DataFrame,
    output_dir: Path,
    cfg: GenerationConfig,
    device: str,
    resume: bool,
) -> None:
    pipe = build_controlnet_pipeline(BASE_MODEL_DIR, CONTROLNET_MODEL_DIR, device, default_torch_dtype(device))
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating ControlNet baseline"):
        output_path = output_dir / f"{row.sample_id}.png"
        if output_path.exists() and resume:
            continue
        edge = Image.open(row.edge_path).convert("L")
        result = pipe(
            prompt=row.caption,
            negative_prompt=cfg.negative_prompt,
            image=edge,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
            generator=generator_for_row(device, cfg.seed, row_idx),
            width=cfg.image_size,
            height=cfg.image_size,
        ).images[0]
        result.save(output_path)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_ip_adapter(
    manifest: pd.DataFrame,
    output_dir: Path,
    cfg: GenerationConfig,
    device: str,
    resume: bool,
) -> None:
    pipe = build_ip_adapter_pipeline(
        BASE_MODEL_DIR,
        IP_ADAPTER_MODEL_DIR,
        device,
        default_torch_dtype(device),
        cfg.ip_adapter_scale,
    )
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating IP-Adapter baseline"):
        output_path = output_dir / f"{row.sample_id}.png"
        if output_path.exists() and resume:
            continue
        source = Image.open(row.image_path).convert("RGB")
        result = pipe(
            prompt=row.caption,
            negative_prompt=cfg.negative_prompt,
            ip_adapter_image=source,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator_for_row(device, cfg.seed + 100_000, row_idx),
            width=cfg.image_size,
            height=cfg.image_size,
        ).images[0]
        result.save(output_path)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_outputs(manifest: pd.DataFrame, output_dirs: dict[str, Path], device: str) -> pd.DataFrame:
    scorer = ClipSimilarityScorer(CLIP_MODEL_DIR, device=device)
    records: list[dict] = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Evaluating metrics"):
        target_image = Image.open(row.image_path).convert("RGB")
        target_edge = Image.open(row.edge_path).convert("L")
        for mode, output_dir in output_dirs.items():
            generated_path = output_dir / f"{row.sample_id}.png"
            generated = Image.open(generated_path).convert("RGB")
            records.append(
                {
                    "sample_id": row.sample_id,
                    "mode": mode,
                    "prompt": row.caption,
                    "source_image_path": row.image_path,
                    "edge_path": row.edge_path,
                    "generated_path": str(generated_path),
                    "canny_mse": canny_mse(generated, target_edge),
                    "clip_similarity": scorer.score(generated, target_image),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest, args.limit)
    output_dirs = ensure_output_dirs()
    device = args.device or detect_device()
    cfg = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
    )

    generate_controlnet(manifest, output_dirs["controlnet"], cfg, device, args.resume)
    generate_ip_adapter(manifest, output_dirs["ip_adapter"], cfg, device, args.resume)
    metrics = evaluate_outputs(manifest, output_dirs, device)
    save_metrics(metrics, METRICS_DIR / "per_sample_metrics.csv", METRICS_DIR / "summary.json")
    print("Baseline generation and evaluation completed.")


if __name__ == "__main__":
    main()
