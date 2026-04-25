from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from midway_project.callbacks import IPAdapterScaleEnableCallback
from midway_project.experiments import (
    build_conflict_manifest,
    build_search_summary,
    evaluate_outputs,
    generator_for_row,
    load_manifest,
    sample_manifest,
)
from midway_project.models import build_combined_pipeline, default_torch_dtype, detect_device
from midway_project.reporting import save_metrics
from midway_project.settings import (
    BASE_MODEL_DIR,
    COMBINED_OUTPUT_DIR,
    CONTROLNET_MODEL_DIR,
    DEFAULT_TAU_CANDIDATES,
    GenerationConfig,
    IP_ADAPTER_MODEL_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run naive and tau-scheduled combined-control experiments.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=10623)
    parser.add_argument("--pairing", type=str, choices=("same", "conflict"), default="conflict")
    parser.add_argument("--pairing-seed", type=int, default=10624)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=10623)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--taus", type=str, default=",".join(str(value) for value in DEFAULT_TAU_CANDIDATES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="search_100_conflict")
    return parser


def parse_taus(taus: str) -> list[float]:
    values = [float(token.strip()) for token in taus.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one tau value is required.")
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Tau must be between 0 and 1, got {value}.")
    return values


def ensure_output_dirs(root: Path, tau_values: list[float]) -> dict[str, Path]:
    output_dirs = {"naive_combined": root / "naive_combined" / "images"}
    for tau in tau_values:
        mode = f"tau_{str(tau).replace('.', 'p')}"
        output_dirs[mode] = root / mode / "images"
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def generate_combined_variants(
    manifest: pd.DataFrame,
    output_dirs: dict[str, Path],
    tau_values: list[float],
    cfg: GenerationConfig,
    device: str,
    resume: bool,
) -> None:
    pipe = build_combined_pipeline(
        BASE_MODEL_DIR,
        CONTROLNET_MODEL_DIR,
        IP_ADAPTER_MODEL_DIR,
        device,
        default_torch_dtype(device),
        cfg.ip_adapter_scale,
    )
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating combined variants"):
        edge = Image.open(row.edge_path).convert("L")
        source = Image.open(row.image_path).convert("RGB")

        naive_path = output_dirs["naive_combined"] / f"{row.sample_id}.png"
        if not (naive_path.exists() and resume):
            pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
            result = pipe(
                prompt=row.caption,
                negative_prompt=cfg.negative_prompt,
                image=edge,
                ip_adapter_image=source,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
                generator=generator_for_row(device, cfg.seed, row_idx),
                width=cfg.image_size,
                height=cfg.image_size,
            ).images[0]
            result.save(naive_path)

        for tau in tau_values:
            mode = f"tau_{str(tau).replace('.', 'p')}"
            output_path = output_dirs[mode] / f"{row.sample_id}.png"
            if output_path.exists() and resume:
                continue

            pipe.set_ip_adapter_scale(0.0)
            callback = IPAdapterScaleEnableCallback(cutoff_step_ratio=tau, scale=cfg.ip_adapter_scale)
            result = pipe(
                prompt=row.caption,
                negative_prompt=cfg.negative_prompt,
                image=edge,
                ip_adapter_image=source,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
                control_guidance_end=tau,
                generator=generator_for_row(device, cfg.seed + int(tau * 10_000), row_idx),
                width=cfg.image_size,
                height=cfg.image_size,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=[],
            ).images[0]
            result.save(output_path)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    tau_values = parse_taus(args.taus)
    full_manifest = load_manifest(args.manifest)
    if args.pairing == "conflict":
        manifest = build_conflict_manifest(full_manifest, args.sample_size, args.sample_seed, args.pairing_seed)
    else:
        manifest = sample_manifest(full_manifest, args.sample_size, args.sample_seed)

    experiment_root = COMBINED_OUTPUT_DIR / args.experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    config_path = experiment_root / "experiment_config.json"
    config_payload = {
        "pairing": args.pairing,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "pairing_seed": args.pairing_seed,
        "taus": tau_values,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "ip_adapter_scale": args.ip_adapter_scale,
    }
    if config_path.exists():
        existing_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_payload != config_payload:
            raise ValueError(
                f"Experiment config mismatch in {config_path}. Use a new --experiment-name or remove the old directory."
            )
    else:
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    manifest.to_csv(experiment_root / "experiment_manifest.csv", index=False)
    output_dirs = ensure_output_dirs(experiment_root, tau_values)

    device = args.device or detect_device()
    cfg = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
    )

    generate_combined_variants(manifest, output_dirs, tau_values, cfg, device, args.resume)
    metrics = evaluate_outputs(manifest, output_dirs, device)
    save_metrics(metrics, experiment_root / "per_sample_metrics.csv", experiment_root / "summary.json")

    search_summary = build_search_summary(metrics)
    search_summary.to_csv(experiment_root / "search_summary.csv", index=False)
    print(search_summary.to_string(index=False))
    print("Combined experiments completed.")


if __name__ == "__main__":
    main()
