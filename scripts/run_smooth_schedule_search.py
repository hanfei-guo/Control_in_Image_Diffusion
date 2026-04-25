from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from midway_project.callbacks import DynamicIPAdapterScaleCallback
from midway_project.experiments import (
    build_conflict_manifest,
    build_search_summary,
    evaluate_outputs,
    generator_for_row,
    load_manifest,
    sample_manifest,
)
from midway_project.models import build_smooth_combined_pipeline, default_torch_dtype, detect_device
from midway_project.reporting import save_metrics
from midway_project.schedules import SmoothScheduleConfig, build_mode_name, make_control_staircase, stage_best
from midway_project.settings import (
    BASE_MODEL_DIR,
    COMBINED_OUTPUT_DIR,
    CONTROLNET_MODEL_DIR,
    DEFAULT_SMOOTH_CONTROL_SCALE_CANDIDATES,
    DEFAULT_SMOOTH_CONTROL_SEGMENTS,
    DEFAULT_SMOOTH_IP_SCALE_CANDIDATES,
    DEFAULT_SMOOTH_SHARPNESS_CANDIDATES,
    DEFAULT_SMOOTH_TAU_CANDIDATES,
    GenerationConfig,
    IP_ADAPTER_MODEL_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a progressive smooth-schedule search without replacing hard-switch results.")
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
    parser.add_argument("--control-segments", type=int, default=DEFAULT_SMOOTH_CONTROL_SEGMENTS)
    parser.add_argument("--taus", type=str, default=",".join(str(value) for value in DEFAULT_SMOOTH_TAU_CANDIDATES))
    parser.add_argument(
        "--sharpness-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SMOOTH_SHARPNESS_CANDIDATES),
    )
    parser.add_argument(
        "--ip-scale-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SMOOTH_IP_SCALE_CANDIDATES),
    )
    parser.add_argument(
        "--control-scale-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SMOOTH_CONTROL_SCALE_CANDIDATES),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="search_100_conflict_smooth")
    return parser


def parse_values(values: str) -> list[float]:
    parsed = [float(token.strip()) for token in values.split(",") if token.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric candidate.")
    return parsed


def ensure_output_dirs(root: Path, modes: list[str]) -> dict[str, Path]:
    output_dirs: dict[str, Path] = {}
    for mode in modes:
        output_path = root / mode / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        output_dirs[mode] = output_path
    return output_dirs


def save_config(root: Path, payload: dict) -> None:
    config_path = root / "experiment_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Experiment config mismatch in {config_path}. Use a new --experiment-name or remove the old directory.")
    else:
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarise_stage(
    metrics: pd.DataFrame,
    search_table: pd.DataFrame,
    stage: str,
    mode_order: list[str],
) -> pd.DataFrame:
    summary = build_search_summary(metrics.loc[metrics["mode"].isin(mode_order)].copy())
    merged = search_table.merge(summary, on="mode", how="left")
    merged["stage"] = stage
    return merged.sort_values("balanced_score", ascending=False).reset_index(drop=True)


def generate_modes(
    manifest: pd.DataFrame,
    output_dirs: dict[str, Path],
    mode_configs: list[dict],
    cfg: GenerationConfig,
    device: str,
    resume: bool,
    control_segments: int,
) -> None:
    pipe = build_smooth_combined_pipeline(
        BASE_MODEL_DIR,
        CONTROLNET_MODEL_DIR,
        IP_ADAPTER_MODEL_DIR,
        device,
        default_torch_dtype(device),
        cfg.ip_adapter_scale,
        control_segments,
    )
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating smooth schedule variants"):
        edge = Image.open(row.edge_path).convert("L")
        edge_batch = [edge] * control_segments
        source = Image.open(row.image_path).convert("RGB")

        for mode_cfg in mode_configs:
            mode = mode_cfg["mode"]
            output_path = output_dirs[mode] / f"{row.sample_id}.png"
            if output_path.exists() and resume:
                continue

            params = SmoothScheduleConfig(
                tau=mode_cfg["tau"],
                sharpness=mode_cfg["sharpness"],
                control_max_scale=mode_cfg["control_max_scale"],
                ip_max_scale=mode_cfg["ip_max_scale"],
            )
            control_scales, control_guidance_end = make_control_staircase(
                params.tau,
                params.sharpness,
                params.control_max_scale,
                control_segments,
            )
            callback = DynamicIPAdapterScaleCallback(params.tau, params.sharpness, params.ip_max_scale)
            pipe.set_ip_adapter_scale(0.0)
            mode_seed_offset = int(hashlib.sha1(mode.encode("utf-8")).hexdigest()[:8], 16)
            result = pipe(
                prompt=row.caption,
                negative_prompt=cfg.negative_prompt,
                image=edge_batch,
                ip_adapter_image=source,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                controlnet_conditioning_scale=control_scales,
                control_guidance_start=[0.0] * control_segments,
                control_guidance_end=control_guidance_end,
                generator=generator_for_row(device, cfg.seed + mode_seed_offset, row_idx),
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
    tau_candidates = parse_values(args.taus)
    sharpness_candidates = parse_values(args.sharpness_candidates)
    ip_scale_candidates = parse_values(args.ip_scale_candidates)
    control_scale_candidates = parse_values(args.control_scale_candidates)

    full_manifest = load_manifest(args.manifest)
    if args.pairing == "conflict":
        manifest = build_conflict_manifest(full_manifest, args.sample_size, args.sample_seed, args.pairing_seed)
    else:
        manifest = sample_manifest(full_manifest, args.sample_size, args.sample_seed)

    experiment_root = COMBINED_OUTPUT_DIR / args.experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(experiment_root / "experiment_manifest.csv", index=False)
    config_payload = {
        "pairing": args.pairing,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "pairing_seed": args.pairing_seed,
        "taus": tau_candidates,
        "sharpness_candidates": sharpness_candidates,
        "ip_scale_candidates": ip_scale_candidates,
        "control_scale_candidates": control_scale_candidates,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "ip_adapter_scale": args.ip_adapter_scale,
        "control_segments": args.control_segments,
        "search_strategy": "progressive_coordinate_search",
    }
    save_config(experiment_root, config_payload)

    device = args.device or detect_device()
    cfg = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
    )

    stage_records: list[dict] = []
    tested_modes: list[str] = []
    cumulative_metrics = pd.DataFrame()
    cumulative_summary_frames: list[pd.DataFrame] = []

    def run_stage(stage: str, candidates: list[dict]) -> pd.DataFrame:
        nonlocal cumulative_metrics
        modes = [candidate["mode"] for candidate in candidates]
        output_dirs = ensure_output_dirs(experiment_root, modes)
        generate_modes(manifest, output_dirs, candidates, cfg, device, args.resume, args.control_segments)
        metrics = evaluate_outputs(manifest, output_dirs, device)
        cumulative_metrics = pd.concat([cumulative_metrics, metrics], ignore_index=True)
        summary_frame = summarise_stage(metrics, pd.DataFrame(candidates), stage, modes)
        cumulative_summary_frames.append(summary_frame)
        return summary_frame

    # Stage 1: search tau with default sharpness and max scales fixed.
    tau_stage = [
        {
            "mode": build_mode_name(
                "smooth_tau",
                {"tau": tau},
            ),
            "tau": tau,
            "sharpness": sharpness_candidates[1] if len(sharpness_candidates) > 1 else sharpness_candidates[0],
            "control_max_scale": args.controlnet_scale,
            "ip_max_scale": args.ip_adapter_scale,
        }
        for tau in tau_candidates
    ]
    tau_summary = run_stage("tau", tau_stage)
    best = stage_best(tau_summary)

    # Stage 2: fix tau and search sharpness.
    sharp_stage = [
        {
            "mode": build_mode_name(
                "smooth_sharpness",
                {"tau": float(best["tau"]), "sharp": sharpness},
            ),
            "tau": float(best["tau"]),
            "sharpness": sharpness,
            "control_max_scale": args.controlnet_scale,
            "ip_max_scale": args.ip_adapter_scale,
        }
        for sharpness in sharpness_candidates
    ]
    sharp_summary = run_stage("sharpness", sharp_stage)
    best = stage_best(sharp_summary)

    # Stage 3: fix tau and sharpness, search IP max scale.
    ip_stage = [
        {
            "mode": build_mode_name(
                "smooth_ipmax",
                {"tau": float(best["tau"]), "sharp": float(best["sharpness"]), "ip": ip_scale},
            ),
            "tau": float(best["tau"]),
            "sharpness": float(best["sharpness"]),
            "control_max_scale": args.controlnet_scale,
            "ip_max_scale": ip_scale,
        }
        for ip_scale in ip_scale_candidates
    ]
    ip_summary = run_stage("ip_max_scale", ip_stage)
    best = stage_best(ip_summary)

    # Stage 4: fix tau, sharpness, and ip max scale, then search control max scale.
    control_stage = [
        {
            "mode": build_mode_name(
                "smooth_ctrlmax",
                {
                    "tau": float(best["tau"]),
                    "sharp": float(best["sharpness"]),
                    "ip": float(best["ip_max_scale"]),
                    "ctrl": control_scale,
                },
            ),
            "tau": float(best["tau"]),
            "sharpness": float(best["sharpness"]),
            "control_max_scale": control_scale,
            "ip_max_scale": float(best["ip_max_scale"]),
        }
        for control_scale in control_scale_candidates
    ]
    control_summary = run_stage("control_max_scale", control_stage)
    best = stage_best(control_summary)

    cumulative_metrics = cumulative_metrics.drop_duplicates(subset=["sample_id", "mode"]).reset_index(drop=True)
    save_metrics(cumulative_metrics, experiment_root / "per_sample_metrics.csv", experiment_root / "summary.json")

    search_trace = pd.concat(cumulative_summary_frames, ignore_index=True)
    search_trace.to_csv(experiment_root / "progressive_search_trace.csv", index=False)

    final_summary = build_search_summary(cumulative_metrics)
    params_table = search_trace.drop_duplicates(subset=["mode"])[
        ["mode", "stage", "tau", "sharpness", "control_max_scale", "ip_max_scale"]
    ]
    final_summary = params_table.merge(final_summary, on="mode", how="left")
    final_summary = final_summary.sort_values("balanced_score", ascending=False).reset_index(drop=True)
    final_summary.to_csv(experiment_root / "search_summary.csv", index=False)
    best_payload = {key: (value.item() if hasattr(value, "item") else value) for key, value in best.to_dict().items()}
    (experiment_root / "best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(final_summary.to_string(index=False))
    print("Smooth schedule search completed.")


if __name__ == "__main__":
    main()
