from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .callbacks import DynamicIPAdapterScaleCallback, IPAdapterScaleEnableCallback
from .experiments import build_search_summary, evaluate_outputs, generator_for_row
from .models import (
    build_combined_pipeline,
    build_smooth_combined_pipeline,
    default_torch_dtype,
)
from .reporting import save_metrics
from .schedules import (
    SmoothScheduleConfig,
    control_weight,
    make_control_staircase,
    semantic_weight,
)
from .settings import BASE_MODEL_DIR, CONTROLNET_MODEL_DIR, IP_ADAPTER_MODEL_DIR, GenerationConfig


def hard_mode_name(tau: float) -> str:
    return f"tau_{str(tau).replace('.', 'p')}"


def save_experiment_config(root: Path, payload: dict) -> None:
    config_path = root / "experiment_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Experiment config mismatch in {config_path}. Use a new output directory or remove the old one.")
    else:
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_output_dirs(root: Path, modes: Iterable[str]) -> dict[str, Path]:
    output_dirs: dict[str, Path] = {}
    for mode in modes:
        output_dir = root / mode / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[mode] = output_dir
    return output_dirs


def generate_hard_modes(
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
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating hard-switch modes"):
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
            mode = hard_mode_name(tau)
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


def generate_smooth_modes(
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
    for row_idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating smooth modes"):
        edge = Image.open(row.edge_path).convert("L")
        edge_batch = [edge] * control_segments
        source = Image.open(row.image_path).convert("RGB")
        for mode_cfg in mode_configs:
            output_path = output_dirs[mode_cfg["mode"]] / f"{row.sample_id}.png"
            if output_path.exists() and resume:
                continue

            schedule = SmoothScheduleConfig(
                tau=float(mode_cfg["tau"]),
                sharpness=float(mode_cfg["sharpness"]),
                control_max_scale=float(mode_cfg["control_max_scale"]),
                ip_max_scale=float(mode_cfg["ip_max_scale"]),
            )
            control_scales, control_guidance_end = make_control_staircase(
                schedule.tau,
                schedule.sharpness,
                schedule.control_max_scale,
                control_segments,
            )
            callback = DynamicIPAdapterScaleCallback(
                schedule.tau,
                schedule.sharpness,
                schedule.ip_max_scale,
            )
            pipe.set_ip_adapter_scale(0.0)
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
                generator=generator_for_row(device, cfg.seed + 1_000_000, row_idx),
                width=cfg.image_size,
                height=cfg.image_size,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=[],
            ).images[0]
            result.save(output_path)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_experiment_outputs(
    manifest: pd.DataFrame,
    output_dirs: dict[str, Path],
    root: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = evaluate_outputs(manifest, output_dirs, device)
    save_metrics(metrics, root / "per_sample_metrics.csv", root / "summary.json")
    summary = build_search_summary(metrics)
    summary.to_csv(root / "search_summary.csv", index=False)
    return metrics, summary


def load_search_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def select_best_mode(summary: pd.DataFrame, prefix: str | None = None) -> pd.Series:
    frame = summary.copy()
    if prefix is not None:
        frame = frame.loc[frame["mode"].astype(str).str.startswith(prefix)].copy()
    if frame.empty:
        raise ValueError("No modes available after filtering.")
    return frame.sort_values("balanced_score", ascending=False).iloc[0]


def compute_pairwise_summary(metrics: pd.DataFrame, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    records: list[dict] = []
    for left_mode, right_mode in comparisons:
        left = metrics.loc[metrics["mode"] == left_mode, ["sample_id", "canny_mse", "clip_similarity"]].set_index("sample_id")
        right = metrics.loc[metrics["mode"] == right_mode, ["sample_id", "canny_mse", "clip_similarity"]].set_index("sample_id")
        joined = left.join(right, how="inner", lsuffix="_left", rsuffix="_right")
        if joined.empty:
            continue
        canny_delta = joined["canny_mse_right"] - joined["canny_mse_left"]
        clip_delta = joined["clip_similarity_left"] - joined["clip_similarity_right"]
        records.append(
            {
                "left_mode": left_mode,
                "right_mode": right_mode,
                "count": int(len(joined)),
                "left_canny_better_rate": float((joined["canny_mse_left"] < joined["canny_mse_right"]).mean()),
                "left_clip_better_rate": float((joined["clip_similarity_left"] > joined["clip_similarity_right"]).mean()),
                "canny_delta_right_minus_left_mean": float(canny_delta.mean()),
                "clip_delta_left_minus_right_mean": float(clip_delta.mean()),
            }
        )
    return pd.DataFrame(records)


def save_pairwise_summary(metrics: pd.DataFrame, comparisons: list[tuple[str, str]], path: Path) -> pd.DataFrame:
    summary = compute_pairwise_summary(metrics, comparisons)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def interesting_tau_samples(
    metrics: pd.DataFrame,
    naive_mode: str,
    early_mode: str,
    late_mode: str,
    limit: int = 3,
) -> list[str]:
    pivot = (
        metrics.loc[metrics["mode"].isin([naive_mode, early_mode, late_mode]), ["sample_id", "mode", "canny_mse", "clip_similarity"]]
        .pivot(index="sample_id", columns="mode")
    )
    pivot.columns = ["__".join(col) for col in pivot.columns]
    pivot = pivot.dropna().reset_index()
    if pivot.empty:
        return []
    pivot["interestingness"] = (
        (pivot[f"clip_similarity__{early_mode}"] - pivot[f"clip_similarity__{late_mode}"]).abs()
        + (pivot[f"canny_mse__{early_mode}"] - pivot[f"canny_mse__{late_mode}"]).abs()
        + (pivot[f"clip_similarity__{early_mode}"] - pivot[f"clip_similarity__{naive_mode}"]).abs()
    )
    pivot = pivot.sort_values("interestingness", ascending=False)
    return pivot["sample_id"].head(limit).tolist()


def best_improvement_samples(
    metrics: pd.DataFrame,
    naive_mode: str,
    candidate_mode: str,
    limit: int = 3,
) -> list[str]:
    pivot = (
        metrics.loc[metrics["mode"].isin([naive_mode, candidate_mode]), ["sample_id", "mode", "canny_mse", "clip_similarity"]]
        .pivot(index="sample_id", columns="mode")
    )
    pivot.columns = ["__".join(col) for col in pivot.columns]
    pivot = pivot.dropna().reset_index()
    if pivot.empty:
        return []
    canny_gain = pivot[f"canny_mse__{naive_mode}"] - pivot[f"canny_mse__{candidate_mode}"]
    clip_gain = pivot[f"clip_similarity__{candidate_mode}"] - pivot[f"clip_similarity__{naive_mode}"]
    pivot["improvement_score"] = canny_gain + clip_gain
    pivot = pivot.sort_values("improvement_score", ascending=False)
    return pivot["sample_id"].head(limit).tolist()


def _load_conflict_assets(manifest_row: pd.Series, image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def export_tau_sweep_grid(
    manifest: pd.DataFrame,
    hard_root: Path,
    sample_ids: list[str],
    tau_modes: list[str],
    output_path: Path,
) -> None:
    columns = ["structure_image", "semantic_image", "naive_combined", *tau_modes]
    title_map = {
        "structure_image": "Structure image",
        "semantic_image": "Semantic reference",
        "naive_combined": "Naive",
        **{mode: mode.replace("_", " ") for mode in tau_modes},
    }
    fig, axes = plt.subplots(len(sample_ids), len(columns), figsize=(2.2 * len(columns), 2.4 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = np.array([axes])
    for row_idx, sample_id in enumerate(sample_ids):
        manifest_row = manifest.loc[manifest["sample_id"] == sample_id].iloc[0]
        images: list[Image.Image] = [
            _load_conflict_assets(manifest_row, Path(manifest_row["structure_image_path"])),
            _load_conflict_assets(manifest_row, Path(manifest_row["semantic_image_path"])),
        ]
        for mode in ["naive_combined", *tau_modes]:
            images.append(Image.open(hard_root / mode / "images" / f"{sample_id}.png").convert("RGB"))
        for col_idx, (ax, key, image) in enumerate(zip(axes[row_idx], columns, images)):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(title_map[key], fontsize=9)
        axes[row_idx, 0].set_ylabel(str(sample_id), fontsize=9, rotation=90)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_final_comparison_gallery(
    manifest: pd.DataFrame,
    final_root: Path,
    sample_ids: list[str],
    modes: list[str],
    output_path: Path,
) -> None:
    columns = ["structure_image", "semantic_image", *modes]
    title_map = {
        "structure_image": "Structure image",
        "semantic_image": "Semantic reference",
        **{mode: mode.replace("_", " ") for mode in modes},
    }
    fig, axes = plt.subplots(len(sample_ids), len(columns), figsize=(2.6 * len(columns), 2.6 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = np.array([axes])
    for row_idx, sample_id in enumerate(sample_ids):
        manifest_row = manifest.loc[manifest["sample_id"] == sample_id].iloc[0]
        images = [
            _load_conflict_assets(manifest_row, Path(manifest_row["structure_image_path"])),
            _load_conflict_assets(manifest_row, Path(manifest_row["semantic_image_path"])),
        ]
        for mode in modes:
            images.append(Image.open(final_root / mode / "images" / f"{sample_id}.png").convert("RGB"))
        for col_idx, (ax, key, image) in enumerate(zip(axes[row_idx], columns, images)):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(title_map[key], fontsize=9)
        axes[row_idx, 0].set_ylabel(str(sample_id), fontsize=9, rotation=90)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_tradeoff_scatter(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    offset_cycle = [(4, 4), (6, -10), (-30, 5), (-32, -10), (8, 12), (-18, 12)]
    if "stage" in summary.columns:
        color_map = {
            "tau": "#4C72B0",
            "sharpness": "#DD8452",
            "ip_max_scale": "#55A868",
            "control_max_scale": "#C44E52",
        }
        marker_map = {
            "tau": "o",
            "sharpness": "s",
            "ip_max_scale": "^",
            "control_max_scale": "D",
        }
        for stage, frame in summary.groupby("stage", sort=False):
            ax.scatter(
                frame["canny_mse_mean"],
                frame["clip_similarity_mean"],
                s=70,
                color=color_map.get(stage, "#4C72B0"),
                marker=marker_map.get(stage, "o"),
                label=str(stage).replace("_", " "),
            )
            for idx, row in enumerate(frame.itertuples(index=False)):
                dx, dy = offset_cycle[idx % len(offset_cycle)]
                ax.annotate(
                    _compact_mode_label(row.mode, getattr(row, "stage", None)),
                    (row.canny_mse_mean, row.clip_similarity_mean),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
                )
        ax.legend(frameon=False, title="Search stage", loc="best")
    else:
        ax.scatter(summary["canny_mse_mean"], summary["clip_similarity_mean"], s=60, color="#4C72B0")
        for idx, row in enumerate(summary.itertuples(index=False)):
            dx, dy = offset_cycle[idx % len(offset_cycle)]
            ax.annotate(
                _compact_mode_label(row.mode),
                (row.canny_mse_mean, row.clip_similarity_mean),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
    ax.set_xlabel("Canny Edge MSE (lower is better)")
    ax.set_ylabel("CLIP similarity (higher is better)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_tau_value(mode: str) -> float | None:
    if mode.startswith("tau_"):
        return float(mode.split("_", 1)[1].replace("p", "."))
    if "smooth_tau__tau_" in mode:
        return float(mode.split("smooth_tau__tau_", 1)[1].replace("p", "."))
    return None


def _compact_mode_label(mode: str, stage: str | None = None) -> str:
    if mode.startswith("tau_"):
        return f"tau={mode.split('_', 1)[1].replace('p', '.')}"
    if stage == "tau" or mode.startswith("smooth_tau__tau_"):
        tau = _parse_tau_value(mode)
        return f"T{tau:.2f}" if tau is not None else mode
    if stage == "sharpness":
        token = mode.split("__sharp_")[-1].replace("p", ".")
        return f"S{token}"
    if stage == "ip_max_scale":
        token = mode.split("__ip_")[-1].replace("p", ".")
        return f"I{token}"
    if stage == "control_max_scale":
        token = mode.split("__ctrl_")[-1].replace("p", ".")
        return f"C{token}"
    return mode.replace("_", " ")


def export_schedule_overview(
    output_path: Path,
    tau_examples: list[float],
    sharpness_examples: list[float],
    reference_tau: float,
    reference_sharpness: float,
    control_max_scale: float = 1.0,
    ip_max_scale: float = 0.8,
) -> None:
    progress = np.linspace(0.0, 1.0, 400)
    tau_examples = sorted(tau_examples)
    sharpness_examples = sorted(sharpness_examples)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    hard_ax = axes[0, 0]
    hard_control = np.where(progress < reference_tau, control_max_scale, 0.0)
    hard_semantic = np.where(progress < reference_tau, 0.0, ip_max_scale)
    hard_ax.step(progress, hard_control, where="post", color="#4C72B0", linewidth=2.5, label="Control weight")
    hard_ax.step(progress, hard_semantic, where="post", color="#C44E52", linewidth=2.5, label="Semantic weight")
    hard_ax.axvline(reference_tau, color="black", linestyle="--", linewidth=1)
    hard_ax.text(reference_tau + 0.01, 0.92, r"$\tau$", fontsize=10)
    hard_ax.set_title("Hard switch")
    hard_ax.set_ylabel("Conditioning weight")
    hard_ax.legend(frameon=False, loc="center right")

    smooth_ax = axes[0, 1]
    smooth_control = [control_weight(t, reference_tau, reference_sharpness, control_max_scale) for t in progress]
    smooth_semantic = [semantic_weight(t, reference_tau, reference_sharpness, ip_max_scale) for t in progress]
    smooth_ax.plot(progress, smooth_control, color="#4C72B0", linewidth=2.5, label="Control weight")
    smooth_ax.plot(progress, smooth_semantic, color="#C44E52", linewidth=2.5, label="Semantic weight")
    smooth_ax.axvline(reference_tau, color="black", linestyle="--", linewidth=1)
    smooth_ax.text(reference_tau + 0.01, 0.92, r"$\tau$", fontsize=10)
    smooth_ax.set_title(f"Smooth switch (tau={reference_tau:.2f}, sharpness={reference_sharpness:g})")
    smooth_ax.legend(frameon=False, loc="center right")

    tau_ax = axes[1, 0]
    tau_colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(tau_examples)))
    for color, tau in zip(tau_colors, tau_examples):
        curve = [semantic_weight(t, tau, reference_sharpness, ip_max_scale) for t in progress]
        tau_ax.plot(progress, curve, color=color, linewidth=2.3, label=f"tau={tau:.2f}")
        tau_ax.axvline(tau, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    tau_ax.set_title("Effect of tau")
    tau_ax.set_xlabel("Denoising progress (0 = pure noise, 1 = clean image)")
    tau_ax.set_ylabel("Semantic weight")
    tau_ax.legend(frameon=False, loc="best")

    sharp_ax = axes[1, 1]
    sharp_colors = plt.cm.Oranges(np.linspace(0.45, 0.85, len(sharpness_examples)))
    for color, sharpness in zip(sharp_colors, sharpness_examples):
        curve = [semantic_weight(t, reference_tau, sharpness, ip_max_scale) for t in progress]
        sharp_ax.plot(progress, curve, color=color, linewidth=2.3, label=f"sharpness={sharpness:g}")
    sharp_ax.axvline(reference_tau, color="black", linestyle="--", linewidth=1)
    sharp_ax.text(reference_tau + 0.01, 0.92, r"$\tau$", fontsize=10)
    sharp_ax.set_title("Effect of sharpness")
    sharp_ax.set_xlabel("Denoising progress (0 = pure noise, 1 = clean image)")
    sharp_ax.set_ylabel("Semantic weight")
    sharp_ax.legend(frameon=False, loc="best")

    for ax in axes.ravel():
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Time-step scheduling overview", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_hard_vs_smooth_tau_metrics(
    hard_summary: pd.DataFrame,
    smooth_summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    hard = hard_summary.loc[hard_summary["mode"].astype(str).str.startswith("tau_")].copy()
    hard["tau"] = hard["mode"].astype(str).map(_parse_tau_value)
    smooth = smooth_summary.loc[smooth_summary["stage"].astype(str) == "tau"].copy()
    if hard.empty or smooth.empty:
        return
    hard = hard.sort_values("tau")
    smooth = smooth.sort_values("tau")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    metric_specs = [
        ("canny_mse_mean", "Canny Edge MSE", "Lower is better"),
        ("clip_similarity_mean", "CLIP similarity", "Higher is better"),
        ("balanced_score", "Balanced score", "Higher is better"),
    ]
    style_map = {
        "hard": {"color": "#4C72B0", "marker": "o", "label": "Hard switch"},
        "smooth": {"color": "#DD8452", "marker": "s", "label": "Smooth schedule"},
    }
    for ax, (metric, panel_title, ylabel) in zip(axes, metric_specs):
        ax.plot(hard["tau"], hard[metric], linewidth=2, **style_map["hard"])
        ax.plot(smooth["tau"], smooth[metric], linewidth=2, **style_map["smooth"])
        ax.set_title(panel_title)
        ax.set_xlabel("Tau")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(set(hard["tau"].tolist()) | set(smooth["tau"].tolist())))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_smooth_stage_metric_sweep(
    smooth_summary: pd.DataFrame,
    stage: str,
    x_column: str,
    output_path: Path,
    title: str,
) -> None:
    frame = smooth_summary.loc[smooth_summary["stage"].astype(str) == stage].copy()
    if frame.empty:
        return
    frame = frame.sort_values(x_column)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    metric_specs = [
        ("canny_mse_mean", "Canny Edge MSE", "Lower is better", "#4C72B0"),
        ("clip_similarity_mean", "CLIP similarity", "Higher is better", "#DD8452"),
        ("balanced_score", "Balanced score", "Higher is better", "#55A868"),
    ]
    for ax, (metric, panel_title, ylabel, color) in zip(axes, metric_specs):
        ax.plot(frame[x_column], frame[metric], color=color, marker="o", linewidth=2)
        ax.set_title(panel_title)
        ax.set_xlabel(x_column.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.set_xticks(frame[x_column].tolist())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_metric_bars(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    ordered = summary.copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].bar(ordered["mode"], ordered["canny_mse_mean"], color="#4C72B0")
    axes[0].set_title("Mean Canny Edge MSE")
    axes[0].set_ylabel("Lower is better")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].bar(ordered["mode"], ordered["clip_similarity_mean"], color="#DD8452")
    axes[1].set_title("Mean CLIP similarity")
    axes[1].set_ylabel("Higher is better")
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)
