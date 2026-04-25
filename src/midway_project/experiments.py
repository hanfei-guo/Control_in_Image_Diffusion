from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from .metrics import ClipSimilarityScorer, canny_mse
from .settings import CLIP_MODEL_DIR


def load_manifest(path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    frame["sample_id"] = frame["sample_id"].str.zfill(12)
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def sample_manifest(frame: pd.DataFrame, sample_size: int | None, sample_seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame.reset_index(drop=True).copy()
    return frame.sample(n=sample_size, random_state=sample_seed).sort_values("sample_id").reset_index(drop=True)


def build_conflict_manifest(frame: pd.DataFrame, sample_size: int, sample_seed: int, pairing_seed: int) -> pd.DataFrame:
    structural = sample_manifest(frame, sample_size, sample_seed).reset_index(drop=True)
    if len(structural) < 2:
        raise ValueError("Conflict pairing requires at least 2 samples.")

    semantic = structural.sample(frac=1.0, random_state=pairing_seed).reset_index(drop=True)
    while np.any(structural["sample_id"].to_numpy() == semantic["sample_id"].to_numpy()):
        semantic = semantic.iloc[np.roll(np.arange(len(semantic)), -1)].reset_index(drop=True)

    records: list[dict] = []
    for structure_row, semantic_row in zip(structural.itertuples(index=False), semantic.itertuples(index=False)):
        pair_id = f"{structure_row.sample_id}__{semantic_row.sample_id}"
        records.append(
            {
                "sample_id": pair_id,
                "structure_sample_id": structure_row.sample_id,
                "semantic_sample_id": semantic_row.sample_id,
                "caption": semantic_row.caption,
                "image_path": semantic_row.image_path,
                "semantic_image_path": semantic_row.image_path,
                "structure_image_path": structure_row.image_path,
                "edge_path": structure_row.edge_path,
            }
        )
    return pd.DataFrame(records)


def generator_for_row(device: str, base_seed: int, row_index: int) -> torch.Generator:
    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(base_seed + row_index)
    return torch.Generator().manual_seed(base_seed + row_index)


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


def build_search_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("mode")[["canny_mse", "clip_similarity"]]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        "mode",
        "canny_mse_mean",
        "canny_mse_std",
        "canny_mse_median",
        "clip_similarity_mean",
        "clip_similarity_std",
        "clip_similarity_median",
    ]
    canny_min = summary["canny_mse_mean"].min()
    canny_max = summary["canny_mse_mean"].max()
    clip_min = summary["clip_similarity_mean"].min()
    clip_max = summary["clip_similarity_mean"].max()

    if canny_max > canny_min:
        summary["canny_score"] = (canny_max - summary["canny_mse_mean"]) / (canny_max - canny_min)
    else:
        summary["canny_score"] = 1.0
    if clip_max > clip_min:
        summary["clip_score"] = (summary["clip_similarity_mean"] - clip_min) / (clip_max - clip_min)
    else:
        summary["clip_score"] = 1.0

    summary["balanced_score"] = 0.5 * (summary["canny_score"] + summary["clip_score"])
    return summary.sort_values("balanced_score", ascending=False).reset_index(drop=True)
