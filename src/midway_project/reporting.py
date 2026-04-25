from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_metrics(frame: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for mode, group in frame.groupby("mode"):
        summary[mode] = {
            "count": int(len(group)),
            "canny_mse_mean": float(group["canny_mse"].mean()),
            "canny_mse_std": float(group["canny_mse"].std(ddof=0)),
            "clip_similarity_mean": float(group["clip_similarity"].mean()),
            "clip_similarity_std": float(group["clip_similarity"].std(ddof=0)),
        }
    return summary


def save_metrics(frame: pd.DataFrame, csv_path: str | Path, json_path: str | Path) -> dict:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    summary = summarize_metrics(frame)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
