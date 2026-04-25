from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .settings import (
    COCO_EDGES_DIR,
    COCO_IMAGES_DIR,
    COCO_MANIFEST_PATH,
    COCO_SUBSET_DIR,
    DEFAULT_DATASET_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
)


def ensure_dataset_dirs() -> None:
    COCO_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    COCO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    COCO_EDGES_DIR.mkdir(parents=True, exist_ok=True)


def resize_and_crop(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def canny_image(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    return Image.fromarray(edges, mode="L")


def choose_caption(captions: Iterable[str]) -> str:
    captions = [caption.strip() for caption in captions if caption and caption.strip()]
    if not captions:
        return "A high quality photograph."
    return max(captions, key=len)


def sample_id_from_row(row: dict) -> str:
    return f"{int(row['image_id']):012d}"


def download_and_process_row(row: dict, image_size: int = DEFAULT_IMAGE_SIZE, timeout: int = 30) -> dict:
    sample_id = sample_id_from_row(row)
    image_path = COCO_IMAGES_DIR / f"{sample_id}.png"
    edge_path = COCO_EDGES_DIR / f"{sample_id}.png"
    if image_path.exists() and edge_path.exists():
        return {
            "sample_id": sample_id,
            "image_id": int(row["image_id"]),
            "file_name": row["file_name"],
            "caption": choose_caption(row["captions"]),
            "image_path": str(image_path),
            "edge_path": str(edge_path),
            "coco_url": row["coco_url"],
        }

    response = requests.get(row["coco_url"], timeout=timeout)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    image = resize_and_crop(image, size=image_size)
    edges = canny_image(image)

    image.save(image_path)
    edges.save(edge_path)

    return {
        "sample_id": sample_id,
        "image_id": int(row["image_id"]),
        "file_name": row["file_name"],
        "caption": choose_caption(row["captions"]),
        "image_path": str(image_path),
        "edge_path": str(edge_path),
        "coco_url": row["coco_url"],
    }


def prepare_coco_subset(
    subset_size: int = DEFAULT_DATASET_SIZE,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    overwrite_manifest: bool = False,
    max_workers: int = 12,
) -> pd.DataFrame:
    ensure_dataset_dirs()

    if COCO_MANIFEST_PATH.exists() and not overwrite_manifest:
        manifest = pd.read_csv(COCO_MANIFEST_PATH, dtype={"sample_id": str})
        manifest["sample_id"] = manifest["sample_id"].str.zfill(12)
        if len(manifest) >= subset_size:
            return manifest.head(subset_size).copy()

    dataset = load_dataset("phiyodr/coco2017", split="validation")
    subset = dataset.shuffle(seed=seed).select(range(subset_size))
    rows = [subset[idx] for idx in range(len(subset))]
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_and_process_row, row, image_size)
            for row in rows
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading COCO subset"):
            records.append(future.result())

    manifest = pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)
    manifest.to_csv(COCO_MANIFEST_PATH, index=False)
    return manifest
