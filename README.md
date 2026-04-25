# Midway Goal Implementation

This repository implements the proposal up to the midway milestone:

- local download of the required models
- reproducible COCO 2017 validation subset curation
- baseline generation for `ControlNet (Canny)` and `IP-Adapter`
- automated `Canny MSE` and `CLIP image-image similarity`
- a notebook that loads saved artifacts and visualizes the results

## Environment

The code was validated with `C:\Users\hanfield\anaconda3\envs\neuro\python.exe`.

Install the package in editable mode:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe -m pip install -e .
```

## Commands

Download all model weights to `assets/models`:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe scripts\download_assets.py
```

Prepare the fixed COCO subset locally:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe scripts\prepare_coco_subset.py --subset-size 1000
```

Run the midway baselines and metrics:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe scripts\run_baselines.py --manifest assets\data\coco2017_midway\subset_manifest.csv --resume
```

Run the next-stage combined experiments on a random 100-sample subset:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe scripts\run_combined_experiments.py --manifest assets\data\coco2017_midway\subset_manifest.csv --sample-size 100 --experiment-name search_100 --resume
```

Run the smooth schedule search on a random 100-sample conflicting subset:

```powershell
C:\Users\hanfield\anaconda3\envs\neuro\python.exe scripts\run_smooth_schedule_search.py --manifest assets\data\coco2017_midway\subset_manifest.csv --sample-size 100 --pairing conflict --experiment-name search_100_conflict_smooth --resume
```

## Layout

- `src/midway_project/`: reusable pipeline, dataset, metric, and reporting code
- `scripts/`: end-to-end CLI entry points
- `assets/`: downloaded models and curated local dataset
- `outputs/midway_baselines/`: generated images and metrics
- `outputs/combined_experiments/`: naive-combined and tau-scheduled experiment outputs
- `notebooks/midway_results.ipynb`: result browser and visualization notebook
