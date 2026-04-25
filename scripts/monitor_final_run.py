from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor the final experiment run and verify postprocessing outputs.")
    parser.add_argument("--poll-seconds", type=int, default=1800, help="Seconds between progress snapshots.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "assets" / "data" / "coco2017_midway" / "subset_manifest.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hard-search-root", type=Path, default=PROJECT_ROOT / "outputs" / "combined_experiments" / "search_100_conflict")
    parser.add_argument("--smooth-search-root", type=Path, default=PROJECT_ROOT / "outputs" / "combined_experiments" / "search_100_conflict_smooth")
    parser.add_argument("--final-root", type=Path, default=PROJECT_ROOT / "outputs" / "combined_experiments" / "final_eval_1000_conflict")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures" / "final_stage")
    parser.add_argument("--log-path", type=Path, default=PROJECT_ROOT / "final_monitor.log")
    return parser


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] {message}\n")


def experiment_processes() -> list[psutil.Process]:
    matches: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "run_final_experiments.py" in cmdline:
            matches.append(proc)
    return matches


def count_generated_images(final_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mode_dir in sorted(final_root.glob("*")):
        images_dir = mode_dir / "images"
        if images_dir.is_dir():
            counts[mode_dir.name] = len(list(images_dir.glob("*.png")))
    return counts


def query_gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - best-effort status only
        return f"gpu=unavailable ({exc})"
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return "gpu=unavailable (empty nvidia-smi output)"
    util, mem_used, mem_total, temp, power = [part.strip() for part in line.split(",")]
    return f"gpu_util={util}% mem={mem_used}/{mem_total}MiB temp={temp}C power={power}W"


def missing_final_artifacts(final_root: Path, figures_dir: Path) -> list[Path]:
    expected = [
        final_root / "per_sample_metrics.csv",
        final_root / "search_summary.csv",
        final_root / "pairwise_summary.csv",
        figures_dir / "final_method_bars.png",
        figures_dir / "hard_search_tradeoff.png",
        figures_dir / "smooth_search_tradeoff.png",
        figures_dir / "recommended_samples.json",
        PROJECT_ROOT / "notebooks" / "final_results.ipynb",
    ]
    return [path for path in expected if not path.exists()]


def run_resume_command(manifest: Path, device: str, log_path: Path) -> None:
    append_log(log_path, "Final artifacts are incomplete; running resumable final pipeline.")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_final_experiments.py"),
            "--manifest",
            str(manifest),
            "--device",
            device,
            "--resume",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def build_notebook_if_needed(
    hard_search_root: Path,
    smooth_search_root: Path,
    final_root: Path,
    figures_dir: Path,
    log_path: Path,
) -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "final_results.ipynb"
    if notebook_path.exists():
        return
    append_log(log_path, "Final notebook is missing; building it directly.")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_final_results_notebook.py"),
            "--hard-search-root",
            str(hard_search_root),
            "--smooth-search-root",
            str(smooth_search_root),
            "--final-root",
            str(final_root),
            "--figures-dir",
            str(figures_dir),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def main() -> None:
    args = build_parser().parse_args()
    append_log(args.log_path, "Monitor started.")

    while True:
        processes = experiment_processes()
        counts = count_generated_images(args.final_root)
        count_text = ", ".join(f"{mode}={count}" for mode, count in sorted(counts.items())) or "no image outputs yet"
        gpu_text = query_gpu_snapshot()
        if processes:
            pids = ",".join(str(proc.pid) for proc in processes)
            append_log(args.log_path, f"Main experiment still running. pids={pids}; {count_text}; {gpu_text}")
            time.sleep(args.poll_seconds)
            continue

        append_log(args.log_path, f"Main experiment process not found. Final counts: {count_text}; {gpu_text}")
        missing = missing_final_artifacts(args.final_root, args.figures_dir)
        if missing:
            append_log(args.log_path, "Missing artifacts: " + ", ".join(str(path) for path in missing))
            run_resume_command(args.manifest, args.device, args.log_path)
            continue

        build_notebook_if_needed(
            args.hard_search_root,
            args.smooth_search_root,
            args.final_root,
            args.figures_dir,
            args.log_path,
        )
        append_log(args.log_path, "All final artifacts are present. Monitor exiting.")
        return


if __name__ == "__main__":
    main()
