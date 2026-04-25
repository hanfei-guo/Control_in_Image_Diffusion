from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from midway_project.experiments import build_conflict_manifest, load_manifest
from midway_project.final_stage import (
    best_improvement_samples,
    export_final_comparison_gallery,
    export_hard_vs_smooth_tau_metrics,
    export_metric_bars,
    export_schedule_overview,
    export_smooth_stage_metric_sweep,
    export_tau_sweep_grid,
    export_tradeoff_scatter,
    generate_hard_modes,
    generate_smooth_modes,
    hard_mode_name,
    interesting_tau_samples,
    load_search_summary,
    save_experiment_config,
    save_experiment_outputs,
    save_pairwise_summary,
    select_best_mode,
)
from midway_project.models import detect_device
from midway_project.settings import (
    COMBINED_OUTPUT_DIR,
    DEFAULT_SMOOTH_CONTROL_SEGMENTS,
    DEFAULT_TAU_CANDIDATES,
    GenerationConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final-stage experiments with resumable search and full confirmation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-size", type=int, default=1000)
    parser.add_argument("--search-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=10623)
    parser.add_argument("--pairing-seed", type=int, default=10624)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=10623)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--hard-search-name", type=str, default="search_100_conflict")
    parser.add_argument("--smooth-search-name", type=str, default="search_100_conflict_smooth")
    parser.add_argument("--final-eval-name", type=str, default="final_eval_1000_conflict")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures" / "final_stage")
    parser.add_argument("--taus", type=str, default=",".join(str(value) for value in DEFAULT_TAU_CANDIDATES))
    parser.add_argument("--control-segments", type=int, default=DEFAULT_SMOOTH_CONTROL_SEGMENTS)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_taus(values: str) -> list[float]:
    return [float(token.strip()) for token in values.split(",") if token.strip()]


def run_subprocess(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def ensure_hard_search(args, tau_values: list[float]) -> Path:
    root = COMBINED_OUTPUT_DIR / args.hard_search_name
    summary_path = root / "search_summary.csv"
    if summary_path.exists():
        print(f"Reusing hard-search results from {root}")
        return root

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_combined_experiments.py"),
        "--manifest",
        str(args.manifest),
        "--sample-size",
        str(args.search_size),
        "--pairing",
        "conflict",
        "--sample-seed",
        str(args.sample_seed),
        "--pairing-seed",
        str(args.pairing_seed),
        "--seed",
        str(args.seed),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--controlnet-scale",
        str(args.controlnet_scale),
        "--ip-adapter-scale",
        str(args.ip_adapter_scale),
        "--taus",
        ",".join(str(value) for value in tau_values),
        "--experiment-name",
        args.hard_search_name,
        "--resume",
    ]
    if args.device:
        command.extend(["--device", args.device])
    run_subprocess(command)
    return root


def ensure_smooth_search(args) -> Path:
    root = COMBINED_OUTPUT_DIR / args.smooth_search_name
    summary_path = root / "search_summary.csv"
    if summary_path.exists():
        print(f"Reusing smooth-search results from {root}")
        return root

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_smooth_schedule_search.py"),
        "--manifest",
        str(args.manifest),
        "--sample-size",
        str(args.search_size),
        "--pairing",
        "conflict",
        "--sample-seed",
        str(args.sample_seed),
        "--pairing-seed",
        str(args.pairing_seed),
        "--seed",
        str(args.seed),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--controlnet-scale",
        str(args.controlnet_scale),
        "--ip-adapter-scale",
        str(args.ip_adapter_scale),
        "--control-segments",
        str(args.control_segments),
        "--experiment-name",
        args.smooth_search_name,
        "--resume",
    ]
    if args.device:
        command.extend(["--device", args.device])
    run_subprocess(command)
    return root


def main() -> None:
    args = build_parser().parse_args()
    tau_values = parse_taus(args.taus)

    hard_root = ensure_hard_search(args, tau_values)
    smooth_root = ensure_smooth_search(args)

    hard_summary = load_search_summary(hard_root / "search_summary.csv")
    smooth_summary = load_search_summary(smooth_root / "search_summary.csv")
    best_hard = select_best_mode(hard_summary, prefix="tau_")
    best_smooth = select_best_mode(smooth_summary)

    print("Best hard-switch mode:")
    print(best_hard.to_string())
    print("\nBest smooth mode:")
    print(best_smooth.to_string())

    full_manifest = load_manifest(args.manifest)
    final_manifest = build_conflict_manifest(full_manifest, args.full_size, args.sample_seed, args.pairing_seed)
    final_root = COMBINED_OUTPUT_DIR / args.final_eval_name
    final_root.mkdir(parents=True, exist_ok=True)
    final_manifest.to_csv(final_root / "experiment_manifest.csv", index=False)
    selected_modes_payload = {
        "hard_search_root": str(hard_root),
        "smooth_search_root": str(smooth_root),
        "best_hard_mode": best_hard["mode"],
        "best_hard_tau": float(best_hard["mode"].split("_")[1].replace("p", ".")),
        "best_smooth_mode": best_smooth["mode"],
        "best_smooth_tau": float(best_smooth["tau"]),
        "best_smooth_sharpness": float(best_smooth["sharpness"]),
        "best_smooth_control_max_scale": float(best_smooth["control_max_scale"]),
        "best_smooth_ip_max_scale": float(best_smooth["ip_max_scale"]),
    }
    (final_root / "selected_modes.json").write_text(json.dumps(selected_modes_payload, indent=2), encoding="utf-8")
    save_experiment_config(
        final_root,
        {
            "full_size": args.full_size,
            "sample_seed": args.sample_seed,
            "pairing_seed": args.pairing_seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_scale": args.controlnet_scale,
            "ip_adapter_scale": args.ip_adapter_scale,
            "selected_modes": selected_modes_payload,
        },
    )

    cfg = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
    )
    device = args.device or detect_device()

    final_modes = ["naive_combined", best_hard["mode"], best_smooth["mode"]]
    output_dirs = {
        mode: final_root / mode / "images"
        for mode in final_modes
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    generate_hard_modes(
        final_manifest,
        {mode: path for mode, path in output_dirs.items() if mode in {"naive_combined", best_hard["mode"]}},
        [float(best_hard["mode"].split("_")[1].replace("p", "."))],
        cfg,
        device,
        args.resume,
    )
    generate_smooth_modes(
        final_manifest,
        {best_smooth["mode"]: output_dirs[best_smooth["mode"]]},
        [
            {
                "mode": best_smooth["mode"],
                "tau": float(best_smooth["tau"]),
                "sharpness": float(best_smooth["sharpness"]),
                "control_max_scale": float(best_smooth["control_max_scale"]),
                "ip_max_scale": float(best_smooth["ip_max_scale"]),
            }
        ],
        cfg,
        device,
        args.resume,
        args.control_segments,
    )

    metrics, summary = save_experiment_outputs(final_manifest, output_dirs, final_root, device)
    pairwise = save_pairwise_summary(
        metrics,
        [
            (best_hard["mode"], "naive_combined"),
            (best_smooth["mode"], "naive_combined"),
            (best_smooth["mode"], best_hard["mode"]),
        ],
        final_root / "pairwise_summary.csv",
    )

    hard_metrics = pd.read_csv(hard_root / "per_sample_metrics.csv")
    hard_manifest = pd.read_csv(hard_root / "experiment_manifest.csv", dtype={"sample_id": str})
    tau_modes = [hard_mode_name(tau) for tau in tau_values]
    tau_sample_ids = interesting_tau_samples(hard_metrics, "naive_combined", tau_modes[0], tau_modes[-1], limit=3)
    if tau_sample_ids:
        export_tau_sweep_grid(
            hard_manifest,
            hard_root,
            tau_sample_ids,
            tau_modes,
            args.figures_dir / "tau_sweep_examples",
        )

    final_sample_ids = best_improvement_samples(metrics, "naive_combined", best_smooth["mode"], limit=3)
    if final_sample_ids:
        export_final_comparison_gallery(
            final_manifest,
            final_root,
            final_sample_ids,
            ["naive_combined", best_hard["mode"], best_smooth["mode"]],
            args.figures_dir / "final_method_gallery",
        )

    export_tradeoff_scatter(hard_summary, args.figures_dir / "hard_search_tradeoff", "Hard-switch tau search (conflict subset)")
    export_tradeoff_scatter(smooth_summary, args.figures_dir / "smooth_search_tradeoff", "Smooth-schedule search (conflict subset)")
    smooth_tau_examples = sorted(smooth_summary.loc[smooth_summary["stage"] == "tau", "tau"].dropna().astype(float).unique().tolist())
    smooth_sharpness_examples = sorted(
        smooth_summary.loc[smooth_summary["stage"] == "sharpness", "sharpness"].dropna().astype(float).unique().tolist()
    )
    export_schedule_overview(
        args.figures_dir / "schedule_overview",
        tau_examples=smooth_tau_examples,
        sharpness_examples=smooth_sharpness_examples,
        reference_tau=float(best_smooth["tau"]),
        reference_sharpness=float(best_smooth["sharpness"]),
        control_max_scale=float(best_smooth["control_max_scale"]),
        ip_max_scale=float(best_smooth["ip_max_scale"]),
    )
    export_hard_vs_smooth_tau_metrics(
        hard_summary,
        smooth_summary,
        args.figures_dir / "hard_vs_smooth_tau_metrics",
        "Hard vs smooth tau sweep",
    )
    export_smooth_stage_metric_sweep(
        smooth_summary,
        "sharpness",
        "sharpness",
        args.figures_dir / "smooth_sharpness_metrics",
        "Smooth sharpness sweep at fixed tau",
    )
    export_metric_bars(summary, args.figures_dir / "final_method_bars", "Final full-set comparison")

    (args.figures_dir / "recommended_samples.json").write_text(
        json.dumps(
            {
                "tau_sweep_sample_ids": tau_sample_ids,
                "final_gallery_sample_ids": final_sample_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print("\nPairwise summary:")
    print(pairwise.to_string(index=False))

    notebook_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_final_results_notebook.py"),
        "--hard-search-root",
        str(hard_root),
        "--smooth-search-root",
        str(smooth_root),
        "--final-root",
        str(final_root),
        "--figures-dir",
        str(args.figures_dir),
    ]
    run_subprocess(notebook_command)


if __name__ == "__main__":
    main()
