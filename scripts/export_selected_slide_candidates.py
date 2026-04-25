from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import nbformat as nbf
import pandas as pd
from PIL import Image

from midway_project.experiments import evaluate_outputs
from midway_project.final_stage import generate_hard_modes, hard_mode_name
from midway_project.models import detect_device
from midway_project.reporting import save_metrics
from midway_project.settings import DEFAULT_TAU_CANDIDATES, GenerationConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTED_CASES = (
    {"category": "color", "query": "322895-254516", "sample_id": "000000322895__000000254516"},
    {"category": "color", "query": "190923-47010", "sample_id": "000000190923__000000047010"},
    {"category": "color", "query": "377575-85157", "sample_id": "000000377575__000000085157"},
    {"category": "color", "query": "148730-394940", "sample_id": "000000148730__000000394940"},
    {"category": "color", "query": "336232", "sample_id": "000000336232__000000109976"},
    {"category": "color", "query": "85682", "sample_id": "000000085682__000000187734"},
    {"category": "artifacts_correct", "query": "17959-198915", "sample_id": "000000017959__000000198915"},
    {"category": "artifacts_correct", "query": "491613", "sample_id": "000000491613__000000475191"},
    {"category": "artifacts_correct", "query": "492077", "sample_id": "000000492077__000000189226"},
    {"category": "artifacts_correct", "query": "364322", "sample_id": "000000364322__000000343934"},
    {"category": "artifacts_correct", "query": "322895", "sample_id": "000000322895__000000254516"},
    {"category": "artifacts_correct", "query": "328286", "sample_id": "000000328286__000000307145"},
    {"category": "artifacts_correct", "query": "0001993", "sample_id": "000000001993__000000050165"},
    {"category": "artifacts_correct", "query": "476787", "sample_id": "000000476787__000000280779"},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export curated selected slide cases and their tau sweeps.")
    parser.add_argument(
        "--final-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "combined_experiments" / "final_eval_1000_conflict",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "combined_experiments" / "slide_selected_ablation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "combined_experiments" / "slide_selected_curated",
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=PROJECT_ROOT / "figures" / "final_stage" / "selected_slide_cases_curated",
    )
    parser.add_argument(
        "--output-notebook",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "selected_slide_cases.ipynb",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=10623)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    return parser

def selected_cases_dataframe(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for case in SELECTED_CASES:
        matches = manifest.loc[manifest["sample_id"] == case["sample_id"]].copy()
        if matches.empty:
            raise ValueError(f"Hard-coded sample_id '{case['sample_id']}' was not found in the final manifest.")
        match = matches.iloc[0]
        rows.append(
            {
                "category": case["category"],
                "query": case["query"],
                "resolution_type": "hardcoded_sample_id",
                "sample_id": match["sample_id"],
                "structure_sample_id": match["structure_sample_id"],
                "semantic_sample_id": match["semantic_sample_id"],
                "caption": match["caption"],
                "structure_image_path": match["structure_image_path"],
                "semantic_image_path": match["semantic_image_path"],
                "edge_path": match["edge_path"],
            }
        )
    return pd.DataFrame(rows)


def ensure_output_dirs(root: Path, tau_values: list[float]) -> dict[str, Path]:
    output_dirs: dict[str, Path] = {}
    for mode in ["naive_combined", *[hard_mode_name(tau) for tau in tau_values]]:
        output_dir = root / mode / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[mode] = output_dir
    return output_dirs


def safe_name(text: str) -> str:
    return text.replace("__", "--").replace("/", "_").replace("\\", "_").replace(" ", "")


def mode_label(mode: str) -> str:
    if mode == "naive_combined":
        return "Naive"
    if mode.startswith("tau_"):
        return f"Hard ({mode.replace('tau_', 'tau=')})"
    if mode.startswith("smooth"):
        return "Smooth"
    return mode


def make_comparison_figure(row: pd.Series, final_root: Path, hard_mode: str, smooth_mode: str, output_base: Path) -> None:
    images = [
        Image.open(row.edge_path).convert("L"),
        Image.open(row.semantic_image_path).convert("RGB"),
        Image.open(final_root / "naive_combined" / "images" / f"{row.sample_id}.png").convert("RGB"),
        Image.open(final_root / hard_mode / "images" / f"{row.sample_id}.png").convert("RGB"),
        Image.open(final_root / smooth_mode / "images" / f"{row.sample_id}.png").convert("RGB"),
    ]
    titles = ["Edge Map", "Image Ref", "Naive", mode_label(hard_mode), mode_label(smooth_mode)]
    fig, axes = plt.subplots(1, len(images), figsize=(13.2, 2.9))
    for idx, (ax, image, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(image, cmap="gray" if idx == 0 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{row.query} -> {row.sample_id}", fontsize=11)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_base.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_tau_sweep_figure(row: pd.Series, ablation_root: Path, tau_values: list[float], output_base: Path) -> None:
    visuals = [
        Image.open(row.edge_path).convert("L"),
        Image.open(row.semantic_image_path).convert("RGB"),
        Image.open(ablation_root / "naive_combined" / "images" / f"{row.sample_id}.png").convert("RGB"),
    ]
    titles = ["Edge Map", "Image Ref", "Naive"]
    for tau in tau_values:
        mode = hard_mode_name(tau)
        visuals.append(Image.open(ablation_root / mode / "images" / f"{row.sample_id}.png").convert("RGB"))
        titles.append(f"tau={tau:.2f}")

    fig, axes = plt.subplots(1, len(visuals), figsize=(2.1 * len(visuals), 2.9))
    for idx, (ax, image, title) in enumerate(zip(axes, visuals, titles)):
        ax.imshow(image, cmap="gray" if idx == 0 else None)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Tau Sweep | {row.query} -> {row.sample_id}", fontsize=11)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(output_base.with_suffix(ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_notebook(
    notebook_path: Path,
    selected_df: pd.DataFrame,
    selected_csv: Path,
    curated_metrics_csv: Path,
    curated_summary_csv: Path,
    comparison_root: Path,
    tau_root: Path,
    hard_mode: str,
    smooth_mode: str,
) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Selected Slide Cases\n"
            "This notebook shows the curated 14-case export only. "
            "The sample ids are hard-coded in the export script, so there is no runtime query-resolution rule."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            f"SELECTED_CSV = Path(r'''{selected_csv.resolve()}''')\n"
            f"CURATED_METRICS_CSV = Path(r'''{curated_metrics_csv.resolve()}''')\n"
            f"CURATED_SUMMARY_CSV = Path(r'''{curated_summary_csv.resolve()}''')\n"
            f"COMPARISON_ROOT = Path(r'''{comparison_root.resolve()}''')\n"
            f"TAU_ROOT = Path(r'''{tau_root.resolve()}''')\n"
            f"HARD_MODE = '{hard_mode}'\n"
            f"SMOOTH_MODE = '{smooth_mode}'\n"
            "selected = pd.read_csv(SELECTED_CSV, dtype={'sample_id': str, 'structure_sample_id': str, 'semantic_sample_id': str})\n"
            "metrics = pd.read_csv(CURATED_METRICS_CSV, dtype={'sample_id': str})\n"
            "summary = pd.read_csv(CURATED_SUMMARY_CSV)\n"
            "display(selected)\n"
            "display(summary)"
        ),
        nbf.v4.new_markdown_cell(
            "## Reading Guide\n"
            "- `Naive` means all controls are active throughout denoising.\n"
            f"- `{mode_label(hard_mode)}` is the selected hard-switch schedule.\n"
            f"- `{mode_label(smooth_mode)}` is the selected smooth schedule.\n"
            "- The tau sweep row keeps the same pair fixed and only varies the hard-switch threshold."
        ),
    ]

    for category in ("color", "artifacts_correct"):
        subset = selected_df.loc[selected_df["category"] == category].copy()
        if subset.empty:
            continue
        cells.append(nbf.v4.new_markdown_cell(f"## {category.replace('_', ' ').title()}"))
        for row in subset.itertuples(index=False):
            comparison_path = comparison_root / row.category / f"{safe_name(row.query)}.png"
            tau_path = tau_root / row.category / f"{safe_name(row.query)}.png"
            cells.append(
                nbf.v4.new_markdown_cell(
                    f"### `{row.query}` -> `{row.sample_id}`\n"
                    f"- Resolution: `{row.resolution_type}`\n"
                    f"- Caption: {row.caption}"
                )
            )
            cells.append(
                nbf.v4.new_code_cell(
                    f"sample_id = '{row.sample_id}'\n"
                    "pivot = metrics.loc[metrics['sample_id'] == sample_id, ['mode', 'canny_mse', 'clip_similarity']].copy()\n"
                    "display(pivot.sort_values('mode'))\n"
                    f"display(Image(filename=r'''{comparison_path.resolve()}'''))\n"
                    f"display(Image(filename=r'''{tau_path.resolve()}'''))"
                )
            )

    nb["cells"] = cells
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


def main() -> None:
    args = build_parser().parse_args()
    args.final_root = args.final_root.resolve()
    args.ablation_root = args.ablation_root.resolve()
    args.output_root = args.output_root.resolve()
    args.figures_root = args.figures_root.resolve()
    args.output_notebook = args.output_notebook.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    final_manifest = pd.read_csv(
        args.final_root / "experiment_manifest.csv",
        dtype={"sample_id": str, "structure_sample_id": str, "semantic_sample_id": str},
    )
    selected_modes = json.loads((args.final_root / "selected_modes.json").read_text(encoding="utf-8"))
    hard_mode = selected_modes["best_hard_mode"]
    smooth_mode = selected_modes["best_smooth_mode"]
    tau_values = list(DEFAULT_TAU_CANDIDATES)

    selected = selected_cases_dataframe(final_manifest)

    selected_csv = args.output_root / "selected_cases_resolution.csv"
    selected_json = args.output_root / "selected_cases_resolution.json"
    selected.to_csv(selected_csv, index=False)
    selected_json.write_text(json.dumps(selected.to_dict(orient="records"), indent=2), encoding="utf-8")

    cfg = GenerationConfig(
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
    )
    device = args.device or detect_device()

    subset_manifest = final_manifest.loc[final_manifest["sample_id"].isin(selected["sample_id"].tolist())].copy()
    output_dirs = ensure_output_dirs(args.ablation_root, tau_values)
    generate_hard_modes(subset_manifest, output_dirs, tau_values, cfg, device, resume=True)

    metrics = evaluate_outputs(subset_manifest, output_dirs, device)
    per_sample_csv = args.output_root / "per_sample_metrics.csv"
    summary_json = args.output_root / "summary.json"
    save_metrics(metrics, per_sample_csv, summary_json)
    summary_csv = args.output_root / "summary.csv"
    (
        metrics.groupby("mode", as_index=False)
        .agg(
            count=("sample_id", "count"),
            canny_mse_mean=("canny_mse", "mean"),
            clip_similarity_mean=("clip_similarity", "mean"),
        )
        .to_csv(summary_csv, index=False)
    )

    comparison_root = args.figures_root / "comparisons"
    tau_root = args.figures_root / "tau_sweeps"
    for row in selected.itertuples(index=False):
        output_name = safe_name(row.query)
        make_comparison_figure(
            pd.Series(row._asdict()),
            args.final_root,
            hard_mode,
            smooth_mode,
            comparison_root / row.category / output_name,
        )
        make_tau_sweep_figure(
            pd.Series(row._asdict()),
            args.ablation_root,
            tau_values,
            tau_root / row.category / output_name,
        )

    build_notebook(
        args.output_notebook,
        selected,
        selected_csv,
        per_sample_csv,
        summary_csv,
        comparison_root,
        tau_root,
        hard_mode,
        smooth_mode,
    )

    print(f"Selected cases written to {selected_csv}")
    print(f"Curated metrics written to {per_sample_csv}")
    print(f"Notebook written to {args.output_notebook}")


if __name__ == "__main__":
    main()
