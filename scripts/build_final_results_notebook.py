from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final summary notebook.")
    parser.add_argument("--hard-search-root", type=Path, required=True)
    parser.add_argument("--smooth-search-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "notebooks" / "final_results.ipynb")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.hard_search_root = args.hard_search_root.resolve()
    args.smooth_search_root = args.smooth_search_root.resolve()
    args.final_root = args.final_root.resolve()
    args.figures_dir = args.figures_dir.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Final Results Summary\n"
            "This notebook loads the completed final-stage outputs and organizes them in the same narrative order as the teammate PPT: setup, conflict stress test, quantitative results, qualitative comparisons, and conclusions."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n"
            "\n"
            f"HARD_ROOT = Path(r'''{args.hard_search_root}''')\n"
            f"SMOOTH_ROOT = Path(r'''{args.smooth_search_root}''')\n"
            f"FINAL_ROOT = Path(r'''{args.final_root}''')\n"
            f"FIGURES_DIR = Path(r'''{args.figures_dir}''')\n"
            f"BASELINE_ROOT = Path(r'''{PROJECT_ROOT / 'outputs' / 'midway_baselines' / 'metrics'}''')\n"
            "\n"
            "baseline_summary = json.loads((BASELINE_ROOT / 'summary.json').read_text(encoding='utf-8'))\n"
            "hard_summary = pd.read_csv(HARD_ROOT / 'search_summary.csv')\n"
            "smooth_summary = pd.read_csv(SMOOTH_ROOT / 'search_summary.csv')\n"
            "final_summary = pd.read_csv(FINAL_ROOT / 'search_summary.csv')\n"
            "pairwise_summary = pd.read_csv(FINAL_ROOT / 'pairwise_summary.csv')\n"
            "selected_modes = json.loads((FINAL_ROOT / 'selected_modes.json').read_text(encoding='utf-8'))\n"
            "recommended_samples = json.loads((FIGURES_DIR / 'recommended_samples.json').read_text(encoding='utf-8')) if (FIGURES_DIR / 'recommended_samples.json').exists() else {}\n"
            "hard_metrics = pd.read_csv(HARD_ROOT / 'per_sample_metrics.csv')\n"
            "final_metrics = pd.read_csv(FINAL_ROOT / 'per_sample_metrics.csv')\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Experimental Setup\n"
            "- Backbone: Stable Diffusion v1.5\n"
            "- Structure controller: ControlNet (Canny)\n"
            "- Semantic controller: IP-Adapter\n"
            "- Stress test: conflict pairing, where structure comes from image A and semantics come from an unrelated image B\n"
            "- Final workflow: hard-switch search on a smaller conflict subset, smooth-schedule search on a smaller conflict subset, then full confirmation on the selected final methods"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Markdown('### Midway Baseline Recap'))\n"
            "display(pd.DataFrame(baseline_summary).T)\n"
            "display(Markdown('### Selected Final Modes'))\n"
            "display(pd.DataFrame([selected_modes]))\n"
            "if (FIGURES_DIR / 'schedule_overview.png').exists():\n"
            "    display(Markdown('### Time-Step Scheduling Overview'))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'schedule_overview.png')))\n"
            "    display(Markdown('Suggested caption: hard switch uses a step function at tau, while smooth scheduling replaces the step with a sigmoid transition. In the normalized illustration, tau marks the transition center in time, while sharpness controls how abrupt the transition is around that center.'))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Conflict Stress Test Search Results\n"
            "The hard-switch search shows how the transition point moves the structure/semantic trade-off. The smooth search shows whether a sigmoid-style transition can produce a better compromise than a single hard cutoff."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Markdown('### Hard-Switch Search Summary'))\n"
            "display(hard_summary)\n"
            "display(Image(filename=str(FIGURES_DIR / 'hard_search_tradeoff.png')))\n"
            "display(Markdown('Suggested caption: each point is one hard-switch setting on the conflict subset; lower-left-to-upper-right movement reveals the structure/semantic trade-off induced by different transition thresholds.'))\n"
            "\n"
            "display(Markdown('### Smooth-Schedule Search Summary'))\n"
            "display(smooth_summary.head(12))\n"
            "display(Image(filename=str(FIGURES_DIR / 'smooth_search_tradeoff.png')))\n"
            "display(Markdown('Suggested caption: smooth schedules were searched progressively over transition position, sharpness, and maximum control scales; stage color shows which parameter was being varied, while compact point labels (T, S, I, C) show the tested value within that stage.'))\n"
            "\n"
            "if (FIGURES_DIR / 'hard_vs_smooth_tau_metrics.png').exists():\n"
            "    display(Markdown('### Hard vs Smooth Tau Sweep'))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'hard_vs_smooth_tau_metrics.png')))\n"
            "    display(Markdown('Suggested caption: at matched tau values, hard and smooth schedules can be compared directly. This plot makes it clear whether smoothing creates a better compromise or simply shifts the same trade-off curve.'))\n"
            "\n"
            "if (FIGURES_DIR / 'smooth_sharpness_metrics.png').exists():\n"
            "    display(Markdown('### Smooth Sharpness Sweep'))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'smooth_sharpness_metrics.png')))\n"
            "    display(Markdown('Suggested caption: increasing smooth sharpness changes how abruptly the semantic controller turns on near tau. Use this plot to explain why the selected sharpness was chosen.'))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Full Final Comparison\n"
            "This is the key table for the final report. It compares the final selected methods on the full conflict-paired evaluation set."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(final_summary)\n"
            "display(pairwise_summary)\n"
            "display(Image(filename=str(FIGURES_DIR / 'final_method_bars.png')))\n"
            "display(Markdown('Suggested caption: full-set comparison between naive combined control, the best hard-switch schedule, and the best smooth schedule. Lower Canny MSE indicates better structure following; higher CLIP similarity indicates better semantic alignment.'))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Qualitative Comparisons\n"
            "This section contains both slide-ready figures requested by the teammate discussion and the broader qualitative grids used for the report."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if (FIGURES_DIR / 'control_ip_adapter_comparison.png').exists():\n"
            "    display(Markdown('### Control + IP-Adapter -> Outputs'))\n"
            "    display(Markdown(f\"Recommended sample IDs: {recommended_samples.get('control_ip_adapter_sample_ids', [])}\"))\n"
            "    display(Markdown(f\"Additional candidate IDs: {recommended_samples.get('control_ip_adapter_candidate_sample_ids', [])}\"))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'control_ip_adapter_comparison.png')))\n"
            "    display(Markdown('Suggested caption: each row pairs one control image with one IP-Adapter reference, then compares the resulting outputs from naive combined control, the best hard switch, and the best smooth schedule.'))\n"
            "\n"
            "if (FIGURES_DIR / 'tau_ablation_single.png').exists():\n"
            "    display(Markdown('### Single-Pair Tau Ablation'))\n"
            "    display(Markdown(f\"Recommended sample ID: {recommended_samples.get('tau_ablation_sample_id', 'N/A')}\"))\n"
            "    display(Markdown(f\"Additional candidate IDs: {recommended_samples.get('tau_ablation_candidate_sample_ids', [])}\"))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'tau_ablation_single.png')))\n"
            "    display(Markdown('Suggested caption: for one fixed conflict pair, moving tau later preserves more structure, while moving tau earlier allows stronger semantic takeover.'))\n"
            "\n"
            "if (FIGURES_DIR / 'tau_sweep_examples.png').exists():\n"
            "    display(Markdown('### Tau Sweep Examples'))\n"
            "    display(Markdown(f\"Recommended sample IDs: {recommended_samples.get('tau_sweep_sample_ids', [])}\"))\n"
            "    display(Markdown(f\"Additional candidate IDs: {recommended_samples.get('tau_sweep_candidate_sample_ids', [])}\"))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'tau_sweep_examples.png')))\n"
            "    display(Markdown('Suggested caption: varying the switch point changes whether the model preserves more structure or more semantic appearance on the same conflict pair.'))\n"
            "\n"
            "if (FIGURES_DIR / 'final_method_gallery.png').exists():\n"
            "    display(Markdown('### Final Method Gallery'))\n"
            "    display(Markdown(f\"Recommended sample IDs: {recommended_samples.get('final_gallery_sample_ids', [])}\"))\n"
            "    display(Markdown(f\"Additional candidate IDs: {recommended_samples.get('final_gallery_candidate_sample_ids', [])}\"))\n"
            "    display(Image(filename=str(FIGURES_DIR / 'final_method_gallery.png')))\n"
            "    display(Markdown('Suggested caption: final qualitative comparison between naive combined control and the selected scheduled methods. Use these examples to discuss artifacts, wrong colors, or cases where scheduling improves the trade-off.'))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Recommended Figures For The Final Report / PPT\n"
            "Use these in roughly this order:\n"
            "1. `schedule_overview.png`: explain hard switch, tau, and sharpness before showing any results.\n"
            "2. `hard_search_tradeoff.png`: introduce the hard-switch trade-off.\n"
            "3. `hard_vs_smooth_tau_metrics.png`: compare hard and smooth schedules at matched tau values.\n"
            "4. `smooth_sharpness_metrics.png`: show how the sharpness hyperparameter changes the compromise.\n"
            "5. `final_method_bars.png`: main quantitative comparison on the full evaluation set.\n"
            "6. `control_ip_adapter_comparison.png`: slide-ready control + reference -> output examples.\n"
            "7. `tau_ablation_single.png`: single-pair ablation over different tau values.\n"
            "8. `tau_sweep_examples.png`: broader qualitative explanation of what changing the transition point does.\n"
            "9. `final_method_gallery.png`: qualitative evidence for the final selected methods.\n"
            "\n"
            "If the report is space-constrained, keep figures 1, 3, 5, 6, and 7, and compress figures 2, 4, 8, and 9 into appendix material."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "best_final = final_summary.sort_values('balanced_score', ascending=False).iloc[0]\n"
            "display(Markdown('## 6. Auto-Generated Takeaways'))\n"
            "display(Markdown(\n"
            "    f\"- Best final mode by balanced score: `{best_final['mode']}`.\\n\"\n"
            "    f\"- Full-set Canny Edge MSE mean: `{best_final['canny_mse_mean']:.4f}`.\\n\"\n"
            "    f\"- Full-set CLIP similarity mean: `{best_final['clip_similarity_mean']:.4f}`.\\n\"\n"
            "    f\"- Compare this against the pairwise summary table to decide whether the smooth schedule is meaningfully better than naive or simply trades one objective for the other.\"\n"
            "))"
        )
    )

    nb["cells"] = cells
    nbf.write(nb, args.output.open("w", encoding="utf-8"))
    print(f"Wrote notebook to {args.output}")


if __name__ == "__main__":
    main()
