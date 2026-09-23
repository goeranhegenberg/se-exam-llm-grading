"""Synthese-Abbildung: exakte Uebereinstimmung des Hauptmodells je Prompt-Version
auf allen drei Datensaetzen (Benchmark, menschlich verfasst, authentisch) sowie
der Ensemble-Median (V6). Liest die drei summary-Dateien der Auswertungen.

Aufruf (aus implementation/):
    python -m eval.overview_figure
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import resolve_path

VERSIONS = ["v1_baseline", "v2_begruendung", "v3_thinking", "v4_fewshot",
            "v5_rubrik", "v6_median"]
XLABELS = ["V1\nBaseline", "V2\nBegr.", "V3\nThinking", "V4\nFew-Shot",
           "V5\nRubrik", "V6\nEnsemble"]
DATASETS = [("results/summary.json", "Benchmark (162)", "#4C72B0"),
            ("results/human_summary.json", "menschlich verfasst (33)", "#55A868"),
            ("results/real_summary.json", "authentisch (35)", "#C44E52")]


def _exact(summary, version):
    model = "ensemble" if version == "v6_median" else "main"
    for m in summary["metrics"]:
        if m["model_label"] == model and m["prompt_version"] == version:
            return 100 * m["exact"]
    return np.nan


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures/exact_overview.pdf")
    args = p.parse_args(argv)

    x = np.arange(len(VERSIONS))
    width = 0.8 / len(DATASETS)
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for i, (path, label, color) in enumerate(DATASETS):
        summary = json.loads(resolve_path(path).read_text(encoding="utf-8"))
        vals = [_exact(summary, v) for v in VERSIONS]
        pos = x + (i - (len(DATASETS) - 1) / 2) * width
        ax.bar(pos, vals, width, label=label, color=color)
        for xp, val in zip(pos, vals):
            if not np.isnan(val):
                ax.text(xp, val + 1, f"{val:.0f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=7)
    ax.set_ylabel("Exakte Übereinstimmung (%)", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=7)
    ax.axvline(4.5, color="grey", lw=0.6, ls=":")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=6.5, frameon=False)
    fig.tight_layout()
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print("Abbildung geschrieben:", out)


if __name__ == "__main__":
    main()
