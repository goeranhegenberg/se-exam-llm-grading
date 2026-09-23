"""Synthese-Abbildung (Abbildung 1): exakte Uebereinstimmung des Hauptmodells
je Prompt-Version auf allen drei Datensaetzen (Benchmark, menschlich verfasst,
authentisch) sowie der Ensemble-Median (V6). Liest die drei Summary-Dateien
der Auswertungen.

Aufruf (aus dem Paketverzeichnis):
    python -m eval.overview_figure
"""
import argparse
import json

import numpy as np

from eval.analyze import COLORS, ENSEMBLE, VERSION_LABEL, new_figure, save_figure
from src.config import resolve_path

SYSTEMS = [("main", v) for v in VERSION_LABEL] + [ENSEMBLE]
XLABELS = [lab.replace(" ", "\n", 1) for lab in VERSION_LABEL.values()] + ["V6\nEnsemble"]
DATASETS = [("results/summary.json", "Benchmark (162)", COLORS["main"]),
            ("results/human_summary.json", "menschlich verfasst (33)", COLORS["other"]),
            ("results/real_summary.json", "authentisch (35)", COLORS["ensemble"])]


def exact_of(summary, model, version):
    for m in summary["metrics"]:
        if m["model_label"] == model and m["prompt_version"] == version:
            return 100 * m["exact"]
    return np.nan


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures/exact_overview.pdf")
    args = p.parse_args(argv)

    x = np.arange(len(SYSTEMS))
    width = 0.8 / len(DATASETS)
    fig, ax = new_figure((5.2, 2.5))
    for i, (path, label, color) in enumerate(DATASETS):
        summary = json.loads(resolve_path(path).read_text(encoding="utf-8"))
        vals = [exact_of(summary, *s) for s in SYSTEMS]
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
    ax.axvline(len(SYSTEMS) - 1.5, color="grey", lw=0.6, ls=":")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=6.5, frameon=False)
    save_figure(fig, args.out)
    print("Abbildung geschrieben:", resolve_path(args.out))


if __name__ == "__main__":
    main()
