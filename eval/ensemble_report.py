"""Wertet das V6-Ensemble aus und schreibt die Vergleichstabelle.

Vergleicht die drei Einzelmodelle (jeweils V5), die Median-Baseline und das
GLM-Meta-Merge (V6) auf demselben Benchmark. Zusaetzlich Diagnostik:
  * Anteil der Antworten, bei denen die drei Bewerter uneinig sind
  * Korrektheit der Systeme auf diesem Uneinigkeits-Subset
  * wie oft das LLM-Merge vom Median abweicht und ob es dabei haeufiger
    richtig liegt (Mehrwert ueber triviale Aggregation)

Erzeugt:
  * results/tables/metrics_ensemble.tex
Aufruf (aus implementation/, mit PYTHONPATH=.):
  python -m eval.ensemble_report
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eval.analyze import load_raw, point_estimates, metrics_for
from src.config import resolve_path

CLEAR = {0, 2, 4}
BORDER = {1, 3}

# (model_label, prompt_version) -> Anzeigename
SYSTEMS = [
    (("main", "v5_rubrik"), "GLM 5.2 (V5)"),
    (("sensitivity", "v5_rubrik"), "Mistral Small 4 (V5)"),
    (("tertiary", "v5_rubrik"), "GPT-OSS 120B (V5)"),
    (("ensemble", "v6_median"), "Ensemble (V6, Median)"),
]


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def exact_subset(sub, gtset):
    v = sub.dropna(subset=["pe"])
    v = v[v["gt_points"].isin(gtset)]
    if v.empty:
        return float("nan")
    return float((v["pe"].round() == v["gt_points"]).mean())


def pe_by_answer(pe, key):
    sub = pe[(pe["model_label"] == key[0]) & (pe["prompt_version"] == key[1])]
    return sub.set_index(["question_id", "answer_id"])


def main():
    df = load_raw("results/raw")
    pe = point_estimates(df)

    # ---- Vergleichstabelle ----
    rows = []
    for key, label in SYSTEMS:
        sub = pe[(pe["model_label"] == key[0]) & (pe["prompt_version"] == key[1])]
        if sub.empty:
            print(f"WARN: keine Daten fuer {key}")
            continue
        m = metrics_for(sub)
        rows.append({
            "label": label, "mae": m["mae"], "exact": m["exact"],
            "within1": m["within1"], "run_std": m["run_std"],
            "exact_border": exact_subset(sub, BORDER),
            "exact_clear": exact_subset(sub, CLEAR),
        })
    rep = pd.DataFrame(rows)
    print("\n== V6-Vergleich ==")
    print(f"  {'System':28} {'MAE':>5} {'Exakt':>6} {'pm1':>6} {'runσ':>5} "
          f"{'exGrenz':>7} {'exKlar':>6}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:28} {r['mae']:5.3f} {100*r['exact']:5.1f}% "
              f"{100*r['within1']:5.1f}% {r['run_std']:5.3f} "
              f"{100*r['exact_border']:6.1f}% {100*r['exact_clear']:5.1f}%")

    # ---- Tabelle schreiben ----
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"System & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ & \makecell{Exakt\\(grenzw.)} \\",
             r"\midrule"]
    for _, r in rep.iterrows():
        orig = r["label"]
        lab = (r"\textbf{" + orig + "}") if orig.startswith("Ensemble") else orig
        lines.append(
            f"{lab} & {_fmt(r['mae'])} & {_fmt(100*r['exact'],1)}\\% & "
            f"{_fmt(100*r['within1'],1)}\\% & {_fmt(r['run_std'])} & "
            f"{_fmt(100*r['exact_border'],1)}\\% \\\\")
        if orig.startswith("GPT-OSS"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = resolve_path("results/tables/metrics_ensemble.tex")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTabelle geschrieben: {out}")

    # ---- Disagreement-Diagnostik (auf Antwortebene, pe = Median ueber Reps) ----
    glm = pe_by_answer(pe, ("main", "v5_rubrik"))["pe"]
    mis = pe_by_answer(pe, ("sensitivity", "v5_rubrik"))["pe"]
    oss = pe_by_answer(pe, ("tertiary", "v5_rubrik"))["pe"]
    med = pe_by_answer(pe, ("ensemble", "v6_median"))
    gt = med["gt_points"]
    comp = pd.DataFrame({"glm": glm, "mis": mis, "oss": oss,
                         "med": med["pe"], "gt": gt}).dropna()
    disagree = comp[(comp[["glm", "mis", "oss"]].round().nunique(axis=1) > 1)]
    n_dis = len(disagree)
    print(f"\n== Uneinigkeit der drei Bewerter (Antwortebene, n={len(comp)}) ==")
    print(f"  Uneinige Antworten: {n_dis} ({100*n_dis/len(comp):.1f}%)")
    if n_dis:
        for col, name in [("glm", "GLM"), ("mis", "Mistral"), ("oss", "GPT-OSS"),
                          ("med", "Median")]:
            acc = (disagree[col].round() == disagree["gt"]).mean()
            print(f"    exakt auf Uneinigkeits-Subset [{name:8}]: {100*acc:.1f}%")


if __name__ == "__main__":
    main()
