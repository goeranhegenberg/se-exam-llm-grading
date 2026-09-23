"""Wertet die Validierungslaeufe auf dem menschlich erstellten Datensatz aus.

Der Datensatz ``dataset_human`` (12 Fragen einer realen SE-Klausur, Antworten
von den Autoren verfasst und von Hand nach Rubrik bepunktet) dient als
Ueberpruefung, ob sich die Befunde des synthetischen Benchmarks auf
menschlich geschriebene Antworten uebertragen. Kein Paraphrasen-Anteil,
daher entfaellt die Para-Sigma-Spalte.

Erzeugt:
  * results/tables/metrics_human.tex  -- V1-V5 (Hauptmodell) + V5 der beiden
                                         anderen Modelle + V6 (Ensemble-Median)
  * results/human_summary.json        -- Kennzahlen maschinenlesbar

Aufruf (aus implementation/):
    python -m eval.human_report
    python -m eval.human_report --raw results/raw_human
"""
import argparse
import json

import numpy as np
import pandas as pd

from eval.analyze import (VERSION_LABEL, _fmt, load_raw, metrics_for,
                          point_estimates)
from src.config import resolve_path

# Zeilen der Tabelle: (model_label, prompt_version) -> Anzeigename
MAIN_VERSIONS = list(VERSION_LABEL)  # V1-V5, Hauptmodell
EXTRA_SYSTEMS = [
    (("sensitivity", "v5_rubrik"), "Mistral Small 4 (V5)"),
    (("tertiary", "v5_rubrik"), "GPT-OSS 120B (V5)"),
    (("ensemble", "v6_median"), "Ensemble (V6, Median)"),
]


def _row(pe, model, version, label):
    sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
    if sub.empty:
        return None
    m = metrics_for(sub)
    m.update({"label": label, "model_label": model, "prompt_version": version})
    # Exakte Quote je Punktstufe (null/teil/voll) fuer den Text.
    for lvl in ["null", "teil", "voll"]:
        s = sub[(sub["level"] == lvl)].dropna(subset=["pe"])
        m[f"exact_{lvl}"] = (float((s["pe"].round() == s["gt_points"]).mean())
                             if len(s) else float("nan"))
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung dataset_human.")
    p.add_argument("--raw", default="results/raw_human")
    p.add_argument("--out-table", default="results/tables/metrics_human.tex")
    p.add_argument("--out-summary", default="results/human_summary.json")
    args = p.parse_args(argv)

    df = load_raw(args.raw)
    pe = point_estimates(df)

    rows = []
    for v in MAIN_VERSIONS:
        r = _row(pe, "main", v, VERSION_LABEL[v] + " (GLM 5.2)")
        if r:
            rows.append(r)
    for (model, version), label in EXTRA_SYSTEMS:
        r = _row(pe, model, version, label)
        if r:
            rows.append(r)
    rep = pd.DataFrame(rows)

    print("== Validierung auf dataset_human ==")
    print(f"  {'System':26} {'MAE':>5} {'Exakt':>6} {'pm1':>6} {'runσ':>5} "
          f"{'ex(teil)':>8}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:26} {r['mae']:5.3f} {100*r['exact']:5.1f}% "
              f"{100*r['within1']:5.1f}% {r['run_std']:5.3f} "
              f"{100*r['exact_teil']:7.1f}%")

    # ---- Tabelle ----
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"System & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ \\",
             r"\midrule"]
    for _, r in rep.iterrows():
        lab = r["label"]
        lines.append(
            f"{lab} & {_fmt(r['mae'])} & {_fmt(100*r['exact'],1)}\\% & "
            f"{_fmt(100*r['within1'],1)}\\% & {_fmt(r['run_std'])} \\\\")
        if r["label"].startswith("V5") or r["label"].startswith("GPT-OSS"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = resolve_path(args.out_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTabelle geschrieben: {out}")

    # ---- Uneinigkeit der drei Bewerter (Antwortebene) ----
    def by_answer(model, version):
        sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
        return sub.set_index(["question_id", "answer_id"])

    summary = {"metrics": rep.to_dict(orient="records"),
               "n_records": int(len(df)),
               "models": sorted(df["model_label"].unique().tolist())}
    med = by_answer("ensemble", "v6_median")
    if not med.empty:
        comp = pd.DataFrame({
            "glm": by_answer("main", "v5_rubrik")["pe"],
            "mis": by_answer("sensitivity", "v5_rubrik")["pe"],
            "oss": by_answer("tertiary", "v5_rubrik")["pe"],
            "med": med["pe"], "gt": med["gt_points"]}).dropna()
        dis = comp[comp[["glm", "mis", "oss"]].round().nunique(axis=1) > 1]
        print(f"\nUneinige Antworten: {len(dis)}/{len(comp)} "
              f"({100*len(dis)/len(comp):.1f}%)")
        summary["disagreement"] = {"n": int(len(dis)), "total": int(len(comp))}
        for col, name in [("glm", "GLM"), ("mis", "Mistral"),
                          ("oss", "GPT-OSS"), ("med", "Median")]:
            if len(dis):
                acc = float((dis[col].round() == dis["gt"]).mean())
                print(f"  exakt auf Uneinigkeits-Subset [{name:8}]: {100*acc:.1f}%")
                summary["disagreement"][f"exact_{col}"] = acc

    outs = resolve_path(args.out_summary)
    outs.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"Summary geschrieben: {outs}")
    return summary


if __name__ == "__main__":
    main()
