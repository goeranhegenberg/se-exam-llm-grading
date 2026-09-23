"""Wertet die Validierungslaeufe auf den authentischen Studierendenantworten aus.

Der Datensatz ``dataset_real`` (fuenf offene Aufgaben einer realen
Softwaretechnik-I-Klausur, 35 wortgetreu transkribierte Antworten aus sieben
anonymisierten Boegen, rubrikweise von Hand bepunktet) ist der dritte und
haerteste Validierungsdatensatz: Die Antworten stammen weder von einem LLM noch
von den Autoren, sondern von Pruefungsteilnehmenden.

Weil die Aufgaben unterschiedliche Maximalpunktzahlen haben (2--10 Punkte),
waere ein blosser MAE ueber alle Aufgaben nicht vergleichbar. Zusaetzlich wird
daher der skalennormierte Fehler NMAE berichtet (absoluter Fehler geteilt durch
die Maximalpunktzahl der Aufgabe). Kein Paraphrasen-Anteil, daher entfaellt --
wie bei ``human_report`` -- die Para-Sigma-Spalte.

Erzeugt:
  * results/tables/metrics_real.tex   -- V1-V5 (Hauptmodell) + V5 der beiden
                                         anderen Modelle + V6 (Ensemble-Median)
  * results/tables/per_task_real.tex  -- exakte Quote und NMAE je Aufgabe
  * results/figures/exact_by_system_real.pdf -- exakte Quote je System
  * results/figures/exact_by_task_real.pdf   -- exakte Quote je Aufgabe
  * results/real_summary.json         -- Kennzahlen maschinenlesbar

Aufruf (aus implementation/):
    python -m eval.real_report
    python -m eval.real_report --raw results/raw_real
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.analyze import (VERSION_LABEL, _fmt, load_raw, metrics_for,
                          point_estimates)
from src.config import resolve_path
from src.dataset import load_questions

# Zeilen der Tabelle: (model_label, prompt_version) -> Anzeigename
MAIN_VERSIONS = list(VERSION_LABEL)  # V1-V5, Hauptmodell
EXTRA_SYSTEMS = [
    (("sensitivity", "v5_rubrik"), "Mistral Small 4 (V5)"),
    (("tertiary", "v5_rubrik"), "GPT-OSS 120B (V5)"),
    (("ensemble", "v6_median"), "Ensemble (V6, Median)"),
]

# Spalten der Aufgaben-Tabelle/-Abbildung: (model_label, prompt_version, Label)
TASK_SYSTEMS = [
    ("main", "v1_baseline", "V1"),
    ("main", "v3_thinking", "V3"),
    ("main", "v5_rubrik", "V5"),
    ("ensemble", "v6_median", "V6"),
]

BLUE, GREEN, RED, PURPLE = "#4C72B0", "#55A868", "#C44E52", "#8172B2"


def _nmae(sub):
    """Skalennormierter mittlerer absoluter Fehler (Fehler / max_points)."""
    v = sub.dropna(subset=["pe"])
    if v.empty:
        return float("nan")
    return float(((v["pe"] - v["gt_points"]).abs() / v["max_points"]).mean())


def _row(pe, model, version, label):
    sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
    if sub.empty:
        return None
    m = metrics_for(sub)
    if m is None:
        return None
    m.update({"label": label, "model_label": model, "prompt_version": version,
              "nmae": _nmae(sub)})
    # Exakte Quote je Punktstufe (null/teil/voll) fuer den Text.
    for lvl in ["null", "teil", "voll"]:
        s = sub[(sub["level"] == lvl)].dropna(subset=["pe"])
        m[f"exact_{lvl}"] = (float((s["pe"].round() == s["gt_points"]).mean())
                             if len(s) else float("nan"))
    # Mittlere (vorzeichenbehaftete) Abweichung: negativ = strenger als Referenz.
    v = sub.dropna(subset=["pe"])
    m["bias"] = float((v["pe"] - v["gt_points"]).mean()) if len(v) else float("nan")
    return m


def per_task(pe, questions):
    """Exakte Quote und NMAE je Aufgabe fuer die Spalten aus ``TASK_SYSTEMS``."""
    labels = {q["question_id"]: (q.get("short_label") or q["topic"],
                                 q["max_points"]) for q in questions}
    rows = []
    for qid in sorted(pe["question_id"].unique()):
        topic, mx = labels.get(qid, (qid, np.nan))
        row = {"question_id": qid, "topic": topic, "max_points": mx}
        for model, version, name in TASK_SYSTEMS:
            sub = pe[(pe["model_label"] == model)
                     & (pe["prompt_version"] == version)
                     & (pe["question_id"] == qid)].dropna(subset=["pe"])
            if sub.empty:
                row[f"exact_{name}"] = float("nan")
                row[f"nmae_{name}"] = float("nan")
                continue
            row[f"exact_{name}"] = float(
                (sub["pe"].round() == sub["gt_points"]).mean())
            row[f"nmae_{name}"] = _nmae(sub)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ausgabe
# --------------------------------------------------------------------------- #
def write_metrics_table(rep, out):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"System & MAE & NMAE & Exakt & $\pm$1 & Lauf-$\sigma$ \\",
             r"\midrule"]
    for _, r in rep.iterrows():
        lab = r["label"]
        if lab.startswith("Ensemble"):
            lab = rf"\textbf{{{lab}}}"
        lines.append(
            f"{lab} & {_fmt(r['mae'])} & {_fmt(r['nmae'], 3)} & "
            f"{_fmt(100*r['exact'],1)}\\% & {_fmt(100*r['within1'],1)}\\% & "
            f"{_fmt(r['run_std'])} \\\\")
        if r["label"].startswith("V5") or r["label"].startswith("GPT-OSS"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def write_task_table(tasks, out):
    lines = [r"\begin{tabular}{lr" + "r" * len(TASK_SYSTEMS) + "}", r"\toprule",
             "Aufgabe & Max. & "
             + " & ".join(n for _, _, n in TASK_SYSTEMS) + r" \\",
             r"\midrule"]
    for _, r in tasks.iterrows():
        cells = []
        for _, _, n in TASK_SYSTEMS:
            cells.append(f"{_fmt(100*r[f'exact_{n}'], 0)} "
                         f"({_fmt(r[f'nmae_{n}'], 2)})")
        lines.append(f"{r['label']} & {int(r['max_points'])} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _save(fig, out):
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_by_system(rep, out):
    """Exakte Uebereinstimmung je System; eingefaerbt nach Bewerter-Gruppe."""
    short = {"V1 Baseline (GLM 5.2)": "V1", "V2 Begr. (GLM 5.2)": "V2",
             "V3 Thinking (GLM 5.2)": "V3", "V4 Few-Shot (GLM 5.2)": "V4",
             "V5 Rubrik (GLM 5.2)": "V5", "Mistral Small 4 (V5)": "Mistral\n(V5)",
             "GPT-OSS 120B (V5)": "GPT-OSS\n(V5)",
             "Ensemble (V6, Median)": "V6\nMedian"}
    labels, vals, colors = [], [], []
    for _, r in rep.iterrows():
        labels.append(short.get(r["label"], r["label"]))
        vals.append(100 * r["exact"])
        colors.append(RED if r["model_label"] == "ensemble"
                      else GREEN if r["model_label"] != "main" else BLUE)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(labels, vals, color=colors)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel("Exakte Übereinstimmung (%)")
    ax.set_title("Exakte Übereinstimmung je System")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GREEN, RED)]
    ax.legend(handles, ["GLM 5.2", "weitere Bewerter", "Ensemble"],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=7, frameon=False)
    _save(fig, out)


def plot_by_task(tasks, out):
    """Exakte Uebereinstimmung je Aufgabe, gruppiert nach System."""
    x = np.arange(len(tasks))
    width = 0.8 / len(TASK_SYSTEMS)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, ((_, _, name), color) in enumerate(
            zip(TASK_SYSTEMS, (BLUE, GREEN, RED, PURPLE))):
        vals = [100 * v for v in tasks[f"exact_{name}"]]
        ax.bar(x + (i - (len(TASK_SYSTEMS) - 1) / 2) * width, vals, width,
               label=name, color=color)
    ax.set_ylabel("Exakte Übereinstimmung (%)")
    ax.set_title("Exakte Übereinstimmung je Aufgabe")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['label']}\n({int(r['max_points'])} P.)"
         for _, r in tasks.iterrows()], fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", ncol=len(TASK_SYSTEMS), fontsize=7,
              frameon=False)
    _save(fig, out)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung dataset_real.")
    p.add_argument("--raw", default="results/raw_real")
    p.add_argument("--dataset-dir", default="dataset_real")
    p.add_argument("--out-table", default="results/tables/metrics_real.tex")
    p.add_argument("--out-task-table", default="results/tables/per_task_real.tex")
    p.add_argument("--out-fig-system",
                   default="results/figures/exact_by_system_real.pdf")
    p.add_argument("--out-fig-task",
                   default="results/figures/exact_by_task_real.pdf")
    p.add_argument("--out-summary", default="results/real_summary.json")
    args = p.parse_args(argv)

    questions = load_questions(args.dataset_dir)
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

    print("== Validierung auf dataset_real ==")
    print(f"  {'System':26} {'MAE':>5} {'NMAE':>6} {'Exakt':>6} {'pm1':>6} "
          f"{'runσ':>5} {'Bias':>6}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:26} {r['mae']:5.3f} {r['nmae']:6.3f} "
              f"{100*r['exact']:5.1f}% {100*r['within1']:5.1f}% "
              f"{r['run_std']:5.3f} {r['bias']:+6.2f}")

    # ---- Aufgaben-Ebene ----
    tasks = per_task(pe, questions)
    # Kurzlabel A1..A5 in Aufgabenreihenfolge
    tasks = tasks.sort_values("question_id").reset_index(drop=True)
    tasks["label"] = [f"A{i+1} {t}" for i, t in enumerate(tasks["topic"])]

    print("\n  Aufgabe                            "
          + "  ".join(f"{n:>12}" for _, _, n in TASK_SYSTEMS))
    for _, r in tasks.iterrows():
        cells = "  ".join(f"{100*r[f'exact_{n}']:6.1f}% ({r[f'nmae_{n}']:.2f})"
                          for _, _, n in TASK_SYSTEMS)
        print(f"  {r['label'][:32]:32}  {cells}")

    # ---- Tabellen und Abbildungen ----
    t_out = resolve_path(args.out_table)
    write_metrics_table(rep, t_out)
    print(f"\nTabelle geschrieben: {t_out}")

    tt_out = resolve_path(args.out_task_table)
    write_task_table(tasks, tt_out)
    print(f"Tabelle geschrieben: {tt_out}")

    f1 = resolve_path(args.out_fig_system)
    plot_by_system(rep, f1)
    f2 = resolve_path(args.out_fig_task)
    plot_by_task(tasks, f2)
    print(f"Abbildungen geschrieben: {f1}, {f2}")

    # ---- Parse-Ausfaelle (Thinking-Modus liefert gelegentlich kein JSON) ----
    n_fail = int((~df["parse_ok"].astype(bool)).sum())
    fails = df[~df["parse_ok"].astype(bool)]
    print(f"\nNicht auswertbare Aufrufe: {n_fail}/{len(df)} "
          f"({100*n_fail/len(df):.1f}%)")
    if n_fail:
        for (model, version), g in fails.groupby(["model_label",
                                                  "prompt_version"]):
            print(f"  {model}/{version}: {len(g)}")
    # Antworten ohne jede gueltige Bewertung
    no_valid = pe[pe["n_ok"] == 0]

    summary = {
        "metrics": rep.to_dict(orient="records"),
        "per_task": tasks.to_dict(orient="records"),
        "n_records": int(len(df)),
        "n_parse_fail": n_fail,
        "n_answers_without_valid_rating": int(len(no_valid)),
        "models": sorted(df["model_label"].unique().tolist()),
        "tokens": {
            k: int(df["usage"].dropna().apply(
                lambda u, k=k: (u or {}).get(k, 0) or 0).sum())
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        } if "usage" in df.columns else {},
    }

    # ---- Uneinigkeit der drei Bewerter (Antwortebene) ----
    def by_answer(model, version):
        sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
        return sub.set_index(["question_id", "answer_id"])

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
    outs.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                               default=float), encoding="utf-8")
    print(f"Summary geschrieben: {outs}")
    return summary


if __name__ == "__main__":
    main()
