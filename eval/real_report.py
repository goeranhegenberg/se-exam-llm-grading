"""Wertet die Validierungslaeufe auf den authentischen Studierendenantworten aus.

Der Datensatz ``dataset_real`` (fuenf offene Aufgaben einer realen
Softwaretechnik-I-Klausur, 35 wortgetreu transkribierte Antworten aus sieben
anonymisierten Boegen, rubrikweise von Hand bepunktet) ist der dritte und
haerteste Validierungsdatensatz: Die Antworten stammen weder von einem LLM noch
von den Autoren, sondern von Pruefungsteilnehmenden.

Weil die Aufgaben unterschiedliche Maximalpunktzahlen haben (2--10 Punkte),
waere ein blosser MAE ueber alle Aufgaben nicht vergleichbar. Zusaetzlich wird
daher der skalennormierte Fehler NMAE berichtet (absoluter Fehler geteilt durch
die Maximalpunktzahl der Aufgabe). Keine Formulierungsvarianten, daher keine
Para-σ-Spalte.

Erzeugt:
  * results/tables/metrics_real.tex   -- V1-V5 (Hauptmodell) + V5 der beiden
                                         anderen Modelle + V6 (Ensemble-Median)
  * results/tables/per_task_real.tex  -- exakte Quote und NMAE je Aufgabe
  * results/real_summary.json         -- Kennzahlen maschinenlesbar

Aufruf (aus dem Paketverzeichnis):
    python -m eval.real_report
    python -m eval.real_report --raw results/raw_real
"""
import argparse

import pandas as pd

from eval.analyze import (ENSEMBLE, ENSEMBLE_SYSTEM, MAIN_SYSTEMS, V5_SYSTEMS,
                          _fmt, disagreement_stats, exact_rate, load_raw,
                          metric_cells, point_estimates, system_rows,
                          write_summary, write_tabular)
from src.dataset import load_questions

SYSTEMS = MAIN_SYSTEMS + V5_SYSTEMS[1:] + [ENSEMBLE_SYSTEM]

# Spalten der Aufgaben-Tabelle: (model_label, prompt_version, Label)
TASK_SYSTEMS = [("main", "v1_baseline", "V1"), ("main", "v3_thinking", "V3"),
                ("main", "v5_rubrik", "V5"), (*ENSEMBLE, "V6")]


def nmae(sub):
    """Skalennormierter mittlerer absoluter Fehler (Fehler / max_points)."""
    v = sub.dropna(subset=["pe"])
    if v.empty:
        return float("nan")
    return float(((v["pe"] - v["gt_points"]).abs() / v["max_points"]).mean())


def extra_metrics(sub):
    v = sub.dropna(subset=["pe"])
    # Bias: mittlere vorzeichenbehaftete Abweichung, negativ = strenger als Referenz.
    return {"nmae": nmae(sub),
            "bias": float((v["pe"] - v["gt_points"]).mean()) if len(v) else float("nan")}


def per_task(pe, questions):
    """Exakte Quote und NMAE je Aufgabe fuer die Spalten aus ``TASK_SYSTEMS``."""
    labels = {q["question_id"]: (q.get("short_label") or q["topic"], q["max_points"])
              for q in questions}
    rows = []
    for i, qid in enumerate(sorted(pe["question_id"].unique())):
        topic, mx = labels[qid]
        row = {"question_id": qid, "label": f"A{i+1} {topic}", "max_points": mx}
        for model, version, name in TASK_SYSTEMS:
            sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)
                     & (pe["question_id"] == qid)]
            row[f"exact_{name}"] = exact_rate(sub)
            row[f"nmae_{name}"] = nmae(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung dataset_real.")
    p.add_argument("--raw", default="results/raw_real")
    p.add_argument("--dataset-dir", default="dataset_real")
    p.add_argument("--out-table", default="results/tables/metrics_real.tex")
    p.add_argument("--out-task-table", default="results/tables/per_task_real.tex")
    p.add_argument("--out-summary", default="results/real_summary.json")
    args = p.parse_args(argv)

    questions = load_questions(args.dataset_dir)
    df = load_raw(args.raw)
    pe = point_estimates(df)
    rep = system_rows(pe, SYSTEMS, extra=extra_metrics)

    print("== Validierung auf dataset_real ==")
    print(f"  {'System':26} {'MAE':>5} {'NMAE':>6} {'Exakt':>6} {'pm1':>6} "
          f"{'runσ':>5} {'Bias':>6}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:26} {r['mae']:5.3f} {r['nmae']:6.3f} "
              f"{100*r['exact']:5.1f}% {100*r['within1']:5.1f}% "
              f"{r['run_std']:5.3f} {r['bias']:+6.2f}")

    tasks = per_task(pe, questions)
    print("\n  Aufgabe                            "
          + "  ".join(f"{n:>12}" for _, _, n in TASK_SYSTEMS))
    for _, r in tasks.iterrows():
        cells = "  ".join(f"{100*r[f'exact_{n}']:6.1f}% ({r[f'nmae_{n}']:.2f})"
                          for _, _, n in TASK_SYSTEMS)
        print(f"  {r['label'][:32]:32}  {cells}")

    # Tabelle 4: Kennzahlen je System (MAE, NMAE, Exakt, ±1, Lauf-σ)
    def cells(r):
        c = metric_cells(r)
        return [c[0], _fmt(r["nmae"], 3), *c[1:]]
    write_tabular(args.out_table, "lrrrrr",
                  r"System & MAE & NMAE & Exakt & $\pm$1 & Lauf-$\sigma$",
                  [(r["label"], cells(r)) for _, r in rep.iterrows()],
                  midrule_after={MAIN_SYSTEMS[-1][2], V5_SYSTEMS[-1][2]},
                  bold={ENSEMBLE_SYSTEM[2]})
    # Tabelle 5: exakte Quote in Prozent (NMAE) je Aufgabe
    write_tabular(args.out_task_table, "lr" + "r" * len(TASK_SYSTEMS),
                  "Aufgabe & Max. & " + " & ".join(n for _, _, n in TASK_SYSTEMS),
                  [(r["label"], [str(int(r["max_points"]))]
                    + [f"{_fmt(100*r[f'exact_{n}'], 0)} ({_fmt(r[f'nmae_{n}'], 2)})"
                       for _, _, n in TASK_SYSTEMS])
                   for _, r in tasks.iterrows()])

    # Parse-Ausfaelle und Regex-Rettungen (Thinking-Modus liefert teils kein
    # oder korruptes JSON) -- fuer die Transparenz im Text.
    graded = df[df["model_label"] != "ensemble"]
    fails = graded[~graded["parse_ok"].astype(bool)]
    salvaged = graded[graded.get("parse_salvaged", pd.Series(False, index=graded.index)).fillna(False).astype(bool)]
    print(f"\nNicht auswertbare Aufrufe: {len(fails)}/{len(graded)} "
          f"({100*len(fails)/len(graded):.1f}%)")
    for (model, version), g in fails.groupby(["model_label", "prompt_version"]):
        print(f"  {model}/{version}: {len(g)}")
    print(f"Per Regex gerettete (syntaktisch korrupte) Antworten: {len(salvaged)}")
    for (model, version), g in salvaged.groupby(["model_label", "prompt_version"]):
        print(f"  {model}/{version}: {len(g)}")

    summary = {
        "metrics": rep.to_dict(orient="records"),
        "per_task": tasks.to_dict(orient="records"),
        "n_records": int(len(df)),
        "n_graded_calls": int(len(graded)),
        "n_parse_fail": int(len(fails)),
        "n_parse_salvaged": int(len(salvaged)),
        "n_answers_without_valid_rating": int((pe["n_ok"] == 0).sum()),
        "models": sorted(df["model_label"].unique().tolist()),
        "tokens": {
            k: int(df["usage"].dropna().apply(
                lambda u, k=k: (u or {}).get(k, 0) or 0).sum())
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        } if "usage" in df.columns else {},
    }
    dis = disagreement_stats(pe)
    if dis:
        summary["disagreement"] = dis
    write_summary(args.out_summary, summary)
    return summary


if __name__ == "__main__":
    main()
