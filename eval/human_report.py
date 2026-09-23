"""Wertet die Validierungslaeufe auf dem menschlich erstellten Datensatz aus.

Der Datensatz ``dataset_human`` (12 Fragen einer realen SE-Klausur, Antworten
von den Autoren verfasst und von Hand nach Rubrik bepunktet) dient als
Ueberpruefung, ob sich die Befunde des Benchmarks auf menschlich geschriebene
Antworten uebertragen. Keine Formulierungsvarianten, daher keine Para-σ-Spalte.

Erzeugt:
  * results/tables/metrics_human.tex  -- V1-V5 (Hauptmodell) + V5 der beiden
                                         anderen Modelle + V6 (Ensemble-Median)
  * results/human_summary.json        -- Kennzahlen maschinenlesbar

Aufruf (aus dem Paketverzeichnis):
    python -m eval.human_report
    python -m eval.human_report --raw results/raw_human
"""
import argparse

from eval.analyze import (ENSEMBLE_SYSTEM, MAIN_SYSTEMS, V5_SYSTEMS,
                          disagreement_stats, load_raw, metric_cells,
                          point_estimates, system_rows, write_summary,
                          write_tabular)

SYSTEMS = MAIN_SYSTEMS + V5_SYSTEMS[1:] + [ENSEMBLE_SYSTEM]


def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung dataset_human.")
    p.add_argument("--raw", default="results/raw_human")
    p.add_argument("--out-table", default="results/tables/metrics_human.tex")
    p.add_argument("--out-summary", default="results/human_summary.json")
    args = p.parse_args(argv)

    df = load_raw(args.raw)
    pe = point_estimates(df)
    rep = system_rows(pe, SYSTEMS)

    print("== Validierung auf dataset_human ==")
    print(f"  {'System':26} {'MAE':>5} {'Exakt':>6} {'pm1':>6} {'runσ':>5} "
          f"{'ex(teil)':>8}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:26} {r['mae']:5.3f} {100*r['exact']:5.1f}% "
              f"{100*r['within1']:5.1f}% {r['run_std']:5.3f} "
              f"{100*r['exact_teil']:7.1f}%")

    write_tabular(args.out_table, "lrrrr",
                  r"System & MAE & Exakt & $\pm$1 & Lauf-$\sigma$",
                  [(r["label"], metric_cells(r)) for _, r in rep.iterrows()],
                  midrule_after={MAIN_SYSTEMS[-1][2], V5_SYSTEMS[-1][2]})

    summary = {"metrics": rep.to_dict(orient="records"),
               "n_records": int(len(df)),
               "models": sorted(df["model_label"].unique().tolist())}
    dis = disagreement_stats(pe)
    if dis:
        summary["disagreement"] = dis
    write_summary(args.out_summary, summary)
    return summary


if __name__ == "__main__":
    main()
