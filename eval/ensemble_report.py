"""Wertet das V6-Ensemble auf dem Benchmark aus und schreibt die
Vergleichstabelle (Tabelle 3).

Vergleicht die drei Einzelmodelle (jeweils V5) mit dem Median-Ensemble.
Zusaetzlich Diagnostik: Anteil der Antworten, bei denen die drei Bewerter
uneinig sind, und die Korrektheit der Systeme auf diesem Uneinigkeits-Subset.
Kennzahlen werden je Lauf (``results/raw/run<N>/``) berechnet und gemittelt.

Erzeugt:
  * results/tables/metrics_ensemble.tex
Aufruf (aus dem Paketverzeichnis):
  python -m eval.ensemble_report
"""
import argparse

from eval.analyze import (ENSEMBLE_SYSTEM, SYSTEM_KEYS, V5_SYSTEMS, _fmt,
                          disagreement_over_runs, exact_rate, load_runs,
                          mean_over_runs, metric_cells, point_estimates,
                          system_rows, write_tabular)

CLEAR = {0, 2, 4}   # eindeutig bepunktete Antworten
BORDER = {1, 3}     # grenzwertige Antworten
SYSTEMS = V5_SYSTEMS + [ENSEMBLE_SYSTEM]


def extra_metrics(sub):
    return {"exact_border": exact_rate(sub[sub["gt_points"].isin(BORDER)]),
            "exact_clear": exact_rate(sub[sub["gt_points"].isin(CLEAR)])}


def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung des V6-Ensembles.")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--out-table", default="results/tables/metrics_ensemble.tex")
    args = p.parse_args(argv)

    runs = load_runs(args.raw)
    pes = [point_estimates(df) for _, df in runs]
    rep, _ = mean_over_runs([system_rows(pe, SYSTEMS, extra=extra_metrics) for pe in pes],
                            SYSTEM_KEYS)

    print(f"\n== V6-Vergleich ({len(runs)} Lauf/Läufe) ==")
    print(f"  {'System':28} {'MAE':>5} {'Exakt':>6} {'pm1':>6} {'runσ':>5} "
          f"{'exGrenz':>7} {'exKlar':>6}")
    for _, r in rep.iterrows():
        print(f"  {r['label']:28} {r['mae']:5.3f} {100*r['exact']:5.1f}% "
              f"{100*r['within1']:5.1f}% {r['run_std']:5.3f} "
              f"{100*r['exact_border']:6.1f}% {100*r['exact_clear']:5.1f}%")

    write_tabular(args.out_table, "lrrrrr",
                  r"System & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ & \makecell{Exakt\\(grenzw.)}",
                  [(r["label"], metric_cells(r) + [f"{_fmt(100*r['exact_border'], 1)}\\%"])
                   for _, r in rep.iterrows()],
                  midrule_after={V5_SYSTEMS[-1][2]}, bold={ENSEMBLE_SYSTEM[2]})
    disagreement_over_runs(pes, print_prefix="    ")


if __name__ == "__main__":
    main()
