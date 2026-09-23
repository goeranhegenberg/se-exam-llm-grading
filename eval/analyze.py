"""Wertet die JSONL-Rohdaten des Benchmark-Laufs aus und stellt die
gemeinsamen Bausteine (Laden, Punkt-Schätzwerte, Kennzahlen, Tabellen- und
Abbildungsausgabe, Anzeigenamen) für alle Auswertungsskripte bereit.

Die Rohdaten liegen je unabhängigem Lauf in einem Unterordner ``run<N>/``
(``results/raw/run1``, ``run2``, ...). Alle Kennzahlen werden je Lauf berechnet
und anschließend über die Läufe gemittelt; die Streuung zwischen den Läufen
wird mitgeführt (``*_std`` in den Summaries).

Erzeugt:
  * results/tables/metrics_models.tex   -- beide Modelle, alle Kennzahlen (Tabelle 2)
  * results/tables/dataset_stats.tex    -- Kennzahlen des Datensatzes
  * results/figures/confusion_best.pdf  -- Konfusionsmatrix über die Punktstufen
                                           (Hauptmodell, alle Versionen und Läufe)
  * results/summary.json                -- alle Kennzahlen maschinenlesbar

Aufruf:
    python -m eval.analyze                 # liest results/raw/run*/
    python -m eval.analyze --raw <pfad>    # Verzeichnis mit run*/-Ordnern, ein
                                           # einzelnes Laufverzeichnis oder eine Datei
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.util import iter_jsonl

LEVELS = ["null", "teil", "voll"]
LEVEL_LABEL = {"null": "0 P.", "teil": "Teil", "voll": "voll"}

# --------------------------------------------------------------------------- #
# Anzeigenamen (einzige Quelle für Tabellen, Abbildungen und Konsolenausgaben)
# --------------------------------------------------------------------------- #
VERSION_LABEL = {
    "v1_baseline": "V1 Baseline",
    "v2_begruendung": "V2 Begr.",
    "v3_thinking": "V3 Thinking",
    "v4_fewshot": "V4 Few-Shot",
    "v5_rubrik": "V5 Rubrik",
}
MODEL_NAME = {"main": "GLM 5.2", "sensitivity": "Mistral Small 4",
              "tertiary": "GPT-OSS 120B"}
GRADERS = ["main", "sensitivity", "tertiary"]
# Kurzschlüssel und Namen der Bewerter in den Uneinigkeits-Statistiken
GRADER_KEY = {"main": "glm", "sensitivity": "mis", "tertiary": "oss", "ensemble": "med"}
GRADER_NAME = {"main": "GLM", "sensitivity": "Mistral", "tertiary": "GPT-OSS",
               "ensemble": "Median"}
ENSEMBLE = ("ensemble", "v6_median")
COLORS = {"main": "#4C72B0", "other": "#55A868", "ensemble": "#C44E52",
          "extra": "#8172B2"}

# Systeme (model_label, prompt_version, Anzeigename) der Vergleichstabellen
MAIN_SYSTEMS = [("main", v, f"{lab} ({MODEL_NAME['main']})")
                for v, lab in VERSION_LABEL.items()]
V5_SYSTEMS = [(m, "v5_rubrik", f"{MODEL_NAME[m]} (V5)") for m in GRADERS]
ENSEMBLE_SYSTEM = (*ENSEMBLE, "Ensemble (V6, Median)")


# --------------------------------------------------------------------------- #
# Laden und Aggregieren
# --------------------------------------------------------------------------- #
def load_raw(raw_arg):
    """Rohdaten (eine JSONL-Datei oder alle ``*.jsonl`` eines Verzeichnisses)
    als DataFrame; im Verzeichnismodus werden Mock-Testläufe (``_mock_`` im
    Dateinamen) ausgeblendet, damit sie die Auswertung nicht verfälschen."""
    path = resolve_path(raw_arg)
    files = (sorted(p for p in path.glob("*.jsonl") if "_mock_" not in p.name)
             if path.is_dir() else [path])
    rows = [r for fp in files for r in iter_jsonl(fp)]
    if not rows:
        raise FileNotFoundError(f"Keine Rohdaten unter {path} gefunden.")
    return pd.DataFrame(rows)


def load_runs(raw_arg):
    """Alle unabhängigen Läufe unter ``raw_arg`` als ``[(name, DataFrame)]``:
    Unterordner ``run*/`` zählen je als Lauf; ohne solche Unterordner ist das
    Verzeichnis (bzw. die Datei) selbst der einzige Lauf."""
    path = resolve_path(raw_arg)
    runs = sorted(p for p in path.glob("run*") if p.is_dir()) if path.is_dir() else []
    if not runs:
        return [(path.name, load_raw(path))]
    return [(p.name, load_raw(p)) for p in runs]


def mean_over_runs(frames, keys):
    """Zeilenweise Mittelwerte (und Streuung) numerischer Spalten über die
    Läufe; ``keys`` identifizieren die Zeile, ihre Reihenfolge bleibt die des
    ersten Laufs. Liefert ``(mean, std)`` mit ``std``-Spalten als ``<name>_std``."""
    if len(frames) == 1:
        return frames[0].reset_index(drop=True), None
    allf = pd.concat(frames, ignore_index=True)
    num = [c for c in allf.columns if c not in keys and pd.api.types.is_numeric_dtype(allf[c])]
    other = [c for c in allf.columns if c not in keys and c not in num]
    g = allf.groupby(keys, sort=False)
    mean = g[num].mean()
    std = g[num].std(ddof=0).add_suffix("_std")
    first = g[other].first() if other else None
    out = mean.join(std)
    if first is not None:
        out = out.join(first)
    order = frames[0][keys].drop_duplicates()
    out = order.merge(out.reset_index(), on=keys, how="left")
    out["n_runs"] = len(frames)
    return out, std.reset_index()


PE_KEYS = ["model_label", "prompt_version", "question_id", "answer_id"]
PE_META = ["topic", "qtype", "max_points", "level", "variant", "operation",
           "gt_points", "answer_chars"]


def point_estimates(df):
    """Pro (Modell, Version, Antwort) einen Punkt-Schätzwert (Median der gültigen
    Wiederholungen, ``pe``) und die Streuung über die Wiederholungen
    (``run_std``) berechnen."""
    ok = df[df["parse_ok"].astype(bool)]
    est = ok.groupby(PE_KEYS)["pred_points"].agg(
        pe="median", n_ok="size",
        run_std=lambda s: float(np.std(s.astype(float))) if len(s) > 1 else 0.0)
    meta = df.groupby(PE_KEYS).agg(n_samples=("parse_ok", "size"),
                                   **{m: (m, "first") for m in PE_META})
    pe = meta.join(est).reset_index()
    pe["n_ok"] = pe["n_ok"].fillna(0).astype(int)
    pe["run_std"] = pe["run_std"].fillna(0.0)
    return pe


def exact_rate(v):
    """Anteil der Antworten, deren gerundeter Schätzwert den Referenzwert trifft."""
    v = v.dropna(subset=["pe"])
    return float((v["pe"].round() == v["gt_points"]).mean()) if len(v) else float("nan")


def _safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics_for(pe):
    """Kennzahlen für eine Teilmenge (eine Modell/Version-Kombination).

    Para-σ und Robustheit beziehen sich auf Formulierungsvarianten derselben
    Frage mit GLEICHER Referenzpunktzahl (nur dort ist Streuung ein Fehler);
    Gruppen mit einer einzigen Antwort tragen nichts bei. Datensätze ohne
    Varianten erhalten ``None``.
    """
    v = pe.dropna(subset=["pe"])
    if v.empty:
        return None
    err = v["pe"] - v["gt_points"]
    abs_err = err.abs()
    grp = v.groupby(["question_id", "gt_points"])["pe"]
    para = grp.std(ddof=0)[grp.size() > 1]
    base = v[v["operation"] == "base"].groupby(["question_id", "gt_points"])["pe"].first()
    var = v[v["operation"] != "base"].set_index(["question_id", "gt_points"])["pe"]
    robust = np.abs(var.to_numpy() - base.reindex(var.index).to_numpy())
    robust = robust[~np.isnan(robust)]
    return {
        "n": int(len(v)),
        "parse_rate": float(v["n_ok"].sum() / v["n_samples"].sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "exact": exact_rate(v),
        "within1": float((abs_err <= 1.0).mean()),
        "run_std": float(v["run_std"].mean()),
        "para_std": float(para.mean()) if len(para) else None,
        "robustness": float(robust.mean()) if len(robust) else None,
        "len_bias": _safe_corr(v["answer_chars"], err),
    }


def metrics_table(pe):
    rows = []
    for (model, version), g in pe.groupby(["model_label", "prompt_version"]):
        m = metrics_for(g)
        if m:
            m.update({"model_label": model, "prompt_version": version})
            rows.append(m)
    return pd.DataFrame(rows)


def system_rows(pe, systems, extra=None):
    """Kennzahlen je System ``(model_label, prompt_version, label)`` inklusive
    exakter Quote je Punktstufe; ``extra(sub)`` liefert zusätzliche Spalten."""
    rows = []
    for model, version, label in systems:
        sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
        if sub.empty:
            continue
        m = metrics_for(sub)
        m.update({"label": label, "model_label": model, "prompt_version": version})
        for lvl in LEVELS:
            m[f"exact_{lvl}"] = exact_rate(sub[sub["level"] == lvl])
        if extra:
            m.update(extra(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def by_answer(pe, model, version):
    sub = pe[(pe["model_label"] == model) & (pe["prompt_version"] == version)]
    return sub.set_index(["question_id", "answer_id"])


SYSTEM_KEYS = ["model_label", "prompt_version", "label"]


def disagreement_stats(pe):
    """Antworten, über die die drei V5-Bewerter uneinig sind, und die exakte
    Quote jedes Systems auf dieser Teilmenge. ``None``, wenn kein Ensemble-Lauf
    vorliegt."""
    med = by_answer(pe, *ENSEMBLE)
    if med.empty:
        return None
    cols = {GRADER_KEY[g]: by_answer(pe, g, "v5_rubrik")["pe"] for g in GRADERS}
    comp = pd.DataFrame({**cols, "med": med["pe"], "gt": med["gt_points"]}).dropna()
    dis = comp[comp[[GRADER_KEY[g] for g in GRADERS]].round().nunique(axis=1) > 1]
    stats = {"n": int(len(dis)), "total": int(len(comp))}
    for g in GRADERS + ["ensemble"]:
        if len(dis):
            key = GRADER_KEY[g]
            stats[f"exact_{key}"] = float((dis[key].round() == dis["gt"]).mean())
    return stats


def disagreement_over_runs(pes, print_prefix="  "):
    """``disagreement_stats`` je Lauf, über die Läufe gemittelt und ausgegeben."""
    per_run = [s for s in (disagreement_stats(pe) for pe in pes) if s]
    if not per_run:
        return None
    keys = sorted({k for s in per_run for k in s})
    stats = {k: float(np.mean([s[k] for s in per_run if k in s])) for k in keys}
    stats["n_runs"] = len(per_run)
    print(f"\nUneinige Antworten: {stats['n']:.1f}/{stats['total']:.0f} "
          f"({100*stats['n']/stats['total']:.1f}%, Mittel über {len(per_run)} Lauf/Läufe)")
    for g in GRADERS + ["ensemble"]:
        k = f"exact_{GRADER_KEY[g]}"
        if k in stats:
            print(f"{print_prefix}exakt auf Uneinigkeits-Subset [{GRADER_NAME[g]:8}]: {100*stats[k]:.1f}%")
    return stats


# --------------------------------------------------------------------------- #
# Punktstufen / Konfusionsmatrix
# --------------------------------------------------------------------------- #
def confusion(pe_one):
    """3x3-Konfusionsmatrix über Punktstufen: jeder Schätzwert wird der
    nächstgelegenen Punktstufe seiner Frage zugeordnet (skaleninvariant)."""
    level_pts = pe_one.groupby(["question_id", "level"])["gt_points"].first()
    mat = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for _, r in pe_one.dropna(subset=["pe"]).iterrows():
        lm = level_pts[r["question_id"]]
        pred = min(lm.index, key=lambda l: abs(r["pe"] - lm[l]))
        mat[LEVELS.index(r["level"]), LEVELS.index(pred)] += 1
    return mat


# --------------------------------------------------------------------------- #
# Ausgabe: Tabellen und Abbildungen
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def metric_cells(r):
    """Standard-Spalten MAE, Exakt, ±1, Lauf-σ einer Kennzahlen-Zeile."""
    return [_fmt(r["mae"]), f"{_fmt(100*r['exact'], 1)}\\%",
            f"{_fmt(100*r['within1'], 1)}\\%", _fmt(r["run_std"])]


def write_tabular(out, colspec, header, rows, midrule_after=(), bold=()):
    """LaTeX-``tabular`` (booktabs) schreiben. ``rows`` sind ``(label, cells)``;
    nach den Labels in ``midrule_after`` folgt ein ``\\midrule``, Labels in
    ``bold`` werden fett gesetzt."""
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\",
             r"\midrule"]
    for label, cells in rows:
        shown = rf"\textbf{{{label}}}" if label in bold else label
        lines.append(" & ".join([shown, *cells]) + r" \\")
        if label in midrule_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabelle geschrieben: {out}")


def write_summary(out, summary):
    """Kennzahlen als JSON; NaN wird zu ``null`` (gültiges JSON)."""
    def clean(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    out = resolve_path(out)
    out.write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Summary geschrieben: {out}")


def save_figure(fig, out):
    """Abbildung als PDF (fürs Paper) und PNG (zur Kontrolle) speichern."""
    import matplotlib.pyplot as plt
    out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def new_figure(figsize):
    """matplotlib erst hier laden: Skripte, die nur Tabellen schreiben,
    brauchen es nicht."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)


def write_metrics_models(mt, out):
    """Beide Modelle untereinander, je Prompt-Version eine Zeile: Korrektheit,
    Konsistenz (Lauf-/Para-σ) und Robustheit in einer Tabelle."""
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Version & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ & Para-$\sigma$ & Rob. \\",
             r"\midrule"]
    for i, m in enumerate(k for k in ("main", "sensitivity") if k in set(mt["model_label"])):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\emph{" + MODEL_NAME[m] + r"}} \\")
        for v, label in VERSION_LABEL.items():
            sub = mt[(mt["model_label"] == m) & (mt["prompt_version"] == v)]
            if sub.empty:
                continue
            r0 = sub.iloc[0]
            cells = metric_cells(r0) + [_fmt(r0["para_std"], 3), _fmt(r0["robustness"], 3)]
            lines.append(" & ".join([label, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabelle geschrieben: {out}")


def write_dataset_stats(pe, out):
    base = pe[pe["model_label"] == pe["model_label"].iloc[0]]
    n_ans = base["answer_id"].nunique()
    n_var = base[base["operation"] != "base"]["answer_id"].nunique()
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Kennzahl & Wert \\", r"\midrule",
             f"Fragen & {base['question_id'].nunique()} \\\\",
             f"Antworten gesamt & {n_ans} \\\\",
             f"davon Basis-Antworten & {n_ans - n_var} \\\\",
             f"davon Varianten & {n_var} \\\\",
             f"Punktstufen je Frage & {base['level'].nunique()} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    resolve_path(out).write_text("\n".join(lines), encoding="utf-8")


def plot_confusion(mat, title, out):
    fig, ax = new_figure((3.6, 3.2))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_yticks(range(len(LEVELS)))
    ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS])
    ax.set_yticklabels([LEVEL_LABEL[l] for l in LEVELS])
    ax.set_xlabel("vorhergesagte Stufe")
    ax.set_ylabel("wahre Stufe")
    ax.set_title(title)
    for i in range(len(LEVELS)):
        for j in range(len(LEVELS)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save_figure(fig, out)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung des Benchmark-Laufs.")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)

    runs = load_runs(args.raw)
    pes = [point_estimates(df) for _, df in runs]
    mt, mt_std = mean_over_runs([metrics_table(pe) for pe in pes],
                                ["model_label", "prompt_version"])
    res = resolve_path(args.results_dir)
    (res / "tables").mkdir(parents=True, exist_ok=True)

    write_metrics_models(mt, res / "tables" / "metrics_models.tex")
    write_dataset_stats(pes[0], res / "tables" / "dataset_stats.tex")
    # Konfusionsmatrix: Hauptmodell über ALLE Versionen und Läufe gepoolt --
    # Gesamtbild der Verwechslungen statt einer einzelnen Version.
    mat = sum(confusion(pe[pe["model_label"] == "main"]) for pe in pes)
    plot_confusion(mat, "Konfusionsmatrix", res / "figures" / "confusion_best.pdf")

    main_mt = mt[mt["model_label"] == "main"]
    df_all = pd.concat([df for _, df in runs], ignore_index=True)
    write_summary(res / "summary.json", {
        "best_version": main_mt.sort_values("mae").iloc[0]["prompt_version"],
        "n_runs": len(runs),
        "runs": [name for name, _ in runs],
        "metrics": mt.to_dict(orient="records"),
        "n_records": int(len(df_all)),
        "models": sorted(df_all["model_label"].unique().tolist()),
        "versions": sorted(df_all["prompt_version"].unique().tolist()),
    })
    cols = ["model_label", "prompt_version", "mae", "exact", "within1", "run_std",
            "para_std", "robustness"] + (["exact_std", "mae_std"] if mt_std is not None else [])
    print(f"Läufe: {[n for n, _ in runs]}")
    print(mt[cols].to_string(index=False))


if __name__ == "__main__":
    main()
