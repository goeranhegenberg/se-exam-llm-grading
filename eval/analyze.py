"""Wertet die JSONL-Rohdaten der Bewertungsläufe aus.

Erzeugt:
  * results/tables/metrics_main.tex        -- Metriken je Prompt-Version (Hauptmodell)
  * results/tables/metrics_sensitivity.tex -- Modellvergleich (Haupt- vs. Zweitmodell)
  * results/tables/metrics_models.tex      -- beide Modelle, alle Kennzahlen inkl. Robustheit
  * results/tables/dataset_stats.tex       -- Kennzahlen des Datensatzes
  * results/figures/mae_by_version.pdf     -- MAE je Prompt-Version
  * results/figures/confusion_best.pdf     -- Konfusionsmatrix (Punktstufen) der besten Version
  * results/figures/boxplot_levels.pdf     -- Streuung der vergebenen Punkte je Punktstufe
  * results/summary.json                   -- alle Kennzahlen maschinenlesbar

Aufruf:
    python -m eval.analyze                 # liest results/raw/*.jsonl
    python -m eval.analyze --raw <pfad>    # einzelne Datei oder Verzeichnis
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import resolve_path

LEVELS = ["null", "teil", "voll"]
LEVEL_LABEL = {"null": "0 P.", "teil": "Teil", "voll": "voll"}


# --------------------------------------------------------------------------- #
# Laden und Aggregieren
# --------------------------------------------------------------------------- #
def load_raw(raw_arg):
    path = resolve_path(raw_arg)
    # Im Verzeichnismodus Mock-Testläufe ausblenden, damit sie die echte
    # Auswertung nicht verfälschen (eine konkret angegebene Datei wird genutzt).
    files = (sorted(p for p in path.glob("*.jsonl") if "_mock_" not in p.name)
             if path.is_dir() else [path])
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"Keine Rohdaten unter {path} gefunden.")
    return pd.DataFrame(rows)


def point_estimates(df):
    """Pro (Modell, Version, Antwort) einen Punkt-Schätzwert (Median der gültigen
    Samples) und die Streuung über die Wiederholungen berechnen."""
    keys = ["model_label", "prompt_version", "question_id", "answer_id"]
    meta = ["topic", "qtype", "max_points", "level", "variant", "operation",
            "gt_points", "answer_chars"]
    recs = []
    for key_vals, g in df.groupby(keys):
        ok = g[g["parse_ok"] == True]  # noqa: E712
        preds = ok["pred_points"].astype(float).tolist()
        rec = dict(zip(keys, key_vals))
        first = g.iloc[0]
        for m in meta:
            rec[m] = first.get(m)
        rec["n_samples"] = len(g)
        rec["n_ok"] = len(preds)
        rec["pe"] = float(np.median(preds)) if preds else np.nan
        rec["run_std"] = float(np.std(preds)) if len(preds) > 1 else 0.0
        recs.append(rec)
    return pd.DataFrame(recs)


def _safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics_for(pe):
    """Kennzahlen für eine Teilmenge (eine Modell/Version-Kombination)."""
    v = pe.dropna(subset=["pe"])
    if v.empty:
        return None
    err = v["pe"] - v["gt_points"]
    abs_err = err.abs()
    # Paraphrasen-Streuung: Streuung der Schätzwerte über Varianten derselben
    # Frage mit GLEICHER Referenzpunktzahl. Nicht nach Punktstufe gruppieren:
    # Die Teil-Stufe der erweiterten Fragen enthält bewusst Antworten mit
    # 1, 2 und 3 Punkten, dort hätte selbst ein perfekter Bewerter Streuung.
    para = v.groupby(["question_id", "gt_points"])["pe"].std(ddof=0).dropna()
    # Robustheit: Abweichung der punkterhaltenden Varianten von ihrer
    # Basis-Antwort (gleiche Frage, gleiche Referenzpunktzahl)
    robust = []
    for _, grp in v.groupby(["question_id", "gt_points"]):
        base = grp[grp["operation"] == "base"]["pe"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in grp[grp["operation"] != "base"].iterrows():
            robust.append(abs(r["pe"] - b))
    return {
        "n": int(len(v)),
        "parse_rate": float(v["n_ok"].sum() / v["n_samples"].sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "exact": float((v["pe"].round() == v["gt_points"]).mean()),
        "within1": float((abs_err <= 1.0).mean()),
        "run_std": float(v["run_std"].mean()),
        "para_std": float(para.mean()) if len(para) else 0.0,
        "robustness": float(np.mean(robust)) if robust else 0.0,
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


# --------------------------------------------------------------------------- #
# Punktstufen / Konfusionsmatrix
# --------------------------------------------------------------------------- #
def predicted_level(pe_value, level_map):
    """Schätzwert der nächstgelegenen Punktstufe zuordnen (skaleninvariant)."""
    best, bestd = None, None
    for lvl, val in level_map.items():
        d = abs(pe_value - val)
        if bestd is None or d < bestd:
            best, bestd = lvl, d
    return best


def confusion(pe_one):
    """3x3-Konfusionsmatrix über Punktstufen für eine Modell/Version-Teilmenge."""
    level_maps = {}
    for qid, grp in pe_one.groupby("question_id"):
        level_maps[qid] = {r["level"]: r["gt_points"] for _, r in grp.iterrows()}
    mat = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for _, r in pe_one.dropna(subset=["pe"]).iterrows():
        pl = predicted_level(r["pe"], level_maps[r["question_id"]])
        mat[LEVELS.index(r["level"]), LEVELS.index(pl)] += 1
    return mat


# --------------------------------------------------------------------------- #
# Ausgabe: Tabellen und Abbildungen
# --------------------------------------------------------------------------- #
VERSION_LABEL = {
    "v1_baseline": "V1 Baseline",
    "v2_begruendung": "V2 Begr.",
    "v3_thinking": "V3 Thinking",
    "v4_fewshot": "V4 Few-Shot",
    "v5_rubrik": "V5 Rubrik",
}


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def write_metrics_main(mt, out):
    main = mt[mt["model_label"] == "main"].copy()
    if main.empty:
        main = mt.copy()
    main["order"] = main["prompt_version"].map(
        {v: i for i, v in enumerate(VERSION_LABEL)}).fillna(99)
    main = main.sort_values("order")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Prompt-Version & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ & Para-$\sigma$ \\",
             r"\midrule"]
    for _, r in main.iterrows():
        lines.append(
            f"{VERSION_LABEL.get(r['prompt_version'], r['prompt_version'])} & "
            f"{_fmt(r['mae'])} & {_fmt(100*r['exact'],1)}\\% & "
            f"{_fmt(100*r['within1'],1)}\\% & {_fmt(r['run_std'])} & "
            f"{_fmt(r['para_std'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")


_MODEL_DISPLAY = {
    "z-ai/glm-5.2": "GLM 5.2",
    "mistralai/mistral-small-2603": "Mistral Small 4",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
}


def _short_model(name):
    if not name:
        return "?"
    if name in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[name]
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", str(name))
    return s.split("/")[-1]


def write_metrics_sensitivity(mt, label_map, out):
    """Pivot: je Prompt-Version eine Zeile, je Modell MAE + exakte Quote."""
    models = [m for m in ["main", "sensitivity"] if m in set(mt["model_label"])]
    if not models:
        models = sorted(mt["model_label"].unique())[:2]
    versions = [v for v in VERSION_LABEL if v in set(mt["prompt_version"])]
    colspec = "l" + "rr" * len(models)
    header = "Version"
    for m in models:
        sm = _short_model(label_map.get(m, m))
        header += (f" & \\makecell{{MAE\\\\({sm})}}"
                   f" & \\makecell{{Exakt\\\\({sm})}}")
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\",
             r"\midrule"]
    for v in versions:
        row = VERSION_LABEL.get(v, v)
        for m in models:
            sub = mt[(mt["model_label"] == m) & (mt["prompt_version"] == v)]
            if sub.empty:
                row += " & -- & --"
            else:
                r0 = sub.iloc[0]
                row += f" & {_fmt(r0['mae'])} & {_fmt(100*r0['exact'],1)}\\%"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_models(mt, label_map, out):
    """Beide Modelle untereinander, je Prompt-Version eine Zeile: Korrektheit,
    Konsistenz (Lauf-/Para-sigma) und Robustheit in einer Tabelle."""
    models = [m for m in ["main", "sensitivity"] if m in set(mt["model_label"])]
    versions = [v for v in VERSION_LABEL if v in set(mt["prompt_version"])]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Version & MAE & Exakt & $\pm$1 & Lauf-$\sigma$ & Para-$\sigma$ & Rob. \\",
             r"\midrule"]
    for i, m in enumerate(models):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\emph{" + _short_model(label_map.get(m, m))
                     + r"}} \\")
        for v in versions:
            sub = mt[(mt["model_label"] == m) & (mt["prompt_version"] == v)]
            if sub.empty:
                continue
            r0 = sub.iloc[0]
            lines.append(
                f"{VERSION_LABEL.get(v, v)} & {_fmt(r0['mae'])} & "
                f"{_fmt(100*r0['exact'],1)}\\% & {_fmt(100*r0['within1'],1)}\\% & "
                f"{_fmt(r0['run_std'])} & {_fmt(r0['para_std'], 3)} & "
                f"{_fmt(r0['robustness'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_stats(pe, out):
    base = pe[pe["model_label"] == pe["model_label"].iloc[0]]
    n_q = base["question_id"].nunique()
    n_ans = base["answer_id"].nunique()
    n_var = base[base["operation"] != "base"]["answer_id"].nunique()
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Kennzahl & Wert \\", r"\midrule",
             f"Fragen & {n_q} \\\\",
             f"Antworten gesamt & {n_ans} \\\\",
             f"davon Basis-Antworten & {n_ans - n_var} \\\\",
             f"davon Varianten & {n_var} \\\\",
             f"Punktstufen je Frage & {base['level'].nunique()} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def plot_mae(mt, out):
    main = mt[mt["model_label"] == "main"]
    if main.empty:
        main = mt
    order = [v for v in VERSION_LABEL if v in set(main["prompt_version"])]
    vals = [main[main["prompt_version"] == v]["mae"].iloc[0] for v in order]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar([VERSION_LABEL[v] for v in order], vals, color="#4C72B0")
    ax.set_ylabel("MAE (Punkte)")
    ax.set_title("Mittlerer absoluter Fehler je Prompt-Version")
    for i, val in enumerate(vals):
        ax.text(i, val, _fmt(val), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_confusion(mat, title, out):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
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
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_para_by_version(mt, out):
    """Paraphrasen-Streuung (Para-sigma) je Prompt-Version fuer das Hauptmodell.

    Informativere Alternative zur MAE-Grafik: zeigt, wie die
    Formulierungsabhaengigkeit der Bewertung mit zunehmender Prompt-Struktur
    abnimmt (kleiner ist besser)."""
    main = mt[mt["model_label"] == "main"]
    if main.empty:
        main = mt
    order = [v for v in VERSION_LABEL if v in set(main["prompt_version"])]
    vals = [main[main["prompt_version"] == v]["para_std"].iloc[0] for v in order]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar([VERSION_LABEL[v] for v in order], vals, color="#55A868")
    ax.set_ylabel(r"Para-$\sigma$ (Punkte)")
    ax.set_title("Paraphrasen-Streuung je Prompt-Version")
    for i, val in enumerate(vals):
        ax.text(i, val, _fmt(val), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_boxplot_levels(pe_one, title, out):
    data = [pe_one[pe_one["level"] == l]["pe"].dropna().tolist() for l in LEVELS]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.boxplot(data, tick_labels=[LEVEL_LABEL[l] for l in LEVELS])
    ax.set_xlabel("wahre Punktstufe")
    ax.set_ylabel("vergebene Punkte")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Auswertung der Bewertungsläufe.")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)

    df = load_raw(args.raw)
    pe = point_estimates(df)
    mt = metrics_table(pe)

    res = resolve_path(args.results_dir)
    (res / "tables").mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    label_map = df.groupby("model_label")["model_requested"].first().to_dict()
    write_metrics_main(mt, res / "tables" / "metrics_main.tex")
    write_metrics_sensitivity(mt, label_map, res / "tables" / "metrics_sensitivity.tex")
    write_metrics_models(mt, label_map, res / "tables" / "metrics_models.tex")
    write_dataset_stats(pe, res / "tables" / "dataset_stats.tex")

    # Beste Version (Hauptmodell) = niedrigster MAE -- für summary.json/Text.
    main_mt = mt[mt["model_label"] == "main"]
    if main_mt.empty:
        main_mt = mt
    best_version = main_mt.sort_values("mae").iloc[0]["prompt_version"]

    # Für Konfusionsmatrix/Boxplot: Hauptmodell über ALLE Versionen gepoolt --
    # repräsentatives Gesamtbild (inkl. der wenigen Fehler) statt einer einzelnen,
    # womöglich perfekten Version.
    main_label = "main" if "main" in set(pe["model_label"]) else pe["model_label"].iloc[0]
    pe_main = pe[pe["model_label"] == main_label]

    plot_mae(mt, res / "figures" / "mae_by_version.pdf")
    plot_para_by_version(mt, res / "figures" / "para_by_version.pdf")
    plot_confusion(confusion(pe_main), "Konfusionsmatrix",
                   res / "figures" / "confusion_best.pdf")
    plot_boxplot_levels(pe_main, "Punktvergabe je Stufe",
                        res / "figures" / "boxplot_levels.pdf")

    summary = {
        "best_version": best_version,
        "metrics": mt.to_dict(orient="records"),
        "n_records": int(len(df)),
        "models": sorted(df["model_label"].unique().tolist()),
        "versions": sorted(df["prompt_version"].unique().tolist()),
    }
    (res / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Auswertung geschrieben nach", res)
    print(mt.to_string(index=False))
    print("Beste Version (MAE):", best_version)
    return summary


if __name__ == "__main__":
    main()
