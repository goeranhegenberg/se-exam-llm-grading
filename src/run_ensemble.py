"""V6: Ensemble-Bewertung aus drei Modellen (Median).

Drei quelloffene Reasoning-Modelle unterschiedlicher Anbieter (GLM 5.2,
Mistral Small 4, GPT-OSS 120B) bewerten jede Antwort mit dem V5-Prompt
(Rubrik + Few-Shot + Begruendung + Thinking). Die finale V6-Punktzahl ist der
Median der drei Bewertungen (``v6_median``). Zum Vergleich wird zusaetzlich ein
LLM-gestuetztes Merge berechnet, bei dem GLM 5.2 die drei begruendeten Bewertungen
zusammenfuehrt (``v6_ensemble``); es bringt keinen Mehrwert ueber den Median und
wird im Paper nicht verwendet.

Im Merge-Prompt erhaelt das aggregierende Modell Rubrik, Frage, Antwort und die
drei Vorbewertungen; die Bewerter werden anonymisiert (A/B/C) und ihre
Reihenfolge wird je Antwort deterministisch permutiert, um Positions- und
Markeneffekte zu daempfen.

Voraussetzung: die V5-Rohdaten der drei Modelle (model_label main/sensitivity/
tertiary) liegen in ``results/raw``. Aufruf:
    python -m src.run_ensemble                # echter Lauf (Merge ueber OpenRouter)
    python -m src.run_ensemble --mock --limit 9
"""
import argparse
import hashlib
import statistics
import sys
from pathlib import Path

from . import config as cfg
from .dataset import answer_meta, iter_answer_items, load_questions
from .llm_client import make_client
from .prompt_builder import _fmt, fill_template, load_system
from .runner import run_and_write
from .util import extract_json, extract_points, iter_jsonl, parse_pred, salvage_points

MERGE_ATTEMPTS = 4  # GLM liefert bei temp>0 vereinzelt degenerierte Stub-Antworten
                    # (z.B. "{}"); dann mit variiertem Seed erneut versuchen.

GRADERS = ["main", "sensitivity", "tertiary"]  # Bewerter (model_labels in config)
MERGER = "main"  # GLM 5.2 fuehrt zusammen
_PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]


def load_v5(raw_dir):
    """V5-Records der Bewerter je (model_label, qid, aid, sample) einsammeln."""
    raw = cfg.resolve_path(raw_dir)
    recs = {}
    for fp in sorted(raw.glob("grading_*.jsonl")):
        if "_mock_" in fp.name:
            continue
        for r in iter_jsonl(fp):
            if r.get("prompt_version") == "v5_rubrik" and r.get("model_label") in GRADERS:
                recs[(r["model_label"], r["question_id"], r["answer_id"],
                      r["sample_index"])] = r
    return recs


def parse_grade(rec):
    """(punkte, begruendung) aus einem V5-Record holen."""
    raw = rec.get("raw", "")
    obj, ok = extract_json(raw)
    pts = extract_points(obj) if ok else salvage_points(raw)
    if pts is None:
        pts = rec.get("pred_points")
    begr = str(obj.get("begruendung", "")).strip() if ok and isinstance(obj, dict) else ""
    return pts, begr


def perm_for(answer_id, sample_index):
    """Deterministische Permutation der drei Bewerter (de-biased gegen Position)."""
    h = int(hashlib.sha1(f"{answer_id}|{sample_index}".encode("utf-8")).hexdigest(), 16)
    return _PERMS[h % len(_PERMS)]


def build_merge_prompt(template, q, answer_text, ordered_grades):
    """ordered_grades: Liste [(punkte, begruendung), ...] in Anzeigereihenfolge."""
    lines = [f"[Bewerter {'ABC'[i]}] Punkte: {_fmt(p) if p is not None else '?'} "
             f"-- Begruendung: {b}" for i, (p, b) in enumerate(ordered_grades)]
    return fill_template(template, q, answer_text, vorbewertungen="\n".join(lines))


def median_round(values):
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else None


def load_cells(questions, v5, reps, want=None):
    """Vollstaendige (q, a, sample, [recs])-Zellen, fuer die alle drei Bewerter
    ein V5-Ergebnis haben; ``want`` schraenkt auf answer_ids ein."""
    cells, missing = [], 0
    for q, a in iter_answer_items(questions):
        if want is not None and a["answer_id"] not in want:
            continue
        for s in range(reps):
            recs = [v5.get((g, q["question_id"], a["answer_id"], s)) for g in GRADERS]
            if all(r is not None for r in recs):
                cells.append((q, a, s, recs))
            else:
                missing += 1
    return cells, missing


def one_merge(client, model, system, user, sampling, q, a, seed):
    """Einen Merge-Aufruf mit Retry bei nicht parsbarer/degenerierter Antwort.
    Liefert ``(punkte, res, fehler, versuche)``; ``res`` ist die letzte Antwort."""
    last, err = None, None
    for attempt in range(MERGE_ATTEMPTS):
        sd = (seed + attempt) if seed is not None else None
        try:
            last = client.grade(
                model=model, system=system, user=user,
                temperature=sampling["temperature"], top_p=sampling["top_p"],
                max_tokens=sampling["max_tokens"], seed=sd, json_mode=True,
                thinking=True, ground_truth=a["points"], max_points=q["max_points"])
            pred, _ = parse_pred(last["text"])
            if pred is not None:
                return pred, last, None, attempt + 1
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return None, last, err, MERGE_ATTEMPTS


def main(argv=None):
    p = argparse.ArgumentParser(description="V6-Ensemble-Bewertung mit Meta-Merge.")
    p.add_argument("--config", default=None)
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Nur die ersten N (qid,aid,sample)-Zellen mergen (Test).")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--median-only", action="store_true",
                   help="Nur den Median (v6_median) bilden, kein LLM-Merge, kein API-Aufruf.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    conf = cfg.load_config(args.config)
    questions = load_questions(args.dataset_dir or conf["paths"]["dataset_dir"])
    reps = conf["run"]["repetitions"]
    cells, missing = load_cells(questions, load_v5(args.raw), reps)
    if missing:
        print(f"WARNUNG: {missing} Zellen ohne vollstaendige V5-Daten -- uebersprungen.")
    if args.limit is not None:
        cells = cells[: args.limit]
    if not cells:
        sys.exit("FEHLER: keine vollstaendigen V5-Zellen gefunden (erst V5 fuer "
                 "main/sensitivity/tertiary rechnen).")

    merger_model = conf["models"][MERGER]
    sampling = cfg.sampling_for(conf, MERGER, thinking=True)
    if not args.median_only:
        system = load_system(args.prompts_dir)
        template = (cfg.resolve_path(args.prompts_dir) / "v6_merge.txt").read_text(
            encoding="utf-8")
        client = make_client(conf, mock=args.mock, noise=0)
    concurrency = (1 if (args.mock or args.median_only)
                   else (args.concurrency or conf["run"].get("concurrency", 8)))

    run_id = cfg.utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "raw" /
        f"grading_ensemble_{'mock' if args.mock else 'v6'}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ensemble {run_id}] Bewerter={GRADERS} "
          f"Merger={'nur Median' if args.median_only else merger_model} "
          f"Zellen={len(cells)} Concurrency={concurrency}")
    print(f"  Ausgabe: {out_path}")

    def work(cell):
        q, a, s, recs = cell
        grades = [parse_grade(r) for r in recs]  # in GRADERS-Reihenfolge
        comp = {g: grades[i][0] for i, g in enumerate(GRADERS)}
        base = {**answer_meta(q, a), "sample_index": s, "n_samples": reps,
                "component_points": comp}
        # Median-Baseline (ohne API) -- die im Paper verwendete V6-Punktzahl.
        med = median_round([comp[g] for g in GRADERS])
        mrec = {**base, "model_label": "ensemble", "model_requested": "median",
                "prompt_version": "v6_median", "thinking": False,
                "temperature": None, "top_p": None,
                "pred_points": med, "parse_ok": med is not None, "usage": {}}
        if args.median_only:
            return [mrec]
        # LLM-Merge (GLM) -- mit Retry bei nicht parsbarer Antwort.
        order = perm_for(a["answer_id"], s)
        base["component_order"] = [GRADERS[i] for i in order]
        user = build_merge_prompt(template, q, a["text"], [grades[i] for i in order])
        pred, res, err, attempts = one_merge(client, merger_model, system, user,
                                             sampling, q, a, sampling["seed"])
        rec = {**base, "model_label": "ensemble", "model_requested": merger_model,
               "prompt_version": "v6_ensemble", "thinking": True,
               "temperature": sampling["temperature"], "top_p": sampling["top_p"],
               "pred_points": pred, "parse_ok": pred is not None,
               "merge_attempts": attempts}
        if res is not None:
            rec.update({"model_returned": res.get("model"),
                        "finish_reason": res.get("finish_reason"),
                        "usage": res.get("usage", {}), "raw": res["text"]})
        else:
            rec["error"] = err
        return [rec, mrec]

    counted = "v6_median" if args.median_only else "v6_ensemble"
    run_and_write(work, cells, concurrency, out_path, run_id,
                  count=lambda r: r["prompt_version"] == counted,
                  what="Zellen" if args.median_only else "Merges")
    return str(out_path)


if __name__ == "__main__":
    main()
