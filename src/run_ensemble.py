"""V6: Ensemble-Bewertung aus drei Modellen (Median).

Drei quelloffene Reasoning-Modelle unterschiedlicher Anbieter (GLM 5.2,
Mistral Small 4, GPT-OSS-120B) bewerten jede Antwort mit dem V5-Prompt
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
import datetime as dt
import hashlib
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg
from .dataset import iter_answer_items, load_questions
from .llm_client import GradingClient, MockClient
from .prompt_builder import _fmt, load_system, render_rubric
from .util import extract_json, extract_points, salvage_points

MERGE_ATTEMPTS = 4  # GLM liefert bei temp>0 vereinzelt degenerierte Stub-Antworten
                    # (z.B. "{}"); dann mit variiertem Seed erneut versuchen.

GRADERS = ["main", "sensitivity", "tertiary"]  # Bewerter (model_labels in config)
MERGER = "main"  # GLM 5.2 fuehrt zusammen
_PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]


def _utcstamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_v5(raw_dir):
    """V5-Records je (model_label, qid, aid, sample) einsammeln."""
    raw = cfg.resolve_path(raw_dir)
    recs = {}
    for fp in sorted(raw.glob("*.jsonl")):
        if "_mock_" in fp.name:
            continue
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("prompt_version") != "v5_rubrik":
                    continue
                ml = r.get("model_label")
                if ml not in GRADERS:
                    continue
                recs[(ml, r["question_id"], r["answer_id"], r["sample_index"])] = r
    return recs


def parse_grade(rec):
    """(punkte, begruendung) aus einem V5-Record holen."""
    obj, ok = extract_json(rec.get("raw", "")) if rec else (None, False)
    pts = extract_points(obj) if ok else None
    if pts is None and rec is not None:
        pts = rec.get("pred_points")
    begr = ""
    if ok and isinstance(obj, dict):
        begr = str(obj.get("begruendung", "")).strip()
    return pts, begr


def perm_for(answer_id, sample_index):
    """Deterministische Permutation der drei Bewerter (de-biased gegen Position)."""
    h = int(hashlib.sha1(f"{answer_id}|{sample_index}".encode("utf-8")).hexdigest(), 16)
    return _PERMS[h % len(_PERMS)]


def build_merge_prompt(template, q, answer_text, ordered_grades):
    """ordered_grades: Liste [(punkte, begruendung), ...] in Anzeigereihenfolge."""
    labels = "ABC"
    lines = []
    for i, (pts, begr) in enumerate(ordered_grades):
        p = _fmt(pts) if pts is not None else "?"
        lines.append(f"[Bewerter {labels[i]}] Punkte: {p} -- Begruendung: {begr}")
    return (template
            .replace("{{frage}}", q["question"].strip())
            .replace("{{rubrik}}", render_rubric(q["rubric"]))
            .replace("{{antwort}}", answer_text.strip())
            .replace("{{max_punkte}}", _fmt(q["max_points"]))
            .replace("{{vorbewertungen}}", "\n".join(lines)))


def median_round(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


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
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    conf = cfg.load_config(args.config)
    sampling = conf["sampling"]
    dataset_dir = args.dataset_dir or conf["paths"]["dataset_dir"]
    questions = load_questions(dataset_dir)
    qmap = {q["question_id"]: q for q in questions}
    amap = {a["answer_id"]: (q, a) for q, a in iter_answer_items(questions)}

    v5 = load_v5(args.raw)
    reps = conf["run"]["repetitions"]

    # Vollstaendige (qid,aid,sample)-Zellen bestimmen, fuer die alle drei Bewerter
    # ein V5-Ergebnis haben.
    cells = []
    missing = 0
    for q, a in iter_answer_items(questions):
        for s in range(reps):
            recs = [v5.get((g, q["question_id"], a["answer_id"], s)) for g in GRADERS]
            if all(r is not None for r in recs):
                cells.append((q, a, s, recs))
            else:
                missing += 1
    if missing:
        print(f"WARNUNG: {missing} Zellen ohne vollstaendige V5-Daten -- uebersprungen.")
    if args.limit is not None:
        cells = cells[: args.limit]
    if not cells:
        sys.exit("FEHLER: keine vollstaendigen V5-Zellen gefunden (erst V5 fuer "
                 "main/sensitivity/tertiary rechnen).")

    merger_model = conf["models"][MERGER]
    tt = sampling.get("temperature_thinking", {})
    tp = sampling.get("top_p_thinking", sampling["top_p"])
    merge_temp = tt.get(MERGER, tt.get("default", sampling["temperature"]))
    merge_top_p = (tp.get(MERGER, tp.get("default", sampling["top_p"]))
                   if isinstance(tp, dict) else tp)
    merge_max_tokens = sampling.get("max_tokens_thinking", sampling["max_tokens"])

    system = load_system(args.prompts_dir)
    template = (cfg.resolve_path(args.prompts_dir) / "v6_merge.txt").read_text(
        encoding="utf-8")

    if args.mock:
        client = MockClient(noise=0)
        concurrency = 1
    else:
        api_key = cfg.get_api_key()
        if not api_key:
            sys.exit("FEHLER: Kein API-Key (OPENROUTER_API_KEY) in .env.")
        provider = conf.get("provider", {})
        base_url = provider.get("base_url") or cfg.get_base_url()
        client = GradingClient(api_key, base_url=base_url)
        concurrency = args.concurrency or conf["run"].get("concurrency", 8)

    run_id = _utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "raw" /
        f"grading_ensemble_{'mock' if args.mock else 'v6'}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ensemble {run_id}] Bewerter={GRADERS} Merger={merger_model} "
          f"(temp={merge_temp}, top_p={merge_top_p}) Zellen={len(cells)} "
          f"Concurrency={concurrency}")
    print(f"  Ausgabe: {out_path}")

    def meta_from(rec, q, a):
        return {
            "question_id": q["question_id"], "topic": q["topic"],
            "qtype": q["type"], "max_points": q["max_points"],
            "answer_id": a["answer_id"], "level": a["level"],
            "variant": a["variant"], "operation": a["operation"],
            "gt_points": a["points"], "answer_chars": len(a["text"]),
            "answer_words": len(a["text"].split()),
        }

    def work(cell):
        q, a, s, recs = cell
        grades = [parse_grade(r) for r in recs]  # in GRADERS-Reihenfolge
        comp = {g: (grades[i][0]) for i, g in enumerate(GRADERS)}
        order = perm_for(a["answer_id"], s)
        ordered = [grades[i] for i in order]
        user = build_merge_prompt(template, q, a["text"], ordered)
        base = meta_from(recs[0], q, a)
        base.update({
            "sample_index": s, "n_samples": reps,
            "component_points": comp,
            "component_order": [GRADERS[i] for i in order],
        })
        out_recs = []
        # 1) LLM-Merge (GLM) -- mit Retry bei nicht parsbarer/degenerierter Antwort.
        rec = dict(base)
        rec.update({"model_label": "ensemble", "model_requested": merger_model,
                    "prompt_version": "v6_ensemble", "thinking": True,
                    "temperature": merge_temp, "top_p": merge_top_p})
        base_seed = sampling.get("seed")
        last_res, pred, err, attempts = None, None, None, 0
        for attempt in range(MERGE_ATTEMPTS):
            attempts = attempt + 1
            seed = (base_seed + attempt) if base_seed is not None else None
            try:
                res = client.grade(
                    model=merger_model, system=system, user=user,
                    temperature=merge_temp, top_p=merge_top_p,
                    max_tokens=merge_max_tokens, seed=seed,
                    json_mode=True, extra_body={"reasoning": {"enabled": True}},
                    ground_truth=a["points"], max_points=q["max_points"])
                last_res = res
                obj, ok = extract_json(res["text"])
                pred = extract_points(obj) if ok else None
                if pred is None:
                    pred = salvage_points(res["text"])
                if pred is not None:
                    break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        if last_res is not None:
            rec.update({"model_returned": last_res.get("model"), "pred_points": pred,
                        "parse_ok": pred is not None,
                        "finish_reason": last_res.get("finish_reason"),
                        "usage": last_res.get("usage", {}), "raw": last_res["text"],
                        "merge_attempts": attempts})
        else:
            rec.update({"pred_points": None, "parse_ok": False, "error": err,
                        "merge_attempts": attempts})
        out_recs.append(rec)
        # 2) Median-Baseline (ohne API)
        med = median_round([comp[g] for g in GRADERS])
        mrec = dict(base)
        mrec.update({"model_label": "ensemble", "model_requested": "median",
                     "prompt_version": "v6_median", "thinking": False,
                     "temperature": None, "top_p": None,
                     "pred_points": med, "parse_ok": med is not None,
                     "usage": {}})
        out_recs.append(mrec)
        return out_recs

    totals = {"merge_calls": 0, "parse_fail": 0, "errors": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    lock = threading.Lock()
    n = len(cells)
    with open(out_path, "w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(work, c) for c in cells]
            for i, fut in enumerate(as_completed(futures), 1):
                for rec in fut.result():
                    rec["run_id"] = run_id
                    with lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        if rec["prompt_version"] == "v6_ensemble":
                            totals["merge_calls"] += 1
                            if rec.get("error"):
                                totals["errors"] += 1
                            if not rec.get("parse_ok"):
                                totals["parse_fail"] += 1
                            u = rec.get("usage", {}) or {}
                            totals["prompt_tokens"] += u.get("prompt_tokens", 0)
                            totals["completion_tokens"] += u.get("completion_tokens", 0)
                            totals["reasoning_tokens"] += u.get("reasoning_tokens", 0) or 0
                with lock:
                    if i % 25 == 0 or i == n:
                        print(f"  {i}/{n} Merges "
                              f"(Parse-Fehler: {totals['parse_fail']}, "
                              f"API-Fehler: {totals['errors']})")

    print("--- fertig ---")
    print(f"  Merges: {totals['merge_calls']}, Parse-Fehler: {totals['parse_fail']}, "
          f"API-Fehler: {totals['errors']}")
    print(f"  Merge-Tokens: prompt={totals['prompt_tokens']}, "
          f"completion={totals['completion_tokens']}, "
          f"reasoning={totals['reasoning_tokens']}")
    print(f"  Ergebnisdatei: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
