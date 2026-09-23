"""Experimentier-Runner fuer Merge-Strategien (Ziel: den Median schlagen).

Wiederverwendet die Bausteine aus ``run_ensemble``, erlaubt aber freie Wahl von
Merger-Modell, Merge-Prompt, Self-Consistency und einer Antwort-Teilmenge. Die
Ergebnisse landen in ``results/experiments`` (nicht in ``results/raw``), damit die
regulaere Auswertung unberuehrt bleibt.

Beispiele:
    # strenges, kriteriumsweises Merge mit GLM nur auf dem Hard-Subset:
    python -m src.run_ensemble_exp --merge-prompt v6_merge_strict.txt \
        --answers hard --tag strict_glm
    # mit Self-Consistency (3 Merges, Median) und anderem Merger:
    python -m src.run_ensemble_exp --merger tertiary --sc 3 --tag sc3_oss
"""
import argparse
import datetime as dt
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg
from .dataset import iter_answer_items, load_questions
from .llm_client import GradingClient
from .prompt_builder import load_system
from .run_ensemble import (GRADERS, MERGE_ATTEMPTS, build_merge_prompt, load_v5,
                           median_round, parse_grade, perm_for)
from .util import extract_json, extract_points, salvage_points

# Hard-Subset: die 3 Median-Fehler + die 10 Uneinigkeits-Faelle (aus diag_hard).
HARD = [
    "se-007-teil-para2", "se-013-teil-reform1", "se-016-teil-reform1",
    "se-001-teil-reform1", "se-004-teil-reform1", "se-009-teil-base",
    "se-009-teil-para2", "se-012-teil-base", "se-013-teil-para2",
    "se-015-teil-base", "se-015-teil-para2", "se-017-teil-para2",
    "se-018-teil-para2",
]


def _utcstamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_merger(conf, merger):
    models = conf.get("models", {})
    if merger in models:
        return models[merger], merger
    return merger, merger


def main(argv=None):
    p = argparse.ArgumentParser(description="Merge-Strategie-Experiment.")
    p.add_argument("--merger", default="main", help="Modell-Schluessel (config) oder Name.")
    p.add_argument("--merge-prompt", default="v6_merge.txt")
    p.add_argument("--answers", default="all", help="'all', 'hard' oder Komma-Liste von answer_ids.")
    p.add_argument("--sc", type=int, default=1, help="Self-Consistency: k Merges, Median.")
    p.add_argument("--tag", default="exp")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    conf = cfg.load_config(args.config) if getattr(args, "config", None) else cfg.load_config()
    sampling = conf["sampling"]
    dataset_dir = args.dataset_dir or conf["paths"]["dataset_dir"]
    questions = load_questions(dataset_dir)

    if args.answers == "all":
        want = None
    elif args.answers == "hard":
        want = set(HARD)
    else:
        want = set(a.strip() for a in args.answers.split(",") if a.strip())

    v5 = load_v5(args.raw)
    reps = conf["run"]["repetitions"]
    cells = []
    for q, a in iter_answer_items(questions):
        if want is not None and a["answer_id"] not in want:
            continue
        for s in range(reps):
            recs = [v5.get((g, q["question_id"], a["answer_id"], s)) for g in GRADERS]
            if all(r is not None for r in recs):
                cells.append((q, a, s, recs))
    if not cells:
        sys.exit("FEHLER: keine passenden V5-Zellen gefunden.")

    merger_model, merger_label = resolve_merger(conf, args.merger)
    tt = sampling.get("temperature_thinking", {})
    tp = sampling.get("top_p_thinking", sampling["top_p"])
    merge_temp = tt.get(merger_label, tt.get("default", sampling["temperature"]))
    merge_top_p = (tp.get(merger_label, tp.get("default", sampling["top_p"]))
                   if isinstance(tp, dict) else tp)
    merge_max_tokens = sampling.get("max_tokens_thinking", sampling["max_tokens"])

    system = load_system(args.prompts_dir)
    template = (cfg.resolve_path(args.prompts_dir) / args.merge_prompt).read_text(encoding="utf-8")

    api_key = cfg.get_api_key()
    if not api_key:
        sys.exit("FEHLER: kein OPENROUTER_API_KEY in .env.")
    provider = conf.get("provider", {})
    client = GradingClient(api_key, base_url=provider.get("base_url") or cfg.get_base_url())
    concurrency = args.concurrency or conf["run"].get("concurrency", 8)

    run_id = _utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "experiments" /
        f"exp_{args.tag}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pv = f"v6_{args.tag}"
    print(f"[exp {args.tag}] Merger={merger_model} (temp={merge_temp}, top_p={merge_top_p}) "
          f"Prompt={args.merge_prompt} sc={args.sc} Zellen={len(cells)} -> {out_path.name}")

    def one_merge(user, q, a, seed):
        for attempt in range(MERGE_ATTEMPTS):
            sd = (seed + attempt) if seed is not None else None
            try:
                res = client.grade(
                    model=merger_model, system=system, user=user,
                    temperature=merge_temp, top_p=merge_top_p,
                    max_tokens=merge_max_tokens, seed=sd, json_mode=True,
                    extra_body={"reasoning": {"enabled": True}},
                    ground_truth=a["points"], max_points=q["max_points"])
                obj, ok = extract_json(res["text"])
                pred = extract_points(obj) if ok else None
                if pred is None:
                    pred = salvage_points(res["text"])
                if pred is not None:
                    return pred, res
            except Exception:
                pass
        return None, None

    def work(cell):
        q, a, s, recs = cell
        grades = [parse_grade(r) for r in recs]
        comp = {g: grades[i][0] for i, g in enumerate(GRADERS)}
        order = perm_for(a["answer_id"], s)
        user = build_merge_prompt(template, q, a["text"], [grades[i] for i in order])
        base_seed = sampling.get("seed")
        preds, last = [], None
        for k in range(max(1, args.sc)):
            seed = (base_seed + 100 * k) if base_seed is not None else None
            pr, res = one_merge(user, q, a, seed)
            if pr is not None:
                preds.append(pr)
                last = res
        final = median_round(preds) if preds else None
        rec = {
            "model_label": "ensemble", "model_requested": merger_model,
            "prompt_version": pv, "tag": args.tag, "thinking": True,
            "temperature": merge_temp, "top_p": merge_top_p, "sc": args.sc,
            "question_id": q["question_id"], "topic": q["topic"], "qtype": q["type"],
            "max_points": q["max_points"], "answer_id": a["answer_id"],
            "level": a["level"], "variant": a["variant"], "operation": a["operation"],
            "gt_points": a["points"], "answer_chars": len(a["text"]),
            "answer_words": len(a["text"].split()), "sample_index": s, "n_samples": reps,
            "component_points": comp, "sc_preds": preds, "pred_points": final,
            "parse_ok": final is not None, "run_id": run_id,
            "raw": (last or {}).get("text", "") if last else "",
        }
        return rec

    lock = threading.Lock()
    fail = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(work, c) for c in cells]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if not rec["parse_ok"]:
                        fail += 1
    print(f"--- fertig --- {len(cells)} Zellen, Parse-Fehler: {fail}")
    print(f"  Ergebnisdatei: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
