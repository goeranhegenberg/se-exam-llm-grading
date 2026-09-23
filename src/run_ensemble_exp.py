"""Experimentier-Runner fuer Merge-Strategien (Ziel: den Median schlagen).

Wiederverwendet die Bausteine aus ``run_ensemble``, erlaubt aber freie Wahl von
Merger-Modell, Merge-Prompt, Self-Consistency und einer Antwort-Teilmenge. Die
Ergebnisse landen in ``results/experiments`` (nicht in ``results/raw``), damit die
regulaere Auswertung unberuehrt bleibt. Im Paper nicht verwendet (negativer
Befund: keine Strategie schlaegt den Median).

Beispiele:
    # strenges, kriteriumsweises Merge mit GLM nur auf dem Hard-Subset:
    python -m src.run_ensemble_exp --merge-prompt v6_merge_strict.txt \
        --answers hard --tag strict_glm
    # mit Self-Consistency (3 Merges, Median) und anderem Merger:
    python -m src.run_ensemble_exp --merger tertiary --sc 3 --tag sc3_oss
"""
import argparse
import sys
from pathlib import Path

from . import config as cfg
from .dataset import answer_meta, load_questions
from .llm_client import make_client
from .prompt_builder import load_system
from .run_ensemble import (GRADERS, build_merge_prompt, load_cells, load_v5,
                           median_round, one_merge, parse_grade, perm_for)
from .runner import run_and_write

# Hard-Subset: die 3 Median-Fehler + die 10 Uneinigkeits-Faelle des Benchmark-Laufs
# (siehe ``eval.ensemble_report``).
HARD = [
    "se-007-teil-para2", "se-013-teil-reform1", "se-016-teil-reform1",
    "se-001-teil-reform1", "se-004-teil-reform1", "se-009-teil-base",
    "se-009-teil-para2", "se-012-teil-base", "se-013-teil-para2",
    "se-015-teil-base", "se-015-teil-para2", "se-017-teil-para2",
    "se-018-teil-para2",
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Merge-Strategie-Experiment.")
    p.add_argument("--config", default=None)
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

    conf = cfg.load_config(args.config)
    questions = load_questions(args.dataset_dir or conf["paths"]["dataset_dir"])
    if args.answers == "all":
        want = None
    elif args.answers == "hard":
        want = set(HARD)
    else:
        want = {a.strip() for a in args.answers.split(",") if a.strip()}
    reps = conf["run"]["repetitions"]
    cells, _ = load_cells(questions, load_v5(args.raw), reps, want)
    if not cells:
        sys.exit("FEHLER: keine passenden V5-Zellen gefunden.")

    merger_model, merger_label = cfg.resolve_model(conf, args.merger)
    sampling = cfg.sampling_for(conf, merger_label, thinking=True)
    system = load_system(args.prompts_dir)
    template = (cfg.resolve_path(args.prompts_dir) / args.merge_prompt).read_text(encoding="utf-8")
    client = make_client(conf)
    concurrency = args.concurrency or conf["run"].get("concurrency", 8)

    run_id = cfg.utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "experiments" /
        f"exp_{args.tag}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pv = f"v6_{args.tag}"
    print(f"[exp {args.tag}] Merger={merger_model} (temp={sampling['temperature']}, "
          f"top_p={sampling['top_p']}) Prompt={args.merge_prompt} sc={args.sc} "
          f"Zellen={len(cells)} -> {out_path.name}")

    def work(cell):
        q, a, s, recs = cell
        grades = [parse_grade(r) for r in recs]
        comp = {g: grades[i][0] for i, g in enumerate(GRADERS)}
        order = perm_for(a["answer_id"], s)
        user = build_merge_prompt(template, q, a["text"], [grades[i] for i in order])
        preds, last = [], None
        for k in range(max(1, args.sc)):
            seed = (sampling["seed"] + 100 * k) if sampling["seed"] is not None else None
            pr, res, _, _ = one_merge(client, merger_model, system, user, sampling, q, a, seed)
            if pr is not None:
                preds.append(pr)
                last = res
        final = median_round(preds)
        return {
            "model_label": "ensemble", "model_requested": merger_model,
            "prompt_version": pv, "tag": args.tag, "thinking": True,
            "temperature": sampling["temperature"], "top_p": sampling["top_p"],
            "sc": args.sc, **answer_meta(q, a), "sample_index": s, "n_samples": reps,
            "component_points": comp, "sc_preds": preds, "pred_points": final,
            "parse_ok": final is not None,
            "raw": last["text"] if last else "",
        }

    run_and_write(work, cells, concurrency, out_path, run_id, what="Zellen")
    return str(out_path)


if __name__ == "__main__":
    main()
