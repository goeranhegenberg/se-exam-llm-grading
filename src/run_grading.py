"""Bewertungslauf: ruft das Modell für jede (Prompt-Version, Antwort)-Kombination
auf, parst die Punktzahl und schreibt jede Einzelbewertung als JSONL-Zeile.

Die fünf Prompt-Versionen bilden eine Ablationsleiter, bei der pro Stufe genau
ein Element hinzukommt:

    V1 Baseline   -- ohne Rubrik, nur Punktzahl
    V2 Begründung -- ohne Rubrik, Begründung VOR der Punktzahl
    V3 Thinking   -- wie V2, zusätzlich modell-natives Thinking aktiviert
    V4 Few-Shot   -- ohne Rubrik, verbesserte Anweisung + Few-Shot (besserer Prompt)
    V5 Rubrik     -- wie V4, zusätzlich mit Bewertungsrubrik im Prompt

Beispiele:
    # Offline-Test der gesamten Pipeline ohne API-Schlüssel:
    python -m src.run_grading --mock --limit 6

    # Echter Lauf mit dem Hauptmodell (GLM 5.2) über alle Prompt-Versionen:
    python -m src.run_grading --model main

    # Sensitivitätslauf mit dem Zweitmodell (Mistral Small 4):
    python -m src.run_grading --model sensitivity

    # Abgebrochenen oder von API-Fehlern betroffenen Lauf vervollständigen
    # (fehlende und fehlgeschlagene Aufrufe werden nachgeholt, der Rest bleibt):
    python -m src.run_grading --model sensitivity --out <datei> --resume
"""
import argparse
import json
from pathlib import Path

from . import config as cfg
from .dataset import answer_meta, dataset_summary, iter_answer_items, load_questions
from .llm_client import make_client
from .prompt_builder import build_user_prompt, load_system
from .runner import run_and_write
from .util import iter_jsonl, parse_pred


def job_key(rec):
    return (rec["prompt_version"], rec["answer_id"], rec["sample_index"])


def resume_records(out_path, jobs):
    """Bestehende Ausgabedatei einlesen: Records ohne API-Fehler behalten, die
    zugehörigen Jobs streichen. Liefert ``(behaltene Records, offene Jobs)``."""
    if not out_path.exists():
        return [], jobs
    keep = [r for r in iter_jsonl(out_path) if not r.get("error")]
    done = {job_key(r) for r in keep}
    todo = [j for j in jobs
            if (j["version"], j["answer"]["answer_id"], j["sample_index"]) not in done]
    return keep, todo


def build_jobs(questions, versions, conf, model_label):
    """Erzeugt die Liste der Einzelaufrufe (ein Sample = ein Job). Jede Antwort
    wird ``repetitions`` mal wiederholt, um die Lauf-Streuung zu messen; über
    ``run.thinking`` ist je Version hinterlegt, ob das modell-native Thinking
    aktiv ist (und damit die Sampling-Konfiguration, siehe
    ``config.sampling_for``)."""
    reps = conf["run"]["repetitions"]
    thinking_map = conf["run"].get("thinking", {})
    jobs = []
    for version in versions:
        thinking = bool(thinking_map.get(version, False))
        sampling = cfg.sampling_for(conf, model_label, thinking)
        for q, a in iter_answer_items(questions):
            for s in range(reps):
                jobs.append({"version": version, "question": q, "answer": a,
                             "sample_index": s, "n_samples": reps,
                             "thinking": thinking, **sampling})
    return jobs


def grade_job(job, client, system, model_name, model_label, prompts_dir):
    """Einen einzelnen Job ausführen und den fertigen Ergebnis-Record liefern."""
    q, a, version = job["question"], job["answer"], job["version"]
    user = build_user_prompt(version, q, a["text"], prompts_dir)
    rec = {"model_label": model_label, "model_requested": model_name,
           "prompt_version": version, "thinking": job["thinking"],
           **answer_meta(q, a),
           "sample_index": job["sample_index"], "n_samples": job["n_samples"],
           "temperature": job["temperature"], "top_p": job["top_p"]}
    try:
        res = client.grade(
            model=model_name, system=system, user=user,
            temperature=job["temperature"], top_p=job["top_p"],
            max_tokens=job["max_tokens"], seed=job["seed"],
            json_mode=True, thinking=job["thinking"],
            ground_truth=a["points"], max_points=q["max_points"],
        )
        pred, salvaged = parse_pred(res["text"])
        rec.update({
            "model_returned": res.get("model"),
            "pred_points": pred,
            "parse_ok": pred is not None,
            "parse_salvaged": salvaged,
            "finish_reason": res.get("finish_reason"),
            "usage": res.get("usage", {}),
            "raw": res["text"],
        })
    except Exception as e:  # einzelne Fehler nicht den ganzen Lauf killen
        rec.update({"pred_points": None, "parse_ok": False,
                    "error": f"{type(e).__name__}: {e}"})
    return rec


def limit_answers(questions, limit):
    """Nur die ersten ``limit`` Antworten behalten (für schnelle Tests)."""
    trimmed, count = [], 0
    for q in questions:
        keep = q["answers"][: max(0, limit - count)]
        count += len(keep)
        if keep:
            trimmed.append({**q, "answers": keep})
        if count >= limit:
            break
    return trimmed


def main(argv=None):
    p = argparse.ArgumentParser(description="Prompt-basierter Bewertungslauf.")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default="main",
                   help="Schlüssel aus config.models (main/sensitivity/tertiary) oder Modellname.")
    p.add_argument("--prompt-versions", default=None,
                   help="Kommagetrennt; Default: config.run.prompt_versions.")
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None,
                   help="Parallele Modellaufrufe; Default: config.run.concurrency.")
    p.add_argument("--limit", type=int, default=None,
                   help="Nur die ersten N Antworten verwenden (für schnelle Tests).")
    p.add_argument("--mock", action="store_true", help="Mock-Client statt API.")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true",
                   help="Bestehende --out-Datei fortsetzen: nur fehlende oder mit "
                        "API-Fehler abgebrochene Aufrufe nachholen.")
    args = p.parse_args(argv)

    conf = cfg.load_config(args.config)
    if args.repetitions is not None:
        conf["run"]["repetitions"] = args.repetitions
    versions = (args.prompt_versions.split(",") if args.prompt_versions
                else conf["run"]["prompt_versions"])
    questions = load_questions(args.dataset_dir or conf["paths"]["dataset_dir"])
    if args.limit is not None:
        questions = limit_answers(questions, args.limit)
    summary = dataset_summary(questions)

    model_name, model_label = cfg.resolve_model(conf, args.model)
    client = make_client(conf, mock=args.mock, noise=1)
    # Mock ist deterministisch und billig; seriell reicht.
    concurrency = 1 if args.mock else (args.concurrency or conf["run"].get("concurrency", 8))
    system = load_system(args.prompts_dir)
    jobs = build_jobs(questions, versions, conf, model_label)

    run_id = cfg.utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "raw" /
        f"grading_{model_label}_{'mock' if args.mock else 'live'}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run {run_id}] Modell={model_name} ({model_label}) "
          f"Versionen={versions} Concurrency={concurrency}")
    print(f"  Datensatz: {summary['n_questions']} Fragen, "
          f"{summary['n_answers']} Antworten -> {len(jobs)} Modellaufrufe")
    kept = []
    if args.resume:
        kept, jobs = resume_records(out_path, jobs)
        print(f"  Fortsetzung: {len(kept)} Aufrufe vorhanden, {len(jobs)} offen")
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in kept:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Ausgabe: {out_path}")
    if not jobs:
        print("--- fertig --- nichts nachzuholen")
        return str(out_path)

    run_and_write(lambda job: grade_job(job, client, system, model_name,
                                        model_label, args.prompts_dir),
                  jobs, concurrency, out_path, run_id, append=bool(kept))
    return str(out_path)


if __name__ == "__main__":
    main()
