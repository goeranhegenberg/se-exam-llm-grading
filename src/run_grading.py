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
"""
import argparse
import datetime as dt
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg
from .dataset import dataset_summary, iter_answer_items, load_questions
from .llm_client import GradingClient, MockClient
from .prompt_builder import build_user_prompt, load_system
from .util import extract_json, extract_points, salvage_points


def _utcstamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_model(conf, model_arg):
    """``main``/``sensitivity`` über config auflösen, sonst als Modellname nehmen."""
    models = conf.get("models", {})
    if model_arg in models:
        return models[model_arg], model_arg
    return model_arg, model_arg


def build_jobs(questions, versions, conf, model_label):
    """Erzeugt die Liste der Einzelaufrufe (ein Sample = ein Job).

    Non-Reasoning-Versionen laufen determinismus-nah (Temperatur 0, top_p 1).
    Reasoning-/Thinking-Versionen laufen dagegen mit der vom Anbieter
    empfohlenen Sampling-Konfiguration (``sampling.temperature_thinking``
    je Modell, ``sampling.top_p_thinking``), weil Reasoning-Modelle bei greedy
    decoding (Temperatur 0) zu Wiederholungsschleifen/degenerierter Argumentation
    neigen. Jede Antwort wird ``repetitions`` mal wiederholt, um die Lauf-Streuung
    zu messen; über ``run.thinking`` ist je Version hinterlegt, ob das
    modell-native Thinking aktiv ist.
    """
    reps = conf["run"]["repetitions"]
    sampling = conf["sampling"]
    base_temp = sampling["temperature"]
    base_top_p = sampling["top_p"]
    thinking_map = conf["run"].get("thinking", {})
    mt_default = sampling["max_tokens"]
    mt_thinking = sampling.get("max_tokens_thinking", mt_default)
    # Empfohlene Sampling-Konfiguration für den Thinking-Modus (pro Modell).
    tt_map = sampling.get("temperature_thinking") or {}
    temp_thinking = tt_map.get(model_label, tt_map.get("default", base_temp))
    tp_cfg = sampling.get("top_p_thinking", base_top_p)
    if isinstance(tp_cfg, dict):
        top_p_thinking = tp_cfg.get(model_label, tp_cfg.get("default", base_top_p))
    else:
        top_p_thinking = tp_cfg

    jobs = []
    for version in versions:
        thinking = bool(thinking_map.get(version, False))
        max_tokens = mt_thinking if thinking else mt_default
        temperature = temp_thinking if thinking else base_temp
        top_p = top_p_thinking if thinking else base_top_p
        for q, a in iter_answer_items(questions):
            for s in range(reps):
                jobs.append({
                    "version": version,
                    "question": q,
                    "answer": a,
                    "sample_index": s,
                    "n_samples": reps,
                    "temperature": temperature,
                    "top_p": top_p,
                    "thinking": thinking,
                    "max_tokens": max_tokens,
                })
    return jobs


def grade_job(job, client, system, model_name, sampling, prompts_dir):
    """Einen einzelnen Job ausführen und den fertigen Ergebnis-Record liefern."""
    q, a, version = job["question"], job["answer"], job["version"]
    user = build_user_prompt(version, q, a["text"], prompts_dir)
    rec = {
        "model_label": None,  # vom Aufrufer gesetzt
        "model_requested": model_name,
        "prompt_version": version,
        "thinking": job["thinking"],
        "question_id": q["question_id"],
        "topic": q["topic"],
        "qtype": q["type"],
        "max_points": q["max_points"],
        "answer_id": a["answer_id"],
        "level": a["level"],
        "variant": a["variant"],
        "operation": a["operation"],
        "gt_points": a["points"],
        "answer_chars": len(a["text"]),
        "answer_words": len(a["text"].split()),
        "sample_index": job["sample_index"],
        "n_samples": job["n_samples"],
        "temperature": job["temperature"],
        "top_p": job["top_p"],
    }
    extra_body = {"reasoning": {"enabled": bool(job["thinking"])}}
    try:
        res = client.grade(
            model=model_name, system=system, user=user,
            temperature=job["temperature"], top_p=job["top_p"],
            max_tokens=job["max_tokens"], seed=sampling.get("seed"),
            json_mode=True, extra_body=extra_body,
            ground_truth=a["points"], max_points=q["max_points"],
        )
        obj, ok = extract_json(res["text"])
        pred = extract_points(obj) if ok else None
        salvaged = False
        if pred is None:
            pred = salvage_points(res["text"])
            salvaged = pred is not None
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Prompt-basierter Bewertungslauf.")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default="main",
                   help="Schlüssel aus config.models (main/sensitivity) oder Modellname.")
    p.add_argument("--prompt-versions", default=None,
                   help="Kommagetrennt; Default: config.run.prompt_versions.")
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None,
                   help="Parallele Modellaufrufe; Default: config.run.concurrency.")
    p.add_argument("--limit", type=int, default=None,
                   help="Nur die ersten N Antworten verwenden (für schnelle Tests).")
    p.add_argument("--mock", action="store_true", help="Mock-Client statt OpenAI.")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    conf = cfg.load_config(args.config)
    if args.repetitions is not None:
        conf["run"]["repetitions"] = args.repetitions
    concurrency = args.concurrency or conf["run"].get("concurrency", 8)
    versions = (args.prompt_versions.split(",") if args.prompt_versions
                else conf["run"]["prompt_versions"])
    dataset_dir = args.dataset_dir or conf["paths"]["dataset_dir"]

    questions = load_questions(dataset_dir)
    if args.limit is not None:
        trimmed, count = [], 0
        for q in questions:
            keep = q["answers"][: max(0, args.limit - count)]
            count += len(keep)
            if keep:
                q = {**q, "answers": keep}
                trimmed.append(q)
            if count >= args.limit:
                break
        questions = trimmed

    summary = dataset_summary(questions)
    model_name, model_label = resolve_model(conf, args.model)

    if args.mock:
        client = MockClient(noise=1)
        model_name = model_name or "mock"
        concurrency = 1  # Mock ist deterministisch + billig; seriell reicht.
    else:
        api_key = cfg.get_api_key()
        if not api_key:
            sys.exit("FEHLER: Kein API-Key gefunden. Trage einen gültigen "
                     "OPENROUTER_API_KEY (oder OPENAI_API_KEY) in "
                     "implementation/.env ein (siehe .env.example).")
        provider = conf.get("provider", {})
        base_url = provider.get("base_url") or cfg.get_base_url()
        extra_body = provider.get("extra_body")  # i.d.R. None -> pro Job gesetzt
        client = GradingClient(api_key, base_url=base_url, extra_body=extra_body)

    system = load_system(args.prompts_dir)
    jobs = build_jobs(questions, versions, conf, model_label)

    run_id = _utcstamp()
    out_path = Path(args.out) if args.out else cfg.resolve_path(
        Path(conf["paths"]["results_dir"]) / "raw" /
        f"grading_{model_label}_{'mock' if args.mock else 'live'}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sampling = conf["sampling"]
    print(f"[run {run_id}] Modell={model_name} ({model_label}) "
          f"Versionen={versions} Concurrency={concurrency}")
    print(f"  Datensatz: {summary['n_questions']} Fragen, "
          f"{summary['n_answers']} Antworten -> {len(jobs)} Modellaufrufe")
    print(f"  Ausgabe: {out_path}")

    totals = {"calls": 0, "parse_fail": 0, "errors": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    lock = threading.Lock()
    n = len(jobs)

    with open(out_path, "w", encoding="utf-8") as fh:
        def submit(job):
            rec = grade_job(job, client, system, model_name, sampling,
                            args.prompts_dir)
            rec["run_id"] = run_id
            rec["model_label"] = model_label
            return rec

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(submit, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    totals["calls"] += 1
                    if rec.get("error"):
                        totals["errors"] += 1
                    if not rec.get("parse_ok"):
                        totals["parse_fail"] += 1
                    u = rec.get("usage", {}) or {}
                    totals["prompt_tokens"] += u.get("prompt_tokens", 0)
                    totals["completion_tokens"] += u.get("completion_tokens", 0)
                    totals["reasoning_tokens"] += u.get("reasoning_tokens", 0) or 0
                    if i % 25 == 0 or i == n:
                        print(f"  {i}/{n} Aufrufe "
                              f"(Parse-Fehler: {totals['parse_fail']}, "
                              f"API-Fehler: {totals['errors']})")

    print("--- fertig ---")
    print(f"  Aufrufe: {totals['calls']}, Parse-Fehler: {totals['parse_fail']}, "
          f"API-Fehler: {totals['errors']}")
    print(f"  Tokens: prompt={totals['prompt_tokens']}, "
          f"completion={totals['completion_tokens']}, "
          f"reasoning={totals['reasoning_tokens']}")
    print(f"  Ergebnisdatei: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
