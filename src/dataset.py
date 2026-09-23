"""Benchmark-Datensatz laden und gegen das JSON-Schema validieren."""
import json

import jsonschema

from .config import IMPL_DIR, resolve_path


def load_schema():
    with open(IMPL_DIR / "dataset" / "schema.json", encoding="utf-8") as f:
        return json.load(f)


def load_questions(dataset_dir="dataset"):
    """Lädt alle ``se-*.json``-Fragen, validiert sie und prüft die Konsistenz."""
    d = resolve_path(dataset_dir)
    schema = load_schema()
    questions = []
    for fp in sorted(d.glob("se-*.json")):
        with open(fp, encoding="utf-8") as f:
            q = json.load(f)
        jsonschema.validate(q, schema)

        # Konsistenz: Rubrik-Summe == max_points
        rub_sum = sum(c["points"] for c in q["rubric"])
        if rub_sum != q["max_points"]:
            raise ValueError(
                f"{fp.name}: Rubrik-Summe {rub_sum} != max_points {q['max_points']}"
            )
        # Konsistenz: jede Antwort-Punktzahl <= max_points
        for a in q["answers"]:
            if a["points"] > q["max_points"]:
                raise ValueError(
                    f"{fp.name}/{a['answer_id']}: points {a['points']} > "
                    f"max_points {q['max_points']}"
                )
        questions.append(q)

    if not questions:
        raise FileNotFoundError(f"Keine Benchmark-Fragen in {d} gefunden.")
    return questions


def iter_answer_items(questions):
    """Flache Liste von ``(question, answer)``-Paaren."""
    for q in questions:
        for a in q["answers"]:
            yield q, a


def answer_meta(q, a):
    """Die Frage-/Antwort-Felder, die jeder Ergebnis-Record mitführt."""
    return {
        "question_id": q["question_id"], "topic": q["topic"], "qtype": q["type"],
        "max_points": q["max_points"], "answer_id": a["answer_id"],
        "level": a["level"], "variant": a["variant"], "operation": a["operation"],
        "gt_points": a["points"], "answer_chars": len(a["text"]),
        "answer_words": len(a["text"].split()),
    }


def dataset_summary(questions):
    """Kennzahlen des Datensatzes für Logging und Tabellen."""
    n_answers = sum(len(q["answers"]) for q in questions)
    n_variants = sum(
        1 for q in questions for a in q["answers"] if a["operation"] != "base"
    )
    return {
        "n_questions": len(questions),
        "n_answers": n_answers,
        "n_base": n_answers - n_variants,
        "n_variants": n_variants,
        "topics": [q["topic"] for q in questions],
    }
