"""Prompt-Templates für eine konkrete (Frage, Antwort)-Kombination rendern.

Platzhalter im Template werden als ``{{name}}`` notiert, damit die in den
Templates enthaltenen JSON-Beispiele (einfache geschweifte Klammern) nicht mit
der Ersetzung kollidieren.
"""
from .config import resolve_path

# V3 (Thinking) nutzt denselben Prompt wie V2 (Begruendung, ohne Rubrik); der
# einzige Unterschied ist das aktivierte modell-native Thinking (siehe
# config.run.thinking + Runner). V4 (Few-Shot) hat einen eigenen Prompt.
TEMPLATE_ALIASES = {"v3_thinking": "v2_begruendung"}


def _prompts_dir(prompts_dir="prompts"):
    return resolve_path(prompts_dir)


def load_system(prompts_dir="prompts"):
    with open(_prompts_dir(prompts_dir) / "system.txt", encoding="utf-8") as f:
        return f.read().strip()


def load_template(version, prompts_dir="prompts"):
    name = TEMPLATE_ALIASES.get(version, version)
    with open(_prompts_dir(prompts_dir) / f"{name}.txt", encoding="utf-8") as f:
        return f.read()


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def render_rubric(rubric):
    return "\n".join(
        f"{c['id']}. {c['criterion']} (max. {_fmt(c['points'])} P.)" for c in rubric
    )


def build_user_prompt(version, question, answer_text, prompts_dir="prompts"):
    tpl = load_template(version, prompts_dir)
    return (
        tpl.replace("{{frage}}", question["question"].strip())
        .replace("{{rubrik}}", render_rubric(question["rubric"]))
        .replace("{{antwort}}", answer_text.strip())
        .replace("{{max_punkte}}", _fmt(question["max_points"]))
    )
