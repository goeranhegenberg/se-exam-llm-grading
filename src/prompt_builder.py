"""Prompt-Templates für eine konkrete (Frage, Antwort)-Kombination rendern.

Platzhalter im Template werden als ``{{name}}`` notiert, damit die in den
Templates enthaltenen JSON-Beispiele (einfache geschweifte Klammern) nicht mit
der Ersetzung kollidieren.
"""
from functools import lru_cache

from .config import resolve_path

# V3 (Thinking) nutzt denselben Prompt wie V2 (Begruendung, ohne Rubrik); der
# einzige Unterschied ist das aktivierte modell-native Thinking (siehe
# config.run.thinking + Runner). V4 (Few-Shot) hat einen eigenen Prompt.
TEMPLATE_ALIASES = {"v3_thinking": "v2_begruendung"}


@lru_cache(maxsize=None)
def load_system(prompts_dir="prompts"):
    return (resolve_path(prompts_dir) / "system.txt").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def load_template(version, prompts_dir="prompts"):
    """Template einer Prompt-Version (einmal gelesen, danach aus dem Cache)."""
    name = TEMPLATE_ALIASES.get(version, version)
    return (resolve_path(prompts_dir) / f"{name}.txt").read_text(encoding="utf-8")


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def render_rubric(rubric):
    return "\n".join(
        f"{c['id']}. {c['criterion']} (max. {_fmt(c['points'])} P.)" for c in rubric
    )


def fill_template(tpl, question, answer_text, **extra):
    """Platzhalter ``{{frage}}``, ``{{rubrik}}``, ``{{antwort}}``,
    ``{{max_punkte}}`` (und weitere aus ``extra``) ersetzen."""
    out = (tpl.replace("{{frage}}", question["question"].strip())
           .replace("{{rubrik}}", render_rubric(question["rubric"]))
           .replace("{{antwort}}", answer_text.strip())
           .replace("{{max_punkte}}", _fmt(question["max_points"])))
    for key, val in extra.items():
        out = out.replace("{{" + key + "}}", val)
    return out


def build_user_prompt(version, question, answer_text, prompts_dir="prompts"):
    return fill_template(load_template(version, prompts_dir), question, answer_text)
