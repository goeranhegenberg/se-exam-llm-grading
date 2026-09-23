"""Konfiguration (config.json) und Geheimnisse (.env / Umgebung) laden."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

IMPL_DIR = Path(__file__).resolve().parent.parent


def load_config(path=None):
    cfg_path = Path(path) if path else IMPL_DIR / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def get_api_key():
    """API-Key laden -- ``implementation/.env`` hat Vorrang vor der Umgebung.

    Bevorzugt ``OPENROUTER_API_KEY`` (OpenRouter-Betrieb), fällt sonst auf
    ``OPENAI_API_KEY`` zurück. Hintergrund: Ein im Terminal vererbter (evtl.
    veralteter) Key würde sonst eine gültige Angabe in ``.env`` überschreiben.
    ``override=True`` sorgt dafür, dass der dokumentierte Weg über ``.env``
    zuverlässig greift.
    """
    load_dotenv(IMPL_DIR / ".env", override=True)
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def get_base_url():
    """Optionale Basis-URL eines OpenAI-kompatiblen Endpoints (z.B. OpenRouter).

    Quelle: ``OPENAI_BASE_URL`` aus ``.env``/Umgebung. ``None`` -> Standard-OpenAI.
    """
    load_dotenv(IMPL_DIR / ".env", override=True)
    return os.environ.get("OPENAI_BASE_URL")


def resolve_path(rel):
    """Relative Pfade gegen das implementation/-Verzeichnis auflösen."""
    p = Path(rel)
    return p if p.is_absolute() else (IMPL_DIR / p)
