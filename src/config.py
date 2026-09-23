"""Konfiguration (config.json) und Geheimnisse (.env / Umgebung) laden."""
import datetime as dt
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
    """API-Key laden -- ``.env`` im Paketordner hat Vorrang vor der Umgebung.

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
    """Relative Pfade gegen das Paketverzeichnis auflösen."""
    p = Path(rel)
    return p if p.is_absolute() else (IMPL_DIR / p)


def resolve_model(conf, key):
    """``main``/``sensitivity``/``tertiary`` über ``config.models`` auflösen,
    sonst den Wert als Modellnamen nehmen. Liefert ``(modellname, label)``."""
    models = conf.get("models", {})
    return (models[key], key) if key in models else (key, key)


def sampling_for(conf, model_label, thinking):
    """Sampling-Parameter für einen Aufruf.

    Non-Reasoning-Aufrufe laufen determinismus-nah (``sampling.temperature``,
    ``sampling.top_p``). Reasoning-/Thinking-Aufrufe verwenden die vom Anbieter
    empfohlene Konfiguration je Modell (``temperature_thinking``,
    ``top_p_thinking``, jeweils Dict mit ``default``), weil Reasoning-Modelle
    bei greedy decoding zu Wiederholungsschleifen neigen.
    """
    s = conf["sampling"]
    if not thinking:
        return {"temperature": s["temperature"], "top_p": s["top_p"],
                "max_tokens": s["max_tokens"], "seed": s.get("seed")}
    tt = s.get("temperature_thinking") or {}
    tp = s.get("top_p_thinking") or {}
    return {"temperature": tt.get(model_label, tt.get("default", s["temperature"])),
            "top_p": tp.get(model_label, tp.get("default", s["top_p"])),
            "max_tokens": s.get("max_tokens_thinking", s["max_tokens"]),
            "seed": s.get("seed")}


def utcstamp():
    """Zeitstempel für Lauf-IDs und Dateinamen (UTC, sekundengenau)."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
