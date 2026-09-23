"""Hilfsfunktionen zum robusten Parsen von Modellausgaben."""
import json
import re


def extract_json(text):
    """Parst ein JSON-Objekt aus einer Modellantwort.

    Gibt ``(obj, ok)`` zurück; ``ok`` ist ``False``, wenn kein gültiges JSON
    gefunden wurde. Toleriert Code-Fences und Text um das JSON-Objekt herum.
    """
    if not text:
        return None, False
    text = text.strip()
    # Code-Fences (```json ... ```) entfernen
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # 1) Direktversuch
    try:
        return json.loads(text), True
    except Exception:
        pass
    # 2) Erstes balanciertes {...}-Objekt extrahieren
    start = text.find("{")
    if start == -1:
        return None, False
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1]), True
                except Exception:
                    return None, False
    return None, False


def coerce_points(value):
    """Wandelt einen Punktwert robust in ``float`` um, sonst ``None``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_RE_PUNKTE_GESAMT = re.compile(r'"punkte_gesamt"\s*:\s*(-?\d+(?:[.,]\d+)?)')
_RE_PUNKTE = re.compile(r'"punkte"\s*:\s*(-?\d+(?:[.,]\d+)?)')


def salvage_points(text):
    """Rettungsanker, wenn ``extract_json`` an einer Modellantwort scheitert.

    Manche Modelle (z.B. GLM 5.2 mit aktiviertem Thinking) mischen vereinzelt ein
    korruptes Fremdzeichen ins JSON, sodass es nicht mehr strikt parsbar ist; die
    eigentliche Punktzahl steht aber unbeschädigt im Text. Diese Funktion zieht
    die Gesamtpunktzahl per Regex heraus (bevorzugt ``punkte_gesamt``, sonst das
    erste ``punkte``-Feld). Gibt ``None`` zurück, wenn nichts Plausibles
    gefunden wird.
    """
    if not text:
        return None
    m = _RE_PUNKTE_GESAMT.search(text) or _RE_PUNKTE.search(text)
    return coerce_points(m.group(1).replace(",", ".")) if m else None


def extract_points(obj):
    """Liest die Gesamtpunktzahl aus einem geparsten Bewertungs-Objekt.

    Unterstützt das Baseline-Format (``punkte``) und das strukturierte Format
    (``punkte_gesamt``). Fällt notfalls auf die Summe der Kriteriumspunkte
    zurück.
    """
    if not isinstance(obj, dict):
        return None
    for key in ("punkte_gesamt", "punkte"):
        if key in obj:
            p = coerce_points(obj[key])
            if p is not None:
                return p
    if isinstance(obj.get("kriterien"), list):
        total = 0.0
        any_ok = False
        for k in obj["kriterien"]:
            p = coerce_points(k.get("punkte")) if isinstance(k, dict) else None
            if p is not None:
                total += p
                any_ok = True
        if any_ok:
            return total
    return None
