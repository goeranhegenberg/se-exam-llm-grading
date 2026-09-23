"""Hilfsfunktionen zum robusten Parsen von Modellausgaben und zum Lesen der
JSONL-Rohdaten."""
import json
import re


def iter_jsonl(path):
    """Alle nicht-leeren Zeilen einer JSONL-Datei als Dicts liefern."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


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


_POINT_KEYS = ("punkte_gesamt", "punkte")
_RE_POINTS = [re.compile(rf'"{k}"\s*:\s*(-?\d+(?:[.,]\d+)?)') for k in _POINT_KEYS]


def salvage_points(text):
    """Rettungsanker, wenn ``extract_json`` an einer Modellantwort scheitert.

    Manche Modelle (GLM 5.2 mit aktiviertem Thinking im JSON-Modus) liefern
    JSON mit korrupten Schlüsseln (z.B. ``{"begru{": ...``), sodass es nicht
    strikt parsbar ist; das Punktefeld steht aber unbeschädigt im Text. Diese
    Funktion zieht die Gesamtpunktzahl per Regex heraus (bevorzugt
    ``punkte_gesamt``, sonst das erste ``punkte``-Feld). Gibt ``None``
    zurück, wenn nichts Plausibles gefunden wird. Gerettete Antworten werden
    im Record als ``parse_salvaged`` markiert und in der Auswertung gezählt.
    """
    if not text:
        return None
    for rx in _RE_POINTS:
        m = rx.search(text)
        if m:
            return coerce_points(m.group(1).replace(",", "."))
    return None


def extract_points(obj):
    """Liest die Gesamtpunktzahl aus einem geparsten Bewertungs-Objekt
    (``punkte_gesamt`` oder ``punkte``)."""
    if not isinstance(obj, dict):
        return None
    for key in _POINT_KEYS:
        if key in obj:
            p = coerce_points(obj[key])
            if p is not None:
                return p
    return None


def parse_pred(text):
    """Punktzahl aus einer Modellantwort: strikt (JSON) und sonst per Regex.

    Liefert ``(punkte, gerettet)``; ``punkte`` ist ``None``, wenn auch der
    Rettungsanker nichts findet.
    """
    obj, ok = extract_json(text)
    pred = extract_points(obj) if ok else None
    if pred is not None:
        return pred, False
    pred = salvage_points(text)
    return pred, pred is not None
