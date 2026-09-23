"""Gemeinsame Bausteine der Lauf-Skripte: Ergebnis-JSONL schreiben und
Aufrufe/Fehler/Tokens mitzählen."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class Totals:
    """Zählt Aufrufe, Parse-/API-Fehler und Token-Verbrauch eines Laufs."""

    def __init__(self):
        self.calls = self.parse_fail = self.errors = 0
        self.prompt_tokens = self.completion_tokens = self.reasoning_tokens = 0

    def add(self, rec):
        self.calls += 1
        self.errors += bool(rec.get("error"))
        self.parse_fail += not rec.get("parse_ok")
        u = rec.get("usage") or {}
        self.prompt_tokens += u.get("prompt_tokens", 0) or 0
        self.completion_tokens += u.get("completion_tokens", 0) or 0
        self.reasoning_tokens += u.get("reasoning_tokens", 0) or 0

    def status(self):
        return f"(Parse-Fehler: {self.parse_fail}, API-Fehler: {self.errors})"

    def report(self, what="Aufrufe"):
        print("--- fertig ---")
        print(f"  {what}: {self.calls}, Parse-Fehler: {self.parse_fail}, "
              f"API-Fehler: {self.errors}")
        print(f"  Tokens: prompt={self.prompt_tokens}, "
              f"completion={self.completion_tokens}, "
              f"reasoning={self.reasoning_tokens}")


def run_and_write(work, items, concurrency, out_path, run_id, count=None,
                  what="Aufrufe", append=False):
    """``work(item)`` parallel ausführen (liefert einen Record oder eine Liste
    von Records), jeden Record sofort als JSONL-Zeile schreiben und den
    Fortschritt melden. ``count`` filtert, welche Records in die Zählung
    eingehen (Default: alle); ``append`` hängt an eine bestehende Datei an."""
    totals = Totals()
    n = len(items)
    with open(out_path, "a" if append else "w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=concurrency) as ex:
        pending = {ex.submit(work, it) for it in items}
        for i, fut in enumerate(as_completed(pending), 1):
            pending.discard(fut)
            recs = fut.result()
            for rec in (recs if isinstance(recs, list) else [recs]):
                rec["run_id"] = run_id
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if count is None or count(rec):
                    totals.add(rec)
            fh.flush()
            if i % 25 == 0 or i == n:
                print(f"  {i}/{n} {what} {totals.status()}")
    totals.report(what)
    print(f"  Ergebnisdatei: {out_path}")
    return totals
