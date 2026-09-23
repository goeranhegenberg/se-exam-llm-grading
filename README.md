# Automatische und konsistente Bewertung von Software-Engineering-Klausuren mittels Prompt Engineering

Replikationspaket zur Seminararbeit *Automatische und konsistente Bewertung von
Software-Engineering-Klausuren mittels Prompt Engineering* (Göran Hegenberg,
Seminar PARSE 2026, Universität Leipzig). Enthalten sind der Benchmark
(18 Fragen, 162 bepunktete Antworten), zwei Validierungsdatensätze (33
menschlich verfasste und 35 authentische Studierendenantworten), die fünf
Prompt-Versionen, die Rohdaten aller Modellaufrufe und die Auswertung, die
sämtliche Tabellen und Abbildungen der Arbeit erzeugt.

*English summary:* Benchmark, prompts, raw model outputs and evaluation code for
LLM-based grading of German software-engineering exam answers. Five prompt
versions form an ablation ladder (baseline, rationale, thinking, few-shot,
rubric); the rubric turns out to be the decisive component. Everything needed
to regenerate the paper's tables and figures is included.

Die Pipeline lädt einen Benchmark aus Klausurfragen mit Rubrik und bepunkteten
Antwortvarianten, bewertet die Antworten mit fünf zunehmend reichhaltigen
Prompt-Versionen über zwei LLMs und misst Korrektheit, Konsistenz und Robustheit
der Bewertung. Die fünf Prompt-Versionen bilden eine Ablationsleiter: Jede
Version fügt im Kern einen Baustein hinzu, die Rubrik kommt bewusst als letztes
Element (V5).

## Struktur

```
se-exam-llm-grading/
├── config.json            # Modelle, Sampling-Parameter, Laufeinstellungen
├── requirements.txt       # Python-Abhängigkeiten (getestet mit Python 3.9 und 3.12)
├── .env.example           # Vorlage für den OpenRouter-Key (-> nach .env kopieren)
├── dataset/
│   ├── schema.json        # JSON-Schema einer Benchmark-Frage
│   └── se-0XX.json        # Fragen mit Rubrik + Antwortvarianten (Ground Truth)
├── dataset_human/         # Validierung 1: von den Autoren verfasste Antworten
│   └── se-1XX.json        #   (12 Fragen einer RUB-Klausur, 33 Antworten)
├── dataset_real/          # Validierung 2: authentische Studierendenantworten
│   ├── README.md          #   Herkunft, Ground Truth, Grenzentscheidungen
│   └── se-2XX.json        #   (5 Aufgaben, 35 transkribierte Antworten)
├── prompts/
│   ├── system.txt         # gemeinsamer System-Prompt (Anti-Bias-Regeln)
│   ├── v1_baseline.txt    # nur Antwort + Maximalpunktzahl, nur Punktzahl
│   ├── v2_begruendung.txt # + Begründung VOR der Punktzahl (auch V3 nutzt diesen Prompt)
│   ├── v4_fewshot.txt      # + verbesserte Anweisung + Few-Shot (ohne Rubrik)
│   └── v5_rubrik.txt       # + Bewertungsrubrik (V3 nutzt den V2-Prompt mit Thinking)
├── src/
│   ├── config.py          # Konfiguration + Geheimnisse laden
│   ├── dataset.py         # Benchmark laden + gegen Schema validieren
│   ├── prompt_builder.py  # Prompt-Templates rendern (V4 = V3-Prompt)
│   ├── llm_client.py      # OpenAI-/OpenRouter-Anbindung + Mock-Client
│   ├── util.py            # robustes JSON-Parsing
│   └── run_grading.py     # Bewertungslauf (Runner, nebenläufig)
├── eval/
│   ├── analyze.py         # Metriken, LaTeX-Tabellen, Abbildungen (Benchmark)
│   ├── ensemble_report.py # V6-Vergleich Median vs. Einzelmodelle
│   ├── human_report.py    # Auswertung dataset_human
│   └── real_report.py     # Auswertung dataset_real (inkl. NMAE, je Aufgabe)
└── results/
    ├── raw/               # eine JSONL-Zeile pro Modellaufruf (Benchmark)
    ├── raw_human/         # Rohdaten der Validierung auf dataset_human
    ├── raw_real/          # Rohdaten der Validierung auf dataset_real
    ├── tables/            # \input-fähige LaTeX-Tabellen
    ├── figures/           # PDF- und PNG-Abbildungen
    └── summary.json       # alle Kennzahlen maschinenlesbar
```

## Einrichtung

```powershell
# 1) Virtuelle Umgebung + Abhängigkeiten
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) API-Key hinterlegen (NICHT committen, .env steht in .gitignore)
Copy-Item .env.example .env
#   -> in .env einen GÜLTIGEN OpenRouter-Schlüssel eintragen:
#      OPENROUTER_API_KEY=sk-or-v1-...
#      OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Der Betrieb läuft OpenAI-kompatibel über **OpenRouter**. Der Key wird über
`implementation/.env` geladen (Vorrang vor einer veralteten Umgebungsvariablen).

## Ausführen

Aus dem Ordner `implementation/`:

```powershell
# Offline-Rauchtest der gesamten Pipeline ohne API (deterministischer Mock):
.\.venv\Scripts\python.exe -m src.run_grading --mock --limit 6

# Echter Lauf mit dem Hauptmodell (GLM 5.2) über alle fünf Prompt-Versionen:
.\.venv\Scripts\python.exe -m src.run_grading --model main --concurrency 10

# Lauf mit dem Zweitmodell (Mistral Small 4):
.\.venv\Scripts\python.exe -m src.run_grading --model sensitivity --concurrency 10

# Auswertung -> Tabellen, Abbildungen, summary.json:
.\.venv\Scripts\python.exe -m eval.analyze
```

> **Hinweis:** Vor einem *echten* Lauf alte/fremde Rohdaten aus `results/raw/`
> entfernen oder in einen Unterordner verschieben (z. B. `archive_*`); Mock-Läufe
> (`*_mock_*.jsonl`) blendet `eval.analyze` automatisch aus, da es alle `*.jsonl`
> im Ordner zusammenfasst.

## Datenformat des Benchmarks

Jede Frage ist eine JSON-Datei nach `dataset/schema.json`:

* `rubric`: Liste von Teilkriterien mit Punktwerten; ihre Summe ergibt
  `max_points`.
* `answers`: Antwortvarianten mit Ground-Truth-Punktzahl (`points`),
  Punktstufe (`level` ∈ {null, teil, voll}), Diversifizierungs-Operation
  (`operation`) und den erfüllten Kriterien (`rubric_fulfilled`).

`src/dataset.py` validiert beim Laden Schema *und* Konsistenz (Rubrik-Summe =
`max_points`, Antwortpunkte ≤ `max_points`).

## Prompt-Versionen (Ablationsleiter)

Die fünf Versionen bilden eine Ablationsleiter: Pro Stufe kommt genau **ein**
Element hinzu, sodass sich der Beitrag jeder Technik isoliert zuordnen lässt.
Bewusst wird die Rubrik — die naheliegende „Zutat" — bis zuletzt
zurückgehalten, um zu messen, wie weit Prompt Engineering *ohne* Rubrik trägt
und welchen Mehrwert die Rubrik am Ende noch hat.

| Version | Rubrik | Begründung | Few-Shot | Thinking | Idee |
|---------|:------:|:----------:|:--------:|:--------:|------|
| V1 Baseline   | – | – | – | – | nur Frage + Antwort + Maximalpunktzahl, Ausgabe nur `{punkte}` |
| V2 Begründung | – | ✓ | – | – | Begründung **vor** der Punktzahl im JSON (Reasoning-in-Output) |
| V3 Thinking   | – | ✓ | – | ✓ | wie V2, zusätzlich modell-natives Thinking aktiviert |
| V4 Few-Shot   | – | ✓ | ✓ | ✓ | verbesserte Anweisung + zwei gelöste Beispiele (besserer Prompt) |
| V5 Rubrik     | ✓ | ✓ | ✓ | ✓ | wie V4, zusätzlich mit Bewertungsrubrik im Prompt |

V3 verwendet denselben Prompt wie V2; der einzige Unterschied ist das aktivierte
Thinking (`run.thinking` in `config.json`, umgesetzt über
`extra_body={"reasoning": {"enabled": …}}` pro Aufruf).

### Erweiterung: Ensemble (V6)

Über die Leiter hinaus kombiniert **V6** drei quelloffene Reasoning-Modelle
unterschiedlicher Anbieter (GLM 5.2, Mistral Small 4, `openai/gpt-oss-120b`), die
jede Antwort mit dem V5-Prompt bewerten; die finale Punktzahl ist der **Median**
der drei Bewertungen.

```
python -m src.run_ensemble        # liest die V5-Rohdaten der drei Modelle
python -m eval.ensemble_report    # Vergleichstabelle (Median vs. Einzelmodelle)
```

Voraussetzung ist, dass V5 vorher für alle drei Modelle gerechnet wurde
(`--model main|sensitivity|tertiary --prompt-versions v5_rubrik`). Befund: Der
Median (98,1 % exakt) übertrifft jedes Einzelmodell. (`run_ensemble.py` berechnet
zusätzlich ein LLM-gestütztes Merge zum Vergleich; es bringt keinen Mehrwert über
den Median und wird im Paper nicht verwendet.)

## Validierung auf zwei weiteren Datensätzen

Der Benchmark ist teils LLM-gestützt erzeugt. Ob die Befunde auch für
menschliche Formulierungen gelten, prüfen zwei unabhängige Datensätze; beide
nutzen dieselbe Pipeline, lediglich mit anderem `--dataset-dir`.

| Datensatz | Inhalt | Antworten | Auswertung | Tabellen/Abbildungen |
|-----------|--------|----------:|------------|----------------------|
| `dataset_human` | 12 Fragen einer realen SE-Klausur (RUB, WS 2021/22), Antworten von den Autoren verfasst | 33 | `eval.human_report` | Tabelle 6 |
| `dataset_real`  | 5 offene Aufgaben einer realen Klausur (Hochschule Merseburg, SS 2026), authentische Studierendenantworten | 35 | `eval.real_report` | Tabellen 7/8, Abbildungen 3/4 |

```powershell
# Beispiel dataset_real: Läufe -> Ensemble -> Auswertung
.\.venv\Scripts\python.exe -m src.run_grading --model main        --dataset-dir dataset_real --out results/raw_real/grading_main.jsonl
.\.venv\Scripts\python.exe -m src.run_grading --model sensitivity --dataset-dir dataset_real --out results/raw_real/grading_sensitivity.jsonl
.\.venv\Scripts\python.exe -m src.run_grading --model tertiary --prompt-versions v5_rubrik --dataset-dir dataset_real --out results/raw_real/grading_tertiary.jsonl
.\.venv\Scripts\python.exe -m src.run_ensemble --dataset-dir dataset_real --raw results/raw_real --out results/raw_real/grading_ensemble.jsonl
.\.venv\Scripts\python.exe -m eval.real_report --raw results/raw_real
```

Weil die Aufgaben in `dataset_real` unterschiedliche Maximalpunktzahlen haben
(2–10 Punkte), berichtet `eval.real_report` zusätzlich den skalennormierten
Fehler **NMAE** (absoluter Fehler geteilt durch die Maximalpunktzahl) sowie eine
Aufschlüsselung je Aufgabe. Herkunft, Ground Truth und die offengelegten
Grenzentscheidungen stehen in `dataset_real/README.md`.

> **Hinweis:** Die in `results/tables/metrics_real.tex`, `per_task_real.tex` und
> `results/figures/exact_by_*_real.*` eingecheckten Werte stammen aus dem in der
> Arbeit berichteten Lauf. Ein erneuter Aufruf von `eval.real_report`
> überschreibt sie mit den Werten der jeweils vorliegenden Rohdaten.

## Modellwahl (Begründung)

Berichtet werden zwei **aktuelle, quelloffene** Modelle, die sich bewusst in der
Herkunft, aber nicht im Leistungssegment unterscheiden:

* **`z-ai/glm-5.2`** (Z.ai, China) — Hauptmodell.
* **`mistralai/mistral-small-2603`** (Mistral, Europa; „Mistral Small 4") —
  Zweitmodell für die Modellsensitivität.

Beide sind open-weight, relativ schnell und preisgünstig und unterstützen — für
das Untersuchungsdesign entscheidend — sowohl einen **Reasoning-** als auch einen
**Non-Reasoning-Modus**. So lässt sich das Thinking pro Prompt-Version gezielt
an- bzw. abschalten (V1–V2 aus, V3–V5 an), und der Effekt von Reasoning wird
modellübergreifend vergleichbar. Beide Modellnamen sind in `config.json`
konfigurierbar; berichtet wird stets das tatsächlich verwendete Modell.

## Reproduzierbarkeit

* Modellversion und Parameter (Temperatur, Top-p, Seed, Thinking-Flag) stehen in
  `config.json` und werden pro Aufruf in den Rohdaten protokolliert.
* Jeder Modellaufruf wird als eigene JSONL-Zeile gespeichert (inkl. Roh-Antwort,
  Token- und Reasoning-Token-Verbrauch).
* Die Non-Reasoning-Versionen (V1, V2) laufen determinismus-nah mit
  `temperature = 0`. Die Reasoning-Versionen (V3–V5) laufen mit der vom Anbieter
  empfohlenen Sampling-Konfiguration (`sampling.temperature_thinking` je Modell,
  `sampling.top_p_thinking`), da Reasoning-Modelle bei greedy decoding zu
  Wiederholungsschleifen/degenerierter Argumentation neigen (GLM 5.2: `1.0/0.95`,
  Mistral Small 4: `0.7/0.95`).
* Je Antwort werden `repetitions` Wiederholungen gezogen, um die Lauf-Streuung
  (Lauf-$\sigma$) zu messen; sie liegt bei den Reasoning-Versionen durch das
  stochastische Sampling naturgemäß höher.

## Änderungen 2026-09-23 (Überarbeitung der Auswertung)

* **Para-σ und Robustheit korrigiert** (`eval/analyze.py`, `metrics_for`): beide
  Kennzahlen gruppieren jetzt nach *(Frage, Referenzpunktzahl)* statt nach
  *(Frage, Punktstufe)*. Die Teil-Stufe der zwölf erweiterten Fragen enthält
  bewusst Antworten mit 1, 2 und 3 Punkten; unter der alten Gruppierung hätte
  selbst ein perfekter Bewerter Para-σ = 0,18 gehabt. Korrekt gruppiert:
  Para-σ (GLM) 0,018 → 0,006 (V1 → V5), Robustheit 0,036 → 0,012;
  Mistral 0,060/0,143 → 0,006/0,012.
* Neue Tabelle `results/tables/metrics_models.tex` (beide Modelle, alle
  Kennzahlen inkl. Robustheit) ersetzt im Paper `metrics_main` + `metrics_sensitivity`.
* Neue Synthese-Abbildung `python -m eval.overview_figure` →
  `results/figures/exact_overview.pdf` (exakte Quote V1–V6 auf allen drei
  Datensätzen; liest `summary.json`, `human_summary.json`, `real_summary.json`).

## Herkunft der Daten

* `dataset/` (Benchmark): eigene Fragen, orientiert an frei zugänglichen
  Klausur- und Übungsaufgaben (UZH-Musterklausur, Probeklausur Uni Bonn,
  LMU-Übungsblätter) und Lehrliteratur; Antwortvarianten teils LLM-gestützt
  erzeugt und manuell nachkontrolliert.
* `dataset_human/`: 12 Fragen einer Software-Engineering-Klausur
  (Ruhr-Universität Bochum, WS 2021/22), Antworten von den Autoren verfasst.
* `dataset_real/`: 35 authentische Antworten aus sieben anonymisierten Bögen
  einer Softwaretechnik-I-Klausur (Hochschule Merseburg, SS 2026), zugänglich
  über Kai Holland-Letz; nur als Transkript, ohne Namen und Matrikelnummern.
  Rubriken und Referenzpunkte stammen von den Autoren, nicht von der prüfenden
  Lehrkraft (Details und Grenzentscheidungen in `dataset_real/README.md`).

## Zitieren

Siehe `CITATION.cff`. Die Arbeit selbst ist eine unveröffentlichte
Seminararbeit (Universität Leipzig, 2026).

## Lizenz

MIT (siehe `LICENSE`) für Code, Prompts, Datensätze und Ergebnisdateien.
