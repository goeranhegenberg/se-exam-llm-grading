# `dataset_real` — authentische Studierendenantworten

Dritter und härtester Validierungsdatensatz der Arbeit (Abschnitt 5.8). Anders
als der synthetische Benchmark (`dataset/`, teils LLM-gestützt erzeugt) und der
zweite Validierungsdatensatz (`dataset_human/`, von den Autoren verfasst)
stammen diese Antworten von echten Prüfungsteilnehmenden.

## Herkunft

* **Quelle:** Softwaretechnik-I-Klausur, Hochschule Merseburg, SS 2026.
* **Umfang:** sieben Bögen × fünf offene Aufgaben = **35 Antworten**.
* **Zugang:** Die Bögen waren über Kai Holland-Letz zugänglich, der dieselben
  Bögen in einer eigenen Seminararbeit mit anderen Modellen, eigenen Rubriken
  und eigenen Referenzpunkten untersucht hat.
* **Anonymisierung:** Die Bögen enthalten weder Namen noch Matrikelnummern; die
  entsprechenden Kopffelder sind leer. Aufgenommen sind ausschließlich die
  **transkribierten** Antworttexte, keine Scans.

Die Transkription ist **wortgetreu**, einschließlich Rechtschreib- und
Grammatikfehlern (`Depancy Inversione`, `Inteface agregartion`), englischer
Passagen, Aufzählungsartefakten und **leerer Felder** (nicht bearbeitete
Teilaufgaben stehen als leerer `text`). Genau dieses Klausur-Rauschen ist der
Zweck des Datensatzes; eine Glättung würde ihn entwerten.

## Aufgaben

| ID | Aufgabe | Max. | Typ |
|----|---------|-----:|-----|
| `se-201` | SOLID-Prinzipien nennen | 5 | definition |
| `se-202` | Liskov-Prinzip erklären | 2 | erklaerung |
| `se-203` | Aggregation vs. Komposition | 2 | vergleich |
| `se-204` | Scrum-Bestandteile nennen | 6 | definition |
| `se-205` | Architekturstil für ein Szenario wählen und begründen | 10 | begruendung |

Format und Validierung entsprechen dem Benchmark (`dataset/schema.json`):
Rubrik-Summe = `max_points`, Antwortpunkte ≤ `max_points`. Da es keine
Formulierungsvarianten gibt, ist jede Antwort `operation: "base"`; `variant`
trägt die Bogennummer (`bogen-1` … `bogen-7`). Optional ist `short_label`, die
Kurzbezeichnung für Tabellen und Abbildungen.

## Ground Truth

Offizielle Korrekturpunkte der prüfenden Lehrkraft lagen **nicht** vor. Für jede
Aufgabe wurde daher eine ganzzahlige Rubrik erstellt (ein Punkt je Kriterium,
Summe gleich Klausurpunktzahl) und jede Antwort kriteriumsweise von Hand
bepunktet. `rubric_fulfilled` hält fest, welche Kriterien erfüllt sind, und
bestimmt damit `points`.

Verteilung: **5 Null-, 16 Teil-, 14 Voll-Antworten.**

Diese Ground Truth ist autor:innendefiniert — sie misst, ob ein Modell *unsere*
Rubrik-Anwendung reproduziert, nicht die Übereinstimmung mit der prüfenden
Lehrkraft. Abschnitt 6.3 der Arbeit führt das als Konstruktbedrohung.

### Grenzentscheidungen

Die folgenden Fälle sind vertretbar auch anders zu bepunkten und werden hier
offengelegt:

* **`se-201`, Bogen 3** — `I: Inteface agregartion`. Der Begriff ist gemeint als
  *Interface Segregation*, aber „agregartion“ trifft die Bedeutung nicht
  (Aggregation ≠ Segregation). Kriterium 4 gilt als **nicht** erfüllt (4 statt
  5 Punkte). Dies ist die einzige Abweichung, die auch die Modelle
  durchgängig anders sehen.
* **`se-201`, Bogen 4** — `D: Dependency Principle` ohne „Inversion“. Der
  entscheidende Namensbestandteil fehlt; Kriterium 5 **nicht** erfüllt (4 Punkte).
* **`se-201`, Bogen 6** — nennt kein einziges Prinzip, sondern beschreibt nur den
  Zweck von SOLID. Die Frage verlangt ausdrücklich die Nennung: **0 Punkte**.
* **`se-203`, Bogen 2** — Aggregation und Komposition sind **vertauscht**
  definiert. Da beide Kriterien die jeweils richtige Zuordnung verlangen:
  **0 Punkte**, nicht 1.
* **`se-204`, Bogen 6** — „Entwickler, Kaufer, Softwar Enginerer, scrum master“:
  Scrum Master und Entwickler zählen, „Kaufer“/„Software Engineer“ sind keine
  Scrum-Rollen. „scrum arbeitet in kleine schritten (sprints)“ nennt den Sprint
  als Bestandteil des Vorgehens; der Sprint ist nach dem Scrum Guide selbst ein
  Ereignis und steht deshalb ausdrücklich in Kriterium 4. **3 Punkte**.
* **`se-204`, Bogen 7** — Rollen und Artefakte vollständig, aber **kein
  Ereignis als Bestandteil genannt**: „Sprint“ kommt nur attributiv vor
  („Sprint Prozesse“, „Ergebniss nach einem Sprint“, „Sprint Backlog“), die
  Antwort gliedert sich ausdrücklich in „Rollen“ und „Bestandteile“ und führt
  unter Letzteren nur die drei Artefakte. **5 statt 6 Punkte**. Das ist die
  strittigste Einzelentscheidung des Datensatzes: Wer „nach einem Sprint“ als
  Nennung gelten lässt, kommt auf 6 Punkte (Stufe `voll`) und verschiebt die
  Verteilung auf 5/15/15. Genau hier bricht V3 im Thinking-Modus auf 3 Punkte
  ein (Abschnitt 5.8).
* **`se-205`** — Die Kriterien „Nachteil begründet“ (4), „Vergleich mit Bezug zum
  Szenario“ (7) und „NFR begründet“ (10) haben Ermessensspielraum: Eine bloße
  Aufzählung ohne Begründung bzw. eine Vergleichstabelle ohne Szenariobezug gilt
  als **nicht** erfüllt. Die verbleibenden Modell-Abweichungen bei dieser Aufgabe
  liegen sämtlich an diesen drei Kriterien.
* **`se-205`, Kriterium 8 (API-Risiko) und 9 (zwei NFR)** sind bewusst
  **großzügig** angewendet: „network communication overhead“ (Bogen 4) zählt als
  Risiko, „Zugriff über Mobile oder Desktop“ und „braucht Internetverbindung“
  (Bogen 2) zählen als nicht-funktionale Anforderungen. Kriterium 10 verlangt
  keine getrennte Begründung je Anforderung (Bogen 7: eine Begründung für beide).
* **Leere Antworten** (`se-203`/Bogen 4, `se-204`/Bogen 2 und 4) erhalten
  0 Punkte und die Stufe `null`.

### Nachprüfung (2026-09-08)

Alle 35 Bewertungen wurden ein zweites Mal gegen die Bögen geprüft: Rubrikanzahl
gleich Maximalpunktzahl in allen fünf Aufgaben, `rubric_fulfilled` ↔ `points` ↔
`level` durchgängig konsistent, keine Punktvergabe geändert. Angepasst wurden
nur Texte: Frage- und Rubriktexte mit echten Umlauten (wie in `dataset/` und
in der Klausur), Aufgabe 5 im **Original-Wortlaut** des Bogens statt der
bisherigen Kurzfassung, und Kriterium 4 von `se-204` führt den Sprint jetzt
explizit als Ereignis. Da die Modelle den Fragetext sehen, gehört der noch
ausstehende echte Lauf (`results/raw_real/`) ohnehin wiederholt.

## Lauf und Auswertung

Protokoll wie bei den übrigen Läufen: drei Wiederholungen je Antwort, gleiche
Sampling-Konfiguration. Insgesamt **1.155 Bewertungsaufrufe**
(35 Antworten × 33 = V1–V5 für Haupt- und Zweitmodell sowie V5 für das dritte
Modell, je dreimal).

```bash
# Bewertungslauf (aus implementation/)
python -m src.run_grading --model main         --dataset-dir dataset_real --out results/raw_real/grading_main.jsonl
python -m src.run_grading --model sensitivity  --dataset-dir dataset_real --out results/raw_real/grading_sensitivity.jsonl
python -m src.run_grading --model tertiary --prompt-versions v5_rubrik \
                                               --dataset-dir dataset_real --out results/raw_real/grading_tertiary.jsonl

# Ensemble-Median (V6) aus den V5-Rohdaten der drei Modelle
python -m src.run_ensemble --dataset-dir dataset_real --raw results/raw_real \
                           --out results/raw_real/grading_ensemble.jsonl

# Auswertung -> Tabellen 7/8, Abbildungen 3/4, real_summary.json
python -m eval.real_report --raw results/raw_real
```

Offline-Rauchtest ohne API-Key: dieselben Befehle mit `--mock`. Die Mock-Werte
sind Pseudo-Bewertungen und dürfen **nicht** in die Arbeit übernommen werden.
