# Vorgehensweise — Konsultation Prompt Engineering (III.2)

**Thema:** Automatic and Consistent Evaluation of Software Engineering Exams
**Forschungsfrage:** Wie lassen sich Prompts so gestalten, dass Antworten auf
Software-Engineering-Klausurfragen über verschiedene Formulierungen und
Punktestufen hinweg automatisch und konsistent bewertet werden können?

Ziel dieses Dokuments ist es, der Konsultation eine klare Diskussionsgrundlage
zu geben: Was möchte ich bauen, wie gehe ich systematisch vor, und an welchen
Stellen brauche ich noch Feedback?

---

## 1. Verständnis der Aufgabe

Aus der Aufgabenstellung leite ich vier Bausteine ab:

1. **Benchmark-Datensatz** aus Software-Engineering-Klausurfragen mit Begründung
   der Auswahl.
2. **Bewertungskriterien / Punkteschemata** je Frage, plus Antworten auf
   verschiedenen Punktstufen (Ground Truth).
3. **Diversifizierung** der Antworten durch Paraphrasierung, Umstellung und
   Teilumformulierung — bei gleicher intendierter Punktzahl.
4. **Prompt-System**, das Antworten gegen die Rubrik bepunktet und dabei
   robust und reproduzierbar ist.

Erwarteter Output ist ein Ground-Truth- und prompt-basiertes Bewertungssystem,
inkl. Auswertungsmetriken und versioniertem Repository.

---

## 2. Gesamtplan

Phasen, in der Reihenfolge, wie ich sie umsetzen möchte:

| Phase | Inhalt | Artefakt |
|-------|--------|----------|
| P1 | Recherche und Stand der Forschung | Übersichtstabelle, Kapitel "Grundlagen" |
| P2 | Auswahl der Klausurfragen + Rubriken | Benchmark-Spec (JSON-Schema) |
| P3 | Erzeugung von Antworten je Punktstufe | Ground-Truth-Antworten |
| P4 | Diversifizierung (Paraphrasen, Umstellungen) | Erweiterte Benchmark-Datei |
| P5 | Prompt-Entwurf (Baseline -> iterativ) | `prompts/v1..vn.md` |
| P6 | Evaluation: Korrektheit, Konsistenz, Robustheit | Auswertungs-Skripte + Tabellen |
| P7 | Diskussion, Ausblick, Schreiben | Fertige Seminararbeit |

---

## 3. Systematische Literaturrecherche

Die Bewertungskriterien des Lehrstuhls verlangen explizit, dass *vor* der
Recherche definiert wird: **wonach, wo und wie** gesucht wird, und **wie
gefiltert** wird. Mein Plan:

### 3.1 Suchbegriffe
Cluster, die ich zunächst durchsuche:

- **Bewertung freier Antworten:**
  `automated short answer grading`, `ASAG`, `automatic essay scoring`,
  `LLM exam grading`, `rubric-based scoring`
- **Prompt-Methoden:**
  `prompt engineering`, `chain of thought`, `few-shot grading`,
  `self-consistency`, `LLM-as-a-judge`
- **Robustheit / Konsistenz:**
  `paraphrase robustness LLM`, `LLM grading reliability`,
  `consistency LLM evaluation`, `inter-rater agreement LLM`
- **Software-Engineering-Lehre:**
  `software engineering exam`, `programming concepts assessment`

### 3.2 Suchplattformen
Primär: Google Scholar, ACM Digital Library, IEEE Xplore, arXiv.
Sekundär: Anthropic / OpenAI / Google DeepMind Dokumentation und Blogs (für
Prompt-Patterns, nicht als Primärquelle für wissenschaftliche Aussagen).

### 3.3 Suchstrategie
- Iterativ pro Cluster: Top-Treffer prüfen, anschließend Citation-Tracking
  (vor- und rückwärts).
- Filterkriterien: Aktualität (bevorzugt 2022+), Reproduzierbarkeit (Datensatz
  und Methode öffentlich), Bezug zu freitextlicher Bewertung.
- Alle aufgenommenen Quellen kommen in eine Übersichtstabelle
  (Titel, Jahr, Methodik, Datensatz, Metrik, Relevanz). Diese Tabelle steuert
  später, was in das Grundlagenkapitel kommt.

### 3.4 Analysevorgehen
Pro Quelle notiere ich: Forschungsfrage, Methode, Datensatz, Metriken,
Limitationen, Übertragbarkeit auf meinen Kontext (Software-Engineering-
Klausuren, deutscher Sprachraum).

**Frage an den Prof:** Sollen rein deutschsprachige Klausurfragen verwendet
werden, oder ist Englisch ebenfalls erlaubt / gewünscht?

---

## 4. Benchmark-Datensatz

### 4.1 Auswahl der Fragen
Ich plane einen Benchmark, der gezielt mehrere typische Antworttypen abdeckt:

- **Definition:** "Was ist X?" (z.B. Definition von Coupling/Cohesion)
- **Erklärung:** "Erklären Sie, warum X gilt." (z.B. SOLID-Prinzipien)
- **Vergleich:** "Welche Unterschiede gibt es zwischen X und Y?"
  (z.B. White-Box vs. Black-Box-Tests)
- **Anwendung / Beispiel:** "Geben Sie ein Beispiel für X."
- **Bewertung / Begründung:** "Warum würden Sie X gegenüber Y bevorzugen?"

Größenordnung als Diskussionsvorschlag: ca. **8–12 Fragen**, je
Frage **3 Punktstufen** (z.B. 0, halb, voll), je Punktstufe
**3–5 Paraphrasen** -> insgesamt ~100–180 bewertete Antworten. Klein genug,
um vollständig manuell zu auditieren, groß genug für stabile Statistik.

### 4.2 Rubriken / Punkteschema
Pro Frage definiere ich eine Rubrik mit mehreren Teilkriterien und zugeordneten
Punkten (z.B. *Definition korrekt: 1P, Begründung: 1P, Beispiel: 1P*). Die
Rubrik ist Grundlage sowohl für die Erstellung der Musterantworten als auch
für den Prompt.

### 4.3 Erzeugung der Antworten
Antworten werden je Punktstufe geschrieben. Damit der Punktwert wirklich
"verdient" ist, gehe ich rubrikweise vor: für jede Teilkriterium entscheide
ich, ob es erfüllt ist, und schreibe die Antwort entsprechend. So ist die
Punktzahl reproduzierbar herleitbar.

### 4.4 Diversifizierung
Pro Antwort erzeuge ich mehrere Varianten:

- **Paraphrasierung:** andere Wortwahl, gleicher Inhalt.
- **Umstellung:** andere Reihenfolge der genannten Punkte.
- **Teilumformulierung:** Mischung aus formal / umgangssprachlich,
  unterschiedliche Detailtiefe (knapp vs. ausführlich).

Pro Variante wird notiert, welche Operation angewandt wurde, und es wird
manuell verifiziert, dass die Punktzahl unverändert bleibt.

**Frage an den Prof:** Ist es zulässig, die Diversifizierung mit einem LLM
zu unterstützen (mit anschließender manueller Prüfung), oder sollen die
Varianten vollständig manuell entstehen?

### 4.5 Datenformat
JSON pro Frage:

```json
{
  "question_id": "se-001",
  "question": "...",
  "rubric": [{"criterion": "...", "points": 1}, ...],
  "answers": [
    {
      "points": 2,
      "variant": "base",
      "text": "..."
    },
    {
      "points": 2,
      "variant": "paraphrase-1",
      "text": "..."
    }
  ]
}
```

Versionierung im Git-Repository, sodass jede Änderung am Benchmark
nachvollziehbar bleibt.

---

## 5. Prompt-Engineering-Ansatz

Iteratives Vorgehen, jede Version dokumentiert:

- **V1 — Baseline:** Aufgabenbeschreibung, Rubrik, Antwort, Ausgabe als JSON
  mit Feldern `punkte` und `begruendung`.
- **V2 — Strukturierte Rubrik:** Rubrik wird in Teilkriterien aufgeteilt; das
  Modell bewertet jedes Teilkriterium einzeln und summiert.
- **V3 — Few-Shot:** Hinzunahme weniger gelöster Beispiele (mit Begründung).
- **V4 — Chain-of-Thought:** Explizite Aufforderung zu Zwischenüberlegungen
  vor der finalen Punktzahl.
- **V5 — Self-Consistency:** Mehrfaches Sampling, Mehrheit / Median als
  Endergebnis.

Pro Version wird festgehalten: was wurde geändert, welche Hypothese motiviert
die Änderung, welches Ergebnis ergibt sich auf demselben Benchmark.

**Frage an den Prof:** Welches Modell / welche Modelle sollen primär verwendet
werden? Vorschlag von meiner Seite: ein Hauptmodell (z.B. Claude oder GPT) für
die Auswertung, ergänzend ein zweites Modell für eine
Modell-Sensitivitätsanalyse.

---

## 6. Evaluation

Bewertet wird entlang von vier Dimensionen — passend zu Korrektheit, Tiefe und
methodischer Bewertung aus den Bewertungskriterien:

1. **Korrektheit:** mittlerer absoluter Punktefehler gegenüber Ground Truth,
   Confusion-Matrix über Punktstufen.
2. **Konsistenz:** Streuung über Wiederholungen (gleiche Antwort, mehrere
   Modellläufe) und über Paraphrasen derselben Punktstufe. Hier zeigt sich,
   ob das Setup *unabhängig von der Formulierung* bewertet.
3. **Robustheit:** Verhalten bei gezielten Störungen (Umstellung,
   Teilumformulierung, sprachliche Vereinfachung).
4. **Reproduzierbarkeit:** alle Modellaufrufe mit fixierter Modellversion und
   protokollierten Parametern; Code und Prompts versioniert.

Visualisierung geplant über Box-Plots (Streuung pro Punktstufe), Heatmaps
(Confusion-Matrix) und Tabellen (Vergleich der Prompt-Versionen).

---

## 7. Reproduzierbarkeit und Werkzeuge

- Sprache: Python (Skripte für Benchmark-Aufruf und Auswertung).
- Modellzugriff: über offizielle SDKs (Anthropic / OpenAI), Modellversion und
  Parameter (Temperatur, Top-p, Seed sofern verfügbar) werden geloggt.
- Versionierung: Git-Repository mit Code, Prompts, Benchmark-Daten und
  Auswertungen. Tags für die Stände, die in die Arbeit einfließen.
- Auswertungen erzeugen automatisch Tabellen / Plots, die in der LaTeX-Arbeit
  eingebunden werden.

---

## 8. Risiken und offene Punkte

- **Größe des Benchmarks:** zu klein -> Ergebnisse nicht aussagekräftig; zu
  groß -> manuelle Pflege nicht mehr leistbar. Vorschlag: starten mit ca.
  8 Fragen, ggf. erweitern.
- **Modellabhängigkeit:** Ergebnisse können stark zwischen Modellen
  variieren. Lösungsansatz: zumindest eine Sensitivitätsanalyse mit zwei
  Modellen.
- **Bias in der Bewertung:** Längen-Bias, formale Sprache wird ggf. besser
  bewertet als korrekter Inhalt. Lösungsansatz: gezielte Diversifizierung
  (kurz/lang, formal/informell) und Auswertung dieses Effekts.
- **Kostenbudget:** mehrfaches Sampling und Self-Consistency erhöhen Kosten.
  Lösungsansatz: Budget vorab festlegen und in der Arbeit transparent
  ausweisen.

---

## 9. Konkrete Fragen an den Prof

1. **Sprache des Benchmarks:** Deutsch, Englisch oder gemischt?
2. **Nutzung von LLMs für die Diversifizierung der Antworten** (mit
   anschließendem manuellen Audit) — zulässig?
3. **Modellauswahl:** ein Hauptmodell + Sensitivitätscheck OK, oder gibt es
   Vorgaben?
4. **Größenordnung des Benchmarks:** ist meine Schätzung
   (8–12 Fragen × 3 Punktstufen × 3–5 Paraphrasen) angemessen?
5. **Quellenarten:** Sind Blogposts / Modell-Anbieter-Dokumentation als
   Sekundärquellen für Prompt-Patterns akzeptabel?
6. **Implementierungstiefe:** Wird ein lauffähiges, automatisiertes
   Auswertungsskript erwartet, oder reicht eine reproduzierbare manuelle
   Auswertung?

---

## 10. Zeitlicher Rahmen (Vorschlag)

| Woche | Inhalt |
|-------|--------|
| 1 | Recherche + Übersichtstabelle, erstes Grundlagenkapitel |
| 2 | Benchmark-Spec, Rubriken, erste Antworten |
| 3 | Diversifizierung, Benchmark final |
| 4 | Prompts V1–V3 + erste Auswertungen |
| 5 | Prompts V4–V5, Sensitivitätsanalyse |
| 6 | Schreiben, Diskussion, Anhang, finale Korrektur |
