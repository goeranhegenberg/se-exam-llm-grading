#!/usr/bin/env bash
# Kompletter Bewertungslauf: alle Modelle auf allen drei Datensaetzen, danach der
# Ensemble-Median. Ergebnisse landen in results/{raw,raw_human,raw_real}/run<N>/.
# Idempotent: ein erneuter Aufruf holt nur fehlende/fehlgeschlagene Aufrufe nach.
# Aufruf aus dem Paketverzeichnis:  ./run_all.sh 2      (Laufnummer)
set -uo pipefail
RUN=${1:?Laufnummer angeben, z.B. 2}
PY=${PYTHON:-python}
CONC=${CONCURRENCY:-6}          # Hauptmodell / Drittmodell
CONC_SENS=${CONCURRENCY_SENS:-3}  # Mistral: kleinerer Anbieter-Pool, sonst 429
for ds in "dataset:results/raw" "dataset_human:results/raw_human" "dataset_real:results/raw_real"; do
  dir=${ds%%:*}; out=${ds##*:}/run$RUN; mkdir -p "$out"
  $PY -m src.run_grading --model main --dataset-dir "$dir" --concurrency "$CONC" \
      --out "$out/grading_main.jsonl" --resume >> "$out/run_main.log" 2>&1 &
  $PY -m src.run_grading --model sensitivity --dataset-dir "$dir" --concurrency "$CONC_SENS" \
      --out "$out/grading_sensitivity.jsonl" --resume >> "$out/run_sensitivity.log" 2>&1 &
  $PY -m src.run_grading --model tertiary --prompt-versions v5_rubrik --dataset-dir "$dir" \
      --concurrency "$CONC" --out "$out/grading_tertiary.jsonl" --resume >> "$out/run_tertiary.log" 2>&1 &
  wait
  $PY -m src.run_ensemble --median-only --dataset-dir "$dir" --raw "$out" \
      --out "$out/grading_ensemble.jsonl" > "$out/run_ensemble.log" 2>&1
  echo "Lauf $RUN, $dir: fertig ($(grep -c '"error"' "$out"/grading_{main,sensitivity,tertiary}.jsonl | tr '\n' ' '))"
done
