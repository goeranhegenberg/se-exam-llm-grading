#!/usr/bin/env bash
# Kompletter Bewertungslauf N: jedes Modell arbeitet die drei Datensaetze
# nacheinander ab (die Modelle laufen parallel), danach der Ensemble-Median je
# Datensatz. Ergebnisse: results/{raw,raw_human,raw_real}/run<N>/.
# Idempotent: ein erneuter Aufruf holt nur fehlende/fehlgeschlagene Aufrufe nach.
# Aufruf aus dem Paketverzeichnis:  ./run_all.sh 2
set -uo pipefail
RUN=${1:?Laufnummer angeben, z.B. 2}
PY=${PYTHON:-python}
CONC=${CONCURRENCY:-6}            # Hauptmodell / Drittmodell
CONC_SENS=${CONCURRENCY_SENS:-2}  # Mistral: geteilter Anbieter-Pool (~6-9 Aufrufe/min), sonst nur 429
# Mistral zuerst V5 (Ensemble, Tabellen 3-5), dann die uebrigen Versionen.
SENS_VERSIONS=v5_rubrik,v1_baseline,v2_begruendung,v3_thinking,v4_fewshot
DATASETS=("dataset:results/raw" "dataset_human:results/raw_human" "dataset_real:results/raw_real")

grade_all() {  # $1 = Modell-Label, $2 = Concurrency, $3.. = weitere Argumente
  local model=$1 conc=$2; shift 2
  for ds in "${DATASETS[@]}"; do
    local dir=${ds%%:*} out=${ds##*:}/run$RUN; mkdir -p "$out"
    $PY -m src.run_grading --model "$model" --dataset-dir "$dir" --concurrency "$conc" "$@" \
        --out "$out/grading_$model.jsonl" --resume >> "$out/run_$model.log" 2>&1
  done
}
grade_all main "$CONC" &
grade_all tertiary "$CONC" --prompt-versions v5_rubrik &
grade_all sensitivity "$CONC_SENS" --prompt-versions "$SENS_VERSIONS" &
wait
for ds in "${DATASETS[@]}"; do
  dir=${ds%%:*}; out=${ds##*:}/run$RUN
  $PY -m src.run_ensemble --median-only --dataset-dir "$dir" --raw "$out" \
      --out "$out/grading_ensemble.jsonl" > "$out/run_ensemble.log" 2>&1
  echo "Lauf $RUN, $dir: $(cat "$out"/grading_{main,sensitivity,tertiary}.jsonl | grep -c '"error"') API-Fehler"
done
