#!/usr/bin/env bash
# Wiederholt ./run_all.sh <N>, bis alle Ausgabedateien vollstaendig und frei von
# API-Fehlern sind (max. 30 Durchgaenge, 5 min Pause bei Rate-Limits).
set -uo pipefail
RUN=${1:?Laufnummer}
for i in $(seq 1 30); do
  ./run_all.sh "$RUN"
  bad=0
  for d in results/raw results/raw_human results/raw_real; do
    for f in "$d/run$RUN"/grading_{main,sensitivity,tertiary}.jsonl; do
      [ -e "$f" ] || { bad=1; continue; }
      grep -q '"error"' "$f" && bad=1
    done
  done
  if [ "$bad" -eq 0 ]; then echo "Lauf $RUN vollstaendig (Durchgang $i)"; exit 0; fi
  echo "Lauf $RUN: noch Luecken/API-Fehler, naechster Durchgang in 5 min ($i/30)"; sleep 300
done
echo "Lauf $RUN: nach 30 Durchgaengen nicht vollstaendig"; exit 1
