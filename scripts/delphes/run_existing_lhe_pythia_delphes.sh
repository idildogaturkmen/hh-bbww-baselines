#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 SAMPLE_NAME INPUT_LHE_GZ N_EVENTS OUTPUT_TAG"
  exit 1
fi

SAMPLE_NAME="$1"
INPUT_LHE_GZ="$2"
N_EVENTS="$3"
OUTPUT_TAG="$4"

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"
: "${HH4B_SOFTWARE:?Need HH4B_SOFTWARE}"
: "${DELPHES_DIR:?Need DELPHES_DIR}"

mkdir -p \
  "$HH4B_STORE/hepmc" \
  "$HH4B_STORE/root" \
  "$HH4B_STORE/logs" \
  "$HH4B_STORE/metadata" \
  "$HH4B_STORE/parquet"

LHE="${INPUT_LHE_GZ%.gz}"
HEPMC="$HH4B_STORE/hepmc/${OUTPUT_TAG}_pythia8.hepmc"
ROOT_OUT="$HH4B_STORE/root/${OUTPUT_TAG}_pythia8_delphes.root"

echo "Sample: $SAMPLE_NAME"
echo "Input LHE.GZ: $INPUT_LHE_GZ"
echo "Events requested: $N_EVENTS"
echo "Output tag: $OUTPUT_TAG"
echo "LHE: $LHE"
echo "HEPMC: $HEPMC"
echo "ROOT: $ROOT_OUT"

zcat "$INPUT_LHE_GZ" > "$LHE"

N_FOUND=$(grep -c "<event>" "$LHE")
echo "LHE event count: $N_FOUND"

if [ "$N_FOUND" -lt "$N_EVENTS" ]; then
  echo "ERROR: LHE has only $N_FOUND events, but requested $N_EVENTS"
  exit 2
fi

rm -f "$HEPMC" "$ROOT_OUT"

"$HH4B_SOFTWARE/bin/lhe_to_hepmc3" \
  "$LHE" \
  "$HEPMC" \
  "$N_EVENTS" \
  2>&1 | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_lhe_to_hepmc3.log"

"$DELPHES_DIR/DelphesHepMC3" \
  "${HH4B_DELPHES_CARD:-$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc.tcl}" \
  "$ROOT_OUT" \
  "$HEPMC" \
  2>&1 | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_pythia8_delphes.log"

python3 "$HH4B_REPO/scripts/delphes/inspect_delphes_root.py" "$ROOT_OUT"

python3 "$HH4B_REPO/scripts/delphes/diagnose_delphes_hh4b_smoke.py" \
  --input "$ROOT_OUT" \
  --out "$HH4B_STORE/metadata/${OUTPUT_TAG}_diagnostic.csv" \
  2>&1 | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_diagnostic.log"

python3 "$HH4B_REPO/scripts/delphes/make_delphes_event_summary.py" \
  --input "$ROOT_OUT" \
  --outdir "$HH4B_STORE/parquet" \
  --sample "$OUTPUT_TAG"

echo "DONE: $OUTPUT_TAG"
echo "ROOT: $ROOT_OUT"
echo "HEPMC: $HEPMC"
echo "PARQUET: $HH4B_STORE/parquet/${OUTPUT_TAG}_event_summary.parquet"
echo "DIAGNOSTIC: $HH4B_STORE/metadata/${OUTPUT_TAG}_diagnostic.csv"
