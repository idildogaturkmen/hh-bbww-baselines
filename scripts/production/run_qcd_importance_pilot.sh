#!/usr/bin/env bash
set -eo pipefail

N_EVENTS="${1:-20}"

export HH4B_REPO="/uscms_data/d3/$USER/repos/hh-bbww-baselines"
export HH4B_STORE="/uscms_data/d3/$USER/hh4b_delphes"

cd "$HH4B_REPO"

source \
  scripts/delphes/setup_lpc_delphes_env.sh

set -u

CARD="$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"

EXPECTED_CARD_HASH="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

OBSERVED_CARD_HASH=$(
  sha256sum "$CARD" | awk '{print $1}'
)

if [[ "$OBSERVED_CARD_HASH" != "$EXPECTED_CARD_HASH" ]]; then
  echo "ERROR: frozen-v2 card hash mismatch"
  exit 1
fi

for COMMAND in \
  g++ \
  python3 \
  root-config
do
  command -v "$COMMAND" >/dev/null || {
    echo "ERROR: missing command: $COMMAND"
    exit 2
  }
done

CAMPAIGN="qcd_hardqcd_importance_pilot_${N_EVENTS}ev_20260716"

OUTBASE="$HH4B_STORE/qcd_pilots/$CAMPAIGN"
LOGBASE="$HH4B_STORE/logs/$CAMPAIGN"

mkdir -p \
  "$OUTBASE/hepmc" \
  "$OUTBASE/root" \
  "$OUTBASE/parquet" \
  "$OUTBASE/metadata" \
  "$LOGBASE"

GENERATOR_SOURCE="$HH4B_REPO/scripts/production/generate_pythia8_hardqcd_hepmc3.cc"
COMPILE_HELPER="$HH4B_REPO/scripts/production/compile_pythia8_hepmc3.sh"

GENERATOR="$OUTBASE/generate_pythia8_hardqcd_hepmc3"

echo "Compiling HardQCD generator..."

"$COMPILE_HELPER" \
  "$GENERATOR_SOURCE" \
  "$GENERATOR" \
  "$LOGBASE/compile_arguments.txt"

test -x "$GENERATOR"

BINS=(
  "50 75"
  "75 100"
  "100 200"
  "200 300"
  "300 500"
  "500 700"
  "700 1000"
  "1000 0"
)

BIN_ID=0

for SPECIFICATION in "${BINS[@]}"; do
  read -r PTHAT_MIN PTHAT_MAX \
    <<< "$SPECIFICATION"

  BIN_PADDED=$(printf '%02d' "$BIN_ID")
  SEED=$((910000 + BIN_ID))

  if [[ "$PTHAT_MAX" == "0" ]]; then
    BIN_LABEL="pthat${PTHAT_MIN}toInf"
  else
    BIN_LABEL="pthat${PTHAT_MIN}to${PTHAT_MAX}"
  fi

  TAG="${CAMPAIGN}_bin${BIN_PADDED}_${BIN_LABEL}_seed${SEED}"

  HEPMC="$OUTBASE/hepmc/${TAG}.hepmc"
  ROOT_FILE="$OUTBASE/root/${TAG}_delphes.root"
  GENERATOR_JSON="$OUTBASE/metadata/${TAG}_generator.json"
  EVENT_SUMMARY="$OUTBASE/parquet/${TAG}_event_summary.parquet"
  CANDIDATES="$OUTBASE/parquet/${TAG}_hh4b_candidates_v2.parquet"

  echo
  echo "=================================================="
  echo "BIN_ID=$BIN_ID"
  echo "PTHAT_MIN=$PTHAT_MIN"
  echo "PTHAT_MAX=$PTHAT_MAX"
  echo "SEED=$SEED"
  echo "N_EVENTS=$N_EVENTS"
  echo "=================================================="

  "$GENERATOR" \
    "$HEPMC" \
    "$GENERATOR_JSON" \
    "$N_EVENTS" \
    "$SEED" \
    "$PTHAT_MIN" \
    "$PTHAT_MAX" \
    2>&1 | tee \
    "$LOGBASE/${TAG}_pythia.log"

  N_HEPMC=$(
    grep -c '^E ' "$HEPMC" || true
  )

  if [[ "$N_HEPMC" -ne "$N_EVENTS" ]]; then
    echo "ERROR: HepMC event count mismatch"
    exit 3
  fi

  "$DELPHES_DIR/DelphesHepMC3" \
    "$CARD" \
    "$ROOT_FILE" \
    "$HEPMC" \
    2>&1 | tee \
    "$LOGBASE/${TAG}_delphes.log"

  N_ROOT=$(
    python3 -c \
      'import sys, uproot; print(uproot.open(sys.argv[1])["Delphes"].num_entries)' \
      "$ROOT_FILE"
  )

  if [[ "$N_ROOT" -ne "$N_EVENTS" ]]; then
    echo "ERROR: ROOT event count mismatch"
    exit 4
  fi

  python3 \
    scripts/delphes/make_delphes_event_summary.py \
    --input "$ROOT_FILE" \
    --outdir "$OUTBASE/parquet" \
    --sample "$TAG" \
    2>&1 | tee \
    "$LOGBASE/${TAG}_event_summary.log"

  test -s "$EVENT_SUMMARY"

  python3 \
    scripts/delphes/reconstruct_hh4b_candidates_v2.py \
    --input "$ROOT_FILE" \
    --out "$CANDIDATES" \
    --sample "$TAG" \
    2>&1 | tee \
    "$LOGBASE/${TAG}_candidates.log"

  test -s "$CANDIDATES"

  BIN_ID=$((BIN_ID + 1))
done

python3 \
  scripts/production/audit_qcd_importance_pilot.py \
  "$OUTBASE" \
  --local-only

echo
echo "QCD IMPORTANCE PILOT COMPLETE"
echo "Output: $OUTBASE"
echo "Summary: $OUTBASE/qcd_importance_pilot_summary.csv"
