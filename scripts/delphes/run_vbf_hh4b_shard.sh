#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 SHARD_ID N_EVENTS"
  exit 1
fi

SHARD_ID="$1"
N_EVENTS="$2"

export HH4B_REPO="/uscms_data/d3/${USER}/repos/hh-bbww-baselines"
source "$HH4B_REPO/scripts/delphes/setup_lpc_delphes_env.sh"

TEMPLATE_PROC="$HH4B_STORE/mg5/HH4b_smoke_vbf"
RUN_NAME="run_shard_${SHARD_ID}_${N_EVENTS}"
WORK_PROC="$HH4B_STORE/mg5_shards/${RUN_NAME}"
OUTPUT_TAG="HH4b_smoke_vbf_${RUN_NAME}_pythia8"

mkdir -p "$HH4B_STORE/mg5_shards" "$HH4B_STORE/logs"

echo "Shard: $SHARD_ID"
echo "Events: $N_EVENTS"
echo "Run name: $RUN_NAME"
echo "Work process: $WORK_PROC"
echo "Output tag: $OUTPUT_TAG"

rm -rf "$WORK_PROC"

rsync -a \
  --exclude "Events" \
  --exclude "HTML" \
  --exclude "run_*_debug.log" \
  "$TEMPLATE_PROC/" "$WORK_PROC/"

cd "$WORK_PROC"

SEED=$((50000 + SHARD_ID))

python3 - <<PY
from pathlib import Path

path = Path("Cards/run_card.dat")
lines = []

for line in path.read_text().splitlines():
    if "= nevents" in line:
        line = "  ${N_EVENTS} = nevents ! Number of unweighted events requested"
    elif "= iseed" in line:
        line = "  ${SEED} = iseed ! rnd seed"
    elif "= ptj" in line:
        line = "  20.0 = ptj ! minimum pt for the jets"
    elif "= etaj" in line:
        line = "  5.0 = etaj ! max rap for the jets"
    elif "= drjj" in line:
        line = "  0.4 = drjj ! min distance between jets"
    lines.append(line)

path.write_text("\\n".join(lines) + "\\n")
PY

echo "Generating MG5 events..."
set +e
./bin/generate_events "$RUN_NAME" -f \
  2>&1 | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_mg5.log"
MG5_STATUS=$?
set -e

echo "MG5 exit status: $MG5_STATUS"

LHE_GZ="$WORK_PROC/Events/$RUN_NAME/unweighted_events.lhe.gz"

if [ ! -s "$LHE_GZ" ]; then
  echo "ERROR: Missing LHE file: $LHE_GZ"
  exit 2
fi

N_FOUND=$(zgrep -c "<event>" "$LHE_GZ")
echo "LHE events found: $N_FOUND"

if [ "$N_FOUND" -lt "$N_EVENTS" ]; then
  echo "ERROR: Expected $N_EVENTS events, found $N_FOUND"
  exit 3
fi

"$HH4B_REPO/scripts/delphes/run_existing_lhe_pythia_delphes.sh" \
  HH4b_vbf \
  "$LHE_GZ" \
  "$N_EVENTS" \
  "$OUTPUT_TAG"

python3 "$HH4B_REPO/scripts/delphes/reconstruct_hh4b_candidates.py" \
  --input "$HH4B_STORE/root/${OUTPUT_TAG}_pythia8_delphes.root" \
  --out "$HH4B_STORE/parquet/${OUTPUT_TAG}_hh4b_candidates.parquet" \
  --sample "$OUTPUT_TAG"

echo "DONE shard $SHARD_ID"
