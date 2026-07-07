#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 SHARD_ID N_EVENTS CAMPAIGN CLUSTER_ID [INPUT_TARBALL]"
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  usage
  exit 1
fi

SHARD_ID="$1"
N_EVENTS="$2"
CAMPAIGN="$3"
CLUSTER_ID="$4"
INPUT_TARBALL="${5:-hh4b_vbf_condor_inputs.tar.gz}"

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

if [[ "$(pwd -P)" != /srv* && "${ALLOW_NON_SRV_FOR_TESTING:-0}" != "1" ]]; then
  echo "ERROR: worker job must run under /srv or Condor scratch; PWD=$(pwd -P)"
  exit 2
fi

OUTDIR="$SCRATCH/output"
mkdir -p "$OUTDIR"/{root,parquet,metadata,logs}

finalize() {
  status=$?
  final_status_name="${OUTPUT_TAG:-HH4b_${CAMPAIGN}_cluster_${CLUSTER_ID}_shard_${SHARD_ID}_${N_EVENTS}}_final_status.txt"
  {
    echo "exit_status=$status"
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$OUTDIR/metadata/$final_status_name"
  tar -C "$OUTDIR" -czf "$SCRATCH/shard_output.tar.gz" . || true
  exit "$status"
}
trap finalize EXIT

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd -P)"
echo "SCRATCH=$SCRATCH"
echo "SHARD_ID=$SHARD_ID"
echo "N_EVENTS=$N_EVENTS"
echo "CAMPAIGN=$CAMPAIGN"
echo "CLUSTER_ID=$CLUSTER_ID"
echo "INPUT_TARBALL=$INPUT_TARBALL"
echo "Sample label: VBF HH -> 4b Delphes validation"

for forbidden in /uscms_data /uscms; do
  if [ -e "$forbidden" ]; then
    echo "WARNING: $forbidden exists but this transfer workflow will not use it"
  else
    echo "OK: $forbidden is not visible"
  fi
done

if [ ! -s "$INPUT_TARBALL" ]; then
  echo "ERROR: missing input tarball in scratch: $INPUT_TARBALL"
  exit 3
fi

tar -xzf "$INPUT_TARBALL"

PAYLOAD="$SCRATCH/payload"
HH4B_REPO="$PAYLOAD/repo"
HH4B_STORE="$SCRATCH/store"
DELPHES_DIR="$PAYLOAD/Delphes"
MG5_TEMPLATE="$PAYLOAD/mg5_template"

export HH4B_REPO HH4B_STORE DELPHES_DIR
mkdir -p "$HH4B_STORE" "$SCRATCH/work"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
if [ ! -r "$LCG_SETUP" ]; then
  echo "ERROR: cannot read LCG setup file: $LCG_SETUP"
  exit 4
fi

set +u
# shellcheck disable=SC1090
source "$LCG_SETUP"
set -u

export LD_LIBRARY_PATH="$DELPHES_DIR:${LD_LIBRARY_PATH:-}"

for cmd in python3 root-config pythia8-config HepMC3-config mg5_aMC g++ tar gzip; do
  if ! command -v "$cmd"; then
    echo "ERROR: required command is missing after LCG setup: $cmd"
    exit 5
  fi
  echo "COMMAND: $cmd -> $(command -v "$cmd")"
done

python3 - <<'PY'
import awkward  # noqa: F401
import pandas  # noqa: F401
import pyarrow  # noqa: F401
import uproot  # noqa: F401
print("Python stack OK: awkward, pandas, pyarrow, uproot")
PY

if [ ! -x "$DELPHES_DIR/DelphesHepMC3" ]; then
  echo "ERROR: missing transferred DelphesHepMC3: $DELPHES_DIR/DelphesHepMC3"
  exit 6
fi

RUN_NAME="run_shard_${SHARD_ID}_${N_EVENTS}"
OUTPUT_TAG="HH4b_${CAMPAIGN}_cluster_${CLUSTER_ID}_shard_${SHARD_ID}_${N_EVENTS}"
WORK_PROC="$SCRATCH/work/${RUN_NAME}"
SEED=$((50000 + SHARD_ID))

echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_TAG=$OUTPUT_TAG"
echo "SEED=$SEED"

CONVERTER="$SCRATCH/lhe_to_hepmc3"
echo "Compiling LHE -> HepMC3 converter..."
g++ -O2 -std=c++17 \
  "$HH4B_REPO/scripts/delphes/lhe_to_hepmc3.cc" \
  -o "$CONVERTER" \
  $(pythia8-config --cxxflags) \
  $(HepMC3-config --cxxflags) \
  $(pythia8-config --libs) \
  $(HepMC3-config --libs) \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_compile_lhe_to_hepmc3.log"

cp -a "$MG5_TEMPLATE" "$WORK_PROC"
cd "$WORK_PROC"
mkdir -p Events HTML

python3 - <<PY
from pathlib import Path

path = Path("Cards/run_card.dat")
if not path.is_file():
    raise SystemExit(f"ERROR: missing {path}")

lines = []
for line in path.read_text().splitlines():
    if "= nevents" in line:
        line = "  ${N_EVENTS} = nevents ! Number of unweighted events requested"
    elif "= iseed" in line:
        line = "  ${SEED} = iseed ! rnd seed"
    elif "= ebeam1" in line:
        line = "  6500.0 = ebeam1 ! beam 1 total energy in GeV"
    elif "= ebeam2" in line:
        line = "  6500.0 = ebeam2 ! beam 2 total energy in GeV"
    elif "= cut_decays" in line:
        line = "  False = cut_decays ! Cut decay products"
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
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_mg5.log"
MG5_STATUS=${PIPESTATUS[0]}
set -e
echo "MG5 exit status: $MG5_STATUS"

LHE_GZ="$WORK_PROC/Events/$RUN_NAME/unweighted_events.lhe.gz"
if [ ! -s "$LHE_GZ" ]; then
  echo "ERROR: missing LHE file after MG5: $LHE_GZ"
  exit 7
fi

N_FOUND=$(gzip -cd "$LHE_GZ" | grep -c "<event>")
echo "LHE events found: $N_FOUND"
if [ "$N_FOUND" -lt "$N_EVENTS" ]; then
  echo "ERROR: expected at least $N_EVENTS LHE events, found $N_FOUND"
  exit 8
fi

LHE="$SCRATCH/work/${OUTPUT_TAG}.lhe"
HEPMC="$SCRATCH/work/${OUTPUT_TAG}.hepmc"
ROOT_OUT="$OUTDIR/root/${OUTPUT_TAG}_pythia8_delphes.root"

gzip -cd "$LHE_GZ" > "$LHE"

"$CONVERTER" "$LHE" "$HEPMC" "$N_EVENTS" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_lhe_to_hepmc3.log"

"$DELPHES_DIR/DelphesHepMC3" \
  "$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc.tcl" \
  "$ROOT_OUT" \
  "$HEPMC" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_pythia8_delphes.log"

python3 "$HH4B_REPO/scripts/delphes/inspect_delphes_root.py" "$ROOT_OUT" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_inspect_root.log"

python3 "$HH4B_REPO/scripts/delphes/diagnose_delphes_hh4b_smoke.py" \
  --input "$ROOT_OUT" \
  --out "$OUTDIR/metadata/${OUTPUT_TAG}_diagnostic.csv" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_diagnostic.log"

python3 "$HH4B_REPO/scripts/delphes/make_delphes_event_summary.py" \
  --input "$ROOT_OUT" \
  --outdir "$OUTDIR/parquet" \
  --sample "$OUTPUT_TAG" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_event_summary.log"

python3 "$HH4B_REPO/scripts/delphes/reconstruct_hh4b_candidates.py" \
  --input "$ROOT_OUT" \
  --out "$OUTDIR/parquet/${OUTPUT_TAG}_hh4b_candidates.parquet" \
  --sample "$OUTPUT_TAG" \
  2>&1 | tee "$OUTDIR/logs/${OUTPUT_TAG}_hh4b_candidates.log"

python3 - <<PY
import json
from pathlib import Path

metadata = {
    "sample": "VBF HH -> 4b Delphes validation",
    "campaign": "${CAMPAIGN}",
    "cluster_id": "${CLUSTER_ID}",
    "shard_id": int("${SHARD_ID}"),
    "n_events_requested": int("${N_EVENTS}"),
    "seed": int("${SEED}"),
    "mg5_status": int("${MG5_STATUS}"),
    "lhe_events": int("${N_FOUND}"),
    "output_tag": "${OUTPUT_TAG}",
}
Path("${OUTDIR}/metadata/${OUTPUT_TAG}_metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\\n"
)
PY

echo "DONE shard $SHARD_ID"
echo "ROOT: $ROOT_OUT"
echo "EVENT_SUMMARY: $OUTDIR/parquet/${OUTPUT_TAG}_event_summary.parquet"
echo "HH4B_CANDIDATES: $OUTDIR/parquet/${OUTPUT_TAG}_hh4b_candidates.parquet"
echo "DIAGNOSTIC: $OUTDIR/metadata/${OUTPUT_TAG}_diagnostic.csv"
