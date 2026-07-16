#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 BIN_ID SHARD_ID N_EVENTS CAMPAIGN CLUSTER INPUT_TARBALL EOS_DIR"
  exit 1
fi

BIN_ID="$1"
SHARD_ID="$2"
N_EVENTS="$3"
CAMPAIGN="$4"
CLUSTER_ID="$5"
INPUT_TARBALL="$6"
EOS_DIR="$7"

[[ "$BIN_ID" =~ ^[0-7]$ ]] || {
  echo "ERROR: BIN_ID must be 0--7"
  exit 1
}

[[ "$SHARD_ID" =~ ^[0-9]+$ ]] || {
  echo "ERROR: bad SHARD_ID"
  exit 1
}

[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: bad N_EVENTS"
  exit 1
}

PTHAT_MIN=(50 75 100 200 300 500 700 1000)
PTHAT_MAX=(75 100 200 300 500 700 1000 0)

PTMIN="${PTHAT_MIN[$BIN_ID]}"
PTMAX="${PTHAT_MAX[$BIN_ID]}"

SEED=$((1200000 + BIN_ID * 10000 + SHARD_ID))

BIN_PADDED=$(printf '%02d' "$BIN_ID")
SHARD_PADDED=$(printf '%03d' "$SHARD_ID")

if [[ "$PTMAX" == "0" ]]; then
  BIN_LABEL="pthat${PTMIN}toInf"
else
  BIN_LABEL="pthat${PTMIN}to${PTMAX}"
fi

TAG="${CAMPAIGN}_bin${BIN_PADDED}_${BIN_LABEL}_shard${SHARD_PADDED}_seed${SEED}"

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

RECEIPT="$SCRATCH/job_receipt.json"
STAGE="initializing"
REMOTE_BUNDLE=""
ADLER32=""

finalize() {
  STATUS=$?

  cat > "$RECEIPT" <<EOF
{
  "campaign": "$CAMPAIGN",
  "bin_id": $BIN_ID,
  "shard_id": $SHARD_ID,
  "pthat_min_GeV": $PTMIN,
  "pthat_max_GeV": $PTMAX,
  "n_events": $N_EVENTS,
  "seed": $SEED,
  "cluster_id": "$CLUSTER_ID",
  "stage": "$STAGE",
  "remote_bundle": "$REMOTE_BUNDLE",
  "adler32": "$ADLER32",
  "exit_status": $STATUS,
  "finished_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

  trap - EXIT
  exit "$STATUS"
}
trap finalize EXIT

test -s "$INPUT_TARBALL" || {
  echo "ERROR: missing payload"
  exit 2
}

PAYLOAD_SHA256=$(sha256sum "$INPUT_TARBALL" | awk '{print $1}')

tar -xzf "$INPUT_TARBALL"

PAYLOAD="$SCRATCH/payload"
REPO="$PAYLOAD/repo"
DELPHES="$PAYLOAD/Delphes"

CARD="$REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
SOURCE="$REPO/scripts/production/generate_pythia8_hardqcd_hepmc3.cc"

EXPECTED_CARD_HASH="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"
CARD_HASH=$(sha256sum "$CARD" | awk '{print $1}')

if [[ "$CARD_HASH" != "$EXPECTED_CARD_HASH" ]]; then
  echo "ERROR: card hash mismatch"
  exit 3
fi

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

test -r "$LCG_SETUP" || {
  echo "ERROR: missing LCG environment"
  exit 3
}

set +u
source "$LCG_SETUP"
set -u

export LD_LIBRARY_PATH="$DELPHES:${LD_LIBRARY_PATH:-}"

for COMMAND in \
  g++ \
  pythia8-config \
  HepMC3-config \
  python3 \
  xrdcp \
  xrdfs
do
  command -v "$COMMAND" >/dev/null || {
    echo "ERROR: missing command: $COMMAND"
    exit 4
  }
done

OUT="$SCRATCH/output"

mkdir -p \
  "$OUT/hepmc" \
  "$OUT/root" \
  "$OUT/parquet" \
  "$OUT/metadata" \
  "$OUT/logs"

GENERATOR="$SCRATCH/generate_hardqcd"

STAGE="compiling"

g++ -O2 -std=c++17 \
  "$SOURCE" \
  -o "$GENERATOR" \
  $(pythia8-config --cxxflags) \
  $(HepMC3-config --cxxflags) \
  $(pythia8-config --libs) \
  $(HepMC3-config --libs)

HEPMC="$OUT/hepmc/${TAG}.hepmc"
ROOT_FILE="$OUT/root/${TAG}_delphes.root"
GENERATOR_JSON="$OUT/metadata/${TAG}_generator.json"
EVENT_SUMMARY="$OUT/parquet/${TAG}_event_summary.parquet"
CANDIDATES="$OUT/parquet/${TAG}_hh4b_candidates_v2.parquet"

STAGE="generating"

"$GENERATOR" \
  "$HEPMC" \
  "$GENERATOR_JSON" \
  "$N_EVENTS" \
  "$SEED" \
  "$PTMIN" \
  "$PTMAX" \
  2>&1 | tee "$OUT/logs/${TAG}_pythia.log"

N_HEPMC=$(grep -c '^E ' "$HEPMC" || true)

if [[ "$N_HEPMC" -ne "$N_EVENTS" ]]; then
  echo "ERROR: HepMC count mismatch"
  exit 5
fi

STAGE="delphes"

"$DELPHES/DelphesHepMC3" \
  "$CARD" \
  "$ROOT_FILE" \
  "$HEPMC" \
  2>&1 | tee "$OUT/logs/${TAG}_delphes.log"

N_ROOT=$(
  python3 -c \
    'import sys,uproot; print(uproot.open(sys.argv[1])["Delphes"].num_entries)' \
    "$ROOT_FILE"
)

if [[ "$N_ROOT" -ne "$N_EVENTS" ]]; then
  echo "ERROR: ROOT count mismatch"
  exit 6
fi

STAGE="reconstruction"

python3 \
  "$REPO/scripts/delphes/make_delphes_event_summary.py" \
  --input "$ROOT_FILE" \
  --outdir "$OUT/parquet" \
  --sample "$TAG" \
  2>&1 | tee "$OUT/logs/${TAG}_event_summary.log"

python3 \
  "$REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py" \
  --input "$ROOT_FILE" \
  --out "$CANDIDATES" \
  --sample "$TAG" \
  2>&1 | tee "$OUT/logs/${TAG}_candidates.log"

test -s "$EVENT_SUMMARY"
test -s "$CANDIDATES"

N_CANDIDATES=$(
  python3 -c \
    'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$CANDIDATES"
)

HEPMC_SHA=$(sha256sum "$HEPMC" | awk '{print $1}')
ROOT_SHA=$(sha256sum "$ROOT_FILE" | awk '{print $1}')

PROVENANCE="$OUT/metadata/${TAG}_provenance.json"

python3 - \
  "$GENERATOR_JSON" \
  "$PROVENANCE" \
  "$TAG" \
  "$PAYLOAD_SHA256" \
  "$CARD_HASH" \
  "$HEPMC_SHA" \
  "$ROOT_SHA" \
  "$N_ROOT" \
  "$N_CANDIDATES" <<'PY_PROVENANCE'
from pathlib import Path
import json
import sys

(
    generator_path,
    output_path,
    tag,
    payload_sha,
    card_sha,
    hepmc_sha,
    root_sha,
    n_root,
    n_candidates,
) = sys.argv[1:]

generator = json.loads(Path(generator_path).read_text())

record = {
    "tag": tag,
    "sample": "Pythia8 inclusive HardQCD",
    "physics_role": "inclusive_QCD_importance_stratum",
    "payload_sha256": payload_sha,
    "delphes_card_sha256": card_sha,
    "hepmc_sha256": hepmc_sha,
    "root_sha256": root_sha,
    "root_events": int(n_root),
    "candidate_rows": int(n_candidates),
    "generator": generator,
}

Path(output_path).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY_PROVENANCE

STAGE="bundling"

BUNDLE="$SCRATCH/${TAG}_bundle.tar.gz"

tar -C "$OUT" -czf "$BUNDLE" .

ADLER32=$(
  python3 -c \
    'import sys,zlib; c=1; f=open(sys.argv[1],"rb"); [None for b in iter(lambda:f.read(8*1024*1024),b"") if not (c:=zlib.adler32(b,c))]; print(f"{c & 0xffffffff:08x}")' \
    "$BUNDLE"
)

# Recalculate with a clearer implementation to avoid relying on
# expression behavior above.
ADLER32=$(
  python3 - "$BUNDLE" <<'PY_ADLER'
import sys
import zlib

checksum = 1
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(
        lambda: handle.read(8 * 1024 * 1024),
        b"",
    ):
        checksum = zlib.adler32(chunk, checksum)

print(f"{checksum & 0xffffffff:08x}")
PY_ADLER
)

STAGE="stageout"

EOS_HOST="root://cmseos.fnal.gov"
REMOTE_BUNDLE="$EOS_DIR/${TAG}_bundle.tar.gz"
REMOTE_TMP="${REMOTE_BUNDLE}.partial.${CLUSTER_ID}.${BIN_ID}.$$"

xrdfs "$EOS_HOST" mkdir -p "$EOS_DIR"

xrdcp -f \
  --nopbar \
  --cksum adler32:print \
  "$BUNDLE" \
  "${EOS_HOST}/${REMOTE_TMP}"

REMOTE_SUM=$(
  xrdfs "$EOS_HOST" query checksum "$REMOTE_TMP" |
  awk '{print tolower($NF)}'
)

if [[ "$REMOTE_SUM" != "$ADLER32" ]]; then
  echo "ERROR: temporary EOS checksum mismatch"
  exit 7
fi

xrdfs "$EOS_HOST" mv \
  "$REMOTE_TMP" \
  "$REMOTE_BUNDLE"

FINAL_SUM=$(
  xrdfs "$EOS_HOST" query checksum "$REMOTE_BUNDLE" |
  awk '{print tolower($NF)}'
)

if [[ "$FINAL_SUM" != "$ADLER32" ]]; then
  echo "ERROR: final EOS checksum mismatch"
  exit 8
fi

STAGE="complete_copied_and_verified"

echo "SUCCESS: $REMOTE_BUNDLE"
echo "ADLER32: $ADLER32"
