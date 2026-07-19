#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 14 ]]; then
  echo "Usage:"
  echo "  $0 CAMPAIGN TARGET_TAG SHARD_ID N_EVENTS SEED DATASET_SPLIT SPLIT_SALT GENERATOR_XSEC_PB SOURCE_EOS SOURCE_SHA256 INPUT_TARBALL EOS_DIR EXPECTED_PAYLOAD_SHA256 CLUSTER_ID"
  exit 2
fi

CAMPAIGN="$1"
TARGET_TAG="$2"
SHARD_ID="$3"
N_EVENTS="$4"
SEED="$5"
DATASET_SPLIT="$6"
SPLIT_SALT="$7"
GENERATOR_XSEC_PB="$8"
SOURCE_EOS="$9"
SOURCE_SHA256="${10}"
INPUT_TARBALL="${11}"
EOS_DIR="${12}"
EXPECTED_PAYLOAD_SHA256="${13}"
CLUSTER_ID="${14}"

[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
[[ "$TARGET_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
[[ "$SHARD_ID" =~ ^[0-9]+$ ]]
[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]]
[[ "$SEED" =~ ^[1-9][0-9]*$ ]]
[[ "$DATASET_SPLIT" =~ ^(train|validation)$ ]]
[[ "$SPLIT_SALT" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$SOURCE_EOS" == /store/user/* ]]
[[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_PAYLOAD_SHA256" =~ ^[0-9a-f]{64}$ ]]

python3 - "$GENERATOR_XSEC_PB" <<'PY'
import sys

value = float(sys.argv[1])

if not value > 0:
    raise SystemExit("ERROR: generator cross section must be positive")
PY

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

RECEIPT="$SCRATCH/job_receipt.json"
STAGE="initializing"
REMOTE_BUNDLE=""
ADLER32=""
PAYLOAD_SHA256=""
CARD_SHA256=""
ROOT_SHA256=""
EVENT_SUMMARY_SHA256=""
CANDIDATE_SHA256=""
N_ROOT=0
N_CANDIDATES=0
PAYLOAD_GIT_HEAD=""

finalize() {
  STATUS=$?

  cat > "$RECEIPT" <<EOF
{
  "campaign": "$CAMPAIGN",
  "target_tag": "$TARGET_TAG",
  "shard_id": $SHARD_ID,
  "n_events": $N_EVENTS,
  "seed": $SEED,
  "dataset_split": "$DATASET_SPLIT",
  "split_assignment_unit": "whole_shard",
  "split_salt": "$SPLIT_SALT",
  "generator_xsec_pb": $GENERATOR_XSEC_PB,
  "source_eos": "$SOURCE_EOS",
  "source_sha256": "$SOURCE_SHA256",
  "cluster_id": "$CLUSTER_ID",
  "stage": "$STAGE",
  "remote_bundle": "$REMOTE_BUNDLE",
  "adler32": "$ADLER32",
  "payload_sha256": "$PAYLOAD_SHA256",
  "expected_payload_sha256": "$EXPECTED_PAYLOAD_SHA256",
  "payload_git_head": "$PAYLOAD_GIT_HEAD",
  "delphes_card_sha256": "$CARD_SHA256",
  "root_sha256": "$ROOT_SHA256",
  "event_summary_sha256": "$EVENT_SUMMARY_SHA256",
  "candidate_sha256": "$CANDIDATE_SHA256",
  "root_events": $N_ROOT,
  "candidate_rows": $N_CANDIDATES,
  "exit_status": $STATUS,
  "finished_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

  trap - EXIT
  exit "$STATUS"
}

trap finalize EXIT

test -s "$INPUT_TARBALL"

PAYLOAD_SHA256=$(
  sha256sum "$INPUT_TARBALL" |
  awk '{print $1}'
)

if [[ "$PAYLOAD_SHA256" != "$EXPECTED_PAYLOAD_SHA256" ]]; then
  echo "ERROR: payload SHA-256 mismatch"
  exit 3
fi

tar -xzf "$INPUT_TARBALL"

LCG_SETUP="${HH4B_LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh}"

test -r "$LCG_SETUP"

set +u
source "$LCG_SETUP"
set -u

unset SOURCE

PAYLOAD="$SCRATCH/payload"
REPO="$PAYLOAD/repo"
DELPHES="$PAYLOAD/Delphes"
MANIFEST="$PAYLOAD/manifest.txt"

CARD="$REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"

EVENT_SCRIPT="$REPO/scripts/delphes/make_delphes_event_summary.py"
CANDIDATE_SCRIPT="$REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
PARQUET_WRITER="$REPO/scripts/delphes/write_parquet_from_pickle.py"

EXPECTED_CARD_SHA256="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

CARD_SHA256=$(
  sha256sum "$CARD" |
  awk '{print $1}'
)

if [[ "$CARD_SHA256" != "$EXPECTED_CARD_SHA256" ]]; then
  echo "ERROR: frozen-v2 card SHA-256 mismatch"
  exit 4
fi

WRAPPER_SHA256=$(
  sha256sum "$0" |
  awk '{print $1}'
)

EXPECTED_WRAPPER_SHA256=$(
  awk -F= '$1 == "worker_wrapper_sha256" {print $2}' "$MANIFEST"
)

EXPECTED_EVENT_SHA256=$(
  awk -F= '$1 == "event_summary_script_sha256" {print $2}' "$MANIFEST"
)

EXPECTED_CANDIDATE_SHA256=$(
  awk -F= '$1 == "candidate_script_sha256" {print $2}' "$MANIFEST"
)

EXPECTED_PARQUET_WRITER_SHA256=$(
  awk -F= '$1 == "parquet_writer_sha256" {print $2}' "$MANIFEST"
)

PAYLOAD_GIT_HEAD=$(
  awk -F= '$1 == "git_head" {print $2}' "$MANIFEST"
)

test "$WRAPPER_SHA256" = "$EXPECTED_WRAPPER_SHA256"

test "$(
  sha256sum "$EVENT_SCRIPT" |
  awk '{print $1}'
)" = "$EXPECTED_EVENT_SHA256"

test "$(
  sha256sum "$CANDIDATE_SCRIPT" |
  awk '{print $1}'
)" = "$EXPECTED_CANDIDATE_SHA256"

test "$(
  sha256sum "$PARQUET_WRITER" |
  awk '{print $1}'
)" = "$EXPECTED_PARQUET_WRITER_SHA256"

export LD_LIBRARY_PATH="$DELPHES:${LD_LIBRARY_PATH:-}"

for COMMAND in \
  openssl \
  python3 \
  sha256sum \
  tar \
  xrdcp \
  xrdfs
do
  command -v "$COMMAND" >/dev/null
done

test -r "${X509_USER_PROXY:-}"

openssl x509 \
  -in "$X509_USER_PROXY" \
  -noout \
  -checkend 3600 >/dev/null

OUT="$SCRATCH/output"

mkdir -p \
  "$OUT/input" \
  "$OUT/root" \
  "$OUT/parquet" \
  "$OUT/metadata" \
  "$OUT/logs"

SOURCE_BASENAME=$(basename "$SOURCE_EOS")
HEPMC="$OUT/input/$SOURCE_BASENAME"

ROOT_FILE="$OUT/root/${TARGET_TAG}_delphes.root"
EVENT_SUMMARY="$OUT/parquet/${TARGET_TAG}_event_summary.parquet"
CANDIDATES="$OUT/parquet/${TARGET_TAG}_hh4b_candidates_v2.parquet"
PROVENANCE="$OUT/metadata/${TARGET_TAG}_provenance.json"

STAGE="copying_input"

xrdcp \
  -f \
  --nopbar \
  "root://cmseos.fnal.gov/${SOURCE_EOS}" \
  "$HEPMC"

OBSERVED_SOURCE_SHA256=$(
  sha256sum "$HEPMC" |
  awk '{print $1}'
)

if [[ "$OBSERVED_SOURCE_SHA256" != "$SOURCE_SHA256" ]]; then
  echo "ERROR: source HepMC SHA-256 mismatch"
  exit 5
fi

N_HEPMC=$(grep -c '^E ' "$HEPMC" || true)

if [[ "$N_HEPMC" -ne "$N_EVENTS" ]]; then
  echo "ERROR: HepMC event-count mismatch: $N_HEPMC != $N_EVENTS"
  exit 6
fi

STAGE="delphes"

"$DELPHES/DelphesHepMC3" \
  "$CARD" \
  "$ROOT_FILE" \
  "$HEPMC" \
  2>&1 |
  tee "$OUT/logs/${TARGET_TAG}_delphes.log"

N_ROOT=$(
  python3 -c \
    'import sys, uproot; print(uproot.open(sys.argv[1])["Delphes"].num_entries)' \
    "$ROOT_FILE"
)

if [[ "$N_ROOT" -ne "$N_EVENTS" ]]; then
  echo "ERROR: ROOT event-count mismatch: $N_ROOT != $N_EVENTS"
  exit 7
fi

STAGE="event_summary"

python3 \
  "$EVENT_SCRIPT" \
  --input "$ROOT_FILE" \
  --outdir "$OUT/parquet" \
  --sample "$TARGET_TAG" \
  2>&1 |
  tee "$OUT/logs/${TARGET_TAG}_event_summary.log"

test -s "$EVENT_SUMMARY"

EVENT_ROWS=$(
  python3 -c \
    'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$EVENT_SUMMARY"
)

if [[ "$EVENT_ROWS" -ne "$N_EVENTS" ]]; then
  echo "ERROR: event-summary row-count mismatch"
  exit 8
fi

STAGE="candidate_reconstruction"

set +e

python3 \
  "$CANDIDATE_SCRIPT" \
  --input "$ROOT_FILE" \
  --out "$CANDIDATES" \
  --sample "$TARGET_TAG" \
  2>&1 |
  tee "$OUT/logs/${TARGET_TAG}_candidates.log"

RECONSTRUCTION_STATUS=${PIPESTATUS[0]}

set -e

if [[ "$RECONSTRUCTION_STATUS" -ne 0 ]]; then
  if [[ "$RECONSTRUCTION_STATUS" -ne 134 ]]; then
    echo "ERROR: candidate reconstruction failed: $RECONSTRUCTION_STATUS"
    exit "$RECONSTRUCTION_STATUS"
  fi

  python3 - "$EVENT_SUMMARY" "$CANDIDATES" "$N_EVENTS" <<'PY'
from pathlib import Path
import os
import sys

import pandas as pd

event_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
expected_events = int(sys.argv[3])

if not event_path.is_file() or event_path.stat().st_size == 0:
    raise SystemExit("ERROR: missing event summary")

if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
    raise SystemExit("ERROR: missing candidate Parquet")

events = pd.read_parquet(event_path)
candidates = pd.read_parquet(candidate_path)

if len(events) != expected_events:
    raise SystemExit("ERROR: event summary has wrong row count")

if "n_bjet_pt30_eta25" not in events.columns:
    raise SystemExit("ERROR: event summary lacks n_bjet_pt30_eta25")

if int((events["n_bjet_pt30_eta25"] >= 4).sum()) != 0:
    raise SystemExit("ERROR: exit 134 occurred despite >=4b events")

if len(candidates) != 0:
    raise SystemExit("ERROR: exit 134 occurred with nonempty candidates")

os.write(
    1,
    b"RECOVERED_VALID_EMPTY_CANDIDATE_TEARDOWN_EXIT_134\n",
)
os._exit(0)
PY
fi

test -s "$CANDIDATES"

N_CANDIDATES=$(
  python3 -c \
    'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$CANDIDATES"
)

ROOT_SHA256=$(
  sha256sum "$ROOT_FILE" |
  awk '{print $1}'
)

EVENT_SUMMARY_SHA256=$(
  sha256sum "$EVENT_SUMMARY" |
  awk '{print $1}'
)

CANDIDATE_SHA256=$(
  sha256sum "$CANDIDATES" |
  awk '{print $1}'
)

python3 - \
  "$PROVENANCE" \
  "$CAMPAIGN" \
  "$TARGET_TAG" \
  "$SHARD_ID" \
  "$N_EVENTS" \
  "$SEED" \
  "$DATASET_SPLIT" \
  "$SPLIT_SALT" \
  "$GENERATOR_XSEC_PB" \
  "$SOURCE_EOS" \
  "$SOURCE_SHA256" \
  "$PAYLOAD_SHA256" \
  "$PAYLOAD_GIT_HEAD" \
  "$CARD_SHA256" \
  "$ROOT_SHA256" \
  "$EVENT_SUMMARY_SHA256" \
  "$CANDIDATE_SHA256" \
  "$N_ROOT" \
  "$N_CANDIDATES" <<'PY'
from pathlib import Path
import json
import sys

(
    output_path,
    campaign,
    target_tag,
    shard_id,
    n_events,
    seed,
    dataset_split,
    split_salt,
    generator_xsec_pb,
    source_eos,
    source_sha256,
    payload_sha256,
    payload_git_head,
    card_sha256,
    root_sha256,
    event_summary_sha256,
    candidate_sha256,
    root_events,
    candidate_rows,
) = sys.argv[1:]

record = {
    "schema_version": 1,
    "campaign": campaign,
    "target_tag": target_tag,
    "sample": "inclusive_ttbar_existing_hepmc",
    "physics_role": "inclusive_ttbar_background",
    "shard_id": int(shard_id),
    "n_events": int(n_events),
    "seed": int(seed),
    "dataset_split": dataset_split,
    "split_assignment_unit": "whole_shard",
    "split_salt": split_salt,
    "generator_xsec_pb": float(generator_xsec_pb),
    "normalization_status": (
        "generator_level_provisional_not_final_higher_order"
    ),
    "source_eos": source_eos,
    "source_sha256": source_sha256,
    "payload_sha256": payload_sha256,
    "payload_git_head": payload_git_head,
    "delphes_card_sha256": card_sha256,
    "root_sha256": root_sha256,
    "event_summary_sha256": event_summary_sha256,
    "candidate_sha256": candidate_sha256,
    "root_events": int(root_events),
    "candidate_rows": int(candidate_rows),
}

Path(output_path).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY

rm -f "$HEPMC"
rmdir "$OUT/input"

STAGE="bundling"

BUNDLE="$SCRATCH/${TARGET_TAG}_bundle.tar.gz"

tar \
  -C "$OUT" \
  -czf "$BUNDLE" \
  .

ADLER32=$(
  python3 - "$BUNDLE" <<'PY'
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
PY
)

STAGE="stageout"

EOS_HOST="root://cmseos.fnal.gov"

REMOTE_BUNDLE="$EOS_DIR/${TARGET_TAG}_bundle.tar.gz"
REMOTE_TMP="${REMOTE_BUNDLE}.partial.${CLUSTER_ID}.$$"

xrdfs "$EOS_HOST" mkdir -p "$EOS_DIR"

xrdcp \
  -f \
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
  exit 9
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
  exit 10
fi

STAGE="complete_copied_and_verified"

echo "SUCCESS: $REMOTE_BUNDLE"
echo "ADLER32: $ADLER32"
echo "ROOT_EVENTS: $N_ROOT"
echo "CANDIDATE_ROWS: $N_CANDIDATES"
