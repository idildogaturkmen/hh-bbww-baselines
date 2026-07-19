#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 CAMPAIGN X509_PROXY [PAYLOAD]"
  exit 2
fi

CAMPAIGN="$1"
X509_PROXY="$2"

HH4B_REPO="${HH4B_REPO:-/uscms_data/d3/$USER/repos/hh-bbww-baselines}"
HH4B_STORE="${HH4B_STORE:-/uscms_data/d3/$USER/hh4b_delphes}"

SOURCE_PAYLOAD="${3:-$HH4B_STORE/condor_inputs/ttbar_existing_hepmc_frozen_v2_inputs.tar.gz}"

PLAN="$HH4B_REPO/metadata/delphes/ttbar_readiness_20260718/frozen_v2_remaining_processing_plan.json"

EOS_BASE="${HH4B_EOS_BASE:-/store/user/$USER/hh4b_delphes/run2_13tev/frozen_v2}"

SPLIT_SALT="ttbar-frozen-v2-trainval-v1"

[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]

test -s "$X509_PROXY"
test -s "$SOURCE_PAYLOAD"
test -s "$PLAN"

voms-proxy-info \
  --file "$X509_PROXY" \
  -exists \
  -valid 1:00

if [[ -n "$(git -C "$HH4B_REPO" status --porcelain)" ]]; then
  echo "ERROR: repository must be clean"
  exit 3
fi

mapfile -t PLAN_FIELDS < <(
  python3 - "$PLAN" <<'PY'
from pathlib import Path
import json
import sys

plan = json.loads(
    Path(sys.argv[1]).read_text()
)

assert plan["submission_authorized"] is False

rows = {
    int(row["shard"]): row
    for row in plan["planned_train_validation_shards"]
}

assert set(rows) == {1, 2, 3}

row = rows[1]

assert row["split"] == "train"
assert row["seed"] == 714001
assert row["events"] == 10000

print(row["target_tag"])
print(row["shard"])
print(row["events"])
print(row["seed"])
print(row["split"])
print(row["generator_xsec_pb"])
print(row["source_eos_path"])
print(row["source_hepmc_sha256"])
PY
)

TARGET_TAG="${PLAN_FIELDS[0]}"
SHARD_ID="${PLAN_FIELDS[1]}"
N_EVENTS="${PLAN_FIELDS[2]}"
SEED="${PLAN_FIELDS[3]}"
DATASET_SPLIT="${PLAN_FIELDS[4]}"
GENERATOR_XSEC_PB="${PLAN_FIELDS[5]}"
SOURCE_EOS="${PLAN_FIELDS[6]}"
SOURCE_SHA256="${PLAN_FIELDS[7]}"

export X509_USER_PROXY="$X509_PROXY"

xrdfs root://cmseos.fnal.gov \
  stat "$SOURCE_EOS" >/dev/null

SUBMIT_DIR="$HH4B_STORE/condor_submit/$CAMPAIGN"
LOG_DIR="$HH4B_STORE/condor_logs/$CAMPAIGN"
RETURN_DIR="$HH4B_STORE/condor_return/$CAMPAIGN/receipts"
EOS_DIR="$EOS_BASE/bundles/$CAMPAIGN"

for PATH_TO_CHECK in \
  "$SUBMIT_DIR" \
  "$LOG_DIR" \
  "$(dirname "$RETURN_DIR")"
do
  if [[ -e "$PATH_TO_CHECK" ]]; then
    echo "ERROR: refusing to reuse: $PATH_TO_CHECK"
    exit 4
  fi
done

mkdir -p \
  "$SUBMIT_DIR" \
  "$LOG_DIR" \
  "$RETURN_DIR"

SOURCE_PAYLOAD_SHA256=$(
  sha256sum "$SOURCE_PAYLOAD" |
  awk '{print $1}'
)

PAYLOAD_NAME="${CAMPAIGN}_inputs_${SOURCE_PAYLOAD_SHA256:0:16}.tar.gz"

CAMPAIGN_PAYLOAD="$SUBMIT_DIR/$PAYLOAD_NAME"

cp "$SOURCE_PAYLOAD" "$CAMPAIGN_PAYLOAD"

PAYLOAD_SHA256=$(
  sha256sum "$CAMPAIGN_PAYLOAD" |
  awk '{print $1}'
)

test "$PAYLOAD_SHA256" = "$SOURCE_PAYLOAD_SHA256"

WRAPPER="$HH4B_REPO/scripts/production/run_ttbar_existing_hepmc_bundle.sh"

WRAPPER_SHA256=$(
  sha256sum "$WRAPPER" |
  awk '{print $1}'
)

PAYLOAD_WRAPPER_SHA256=$(
  tar -xOf "$CAMPAIGN_PAYLOAD" payload/manifest.txt |
  awk -F= '$1 == "worker_wrapper_sha256" {print $2}'
)

PAYLOAD_GIT_HEAD=$(
  tar -xOf "$CAMPAIGN_PAYLOAD" payload/manifest.txt |
  awk -F= '$1 == "git_head" {print $2}'
)

GIT_HEAD=$(
  git -C "$HH4B_REPO" rev-parse HEAD
)

test "$WRAPPER_SHA256" = "$PAYLOAD_WRAPPER_SHA256"
test "$GIT_HEAD" = "$PAYLOAD_GIT_HEAD"

chmod 0444 "$CAMPAIGN_PAYLOAD"

CARD_SHA256=$(
  sha256sum "$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl" |
  awk '{print $1}'
)

SUBMIT_FILE="$SUBMIT_DIR/$CAMPAIGN.sub"
CAMPAIGN_MANIFEST="$SUBMIT_DIR/${CAMPAIGN}_manifest.csv"

cat > "$SUBMIT_FILE" <<EOF
universe = vanilla

executable = $WRAPPER

arguments = $CAMPAIGN $TARGET_TAG $SHARD_ID $N_EVENTS $SEED $DATASET_SPLIT $SPLIT_SALT $GENERATOR_XSEC_PB $SOURCE_EOS $SOURCE_SHA256 $PAYLOAD_NAME $EOS_DIR $PAYLOAD_SHA256 \$(Cluster)

output = $LOG_DIR/${CAMPAIGN}_\$(Cluster).out
error = $LOG_DIR/${CAMPAIGN}_\$(Cluster).err
log = $LOG_DIR/${CAMPAIGN}_\$(Cluster).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = $CAMPAIGN_PAYLOAD

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = $RETURN_DIR/${CAMPAIGN}_\$(Cluster)_receipt.json"

request_cpus = 1
request_memory = 4GB
request_disk = 15GB

getenv = False
environment = "LC_ALL=C LANG=C"

use_x509userproxy = True
x509userproxy = $X509_PROXY

+JobBatchName = "$CAMPAIGN"
+DesiredOS = "EL9"

on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)

queue 1
EOF

cat > "$CAMPAIGN_MANIFEST" <<EOF
campaign,target_tag,shard_id,n_events,seed,dataset_split,split_salt,generator_xsec_pb,source_eos,source_sha256,payload_sha256,wrapper_sha256,card_sha256,git_head,eos_directory
$CAMPAIGN,$TARGET_TAG,$SHARD_ID,$N_EVENTS,$SEED,$DATASET_SPLIT,$SPLIT_SALT,$GENERATOR_XSEC_PB,$SOURCE_EOS,$SOURCE_SHA256,$PAYLOAD_SHA256,$WRAPPER_SHA256,$CARD_SHA256,$GIT_HEAD,$EOS_DIR
EOF

echo "TTBAR_SHARD1_SUBMIT_DESCRIPTION_PREPARED"
echo "Submit file: $SUBMIT_FILE"
echo "Manifest: $CAMPAIGN_MANIFEST"
echo "Payload: $CAMPAIGN_PAYLOAD"
echo "Payload SHA-256: $PAYLOAD_SHA256"
echo "EOS directory: $EOS_DIR"
echo "Jobs: 1"
