#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "Usage: $0 CAMPAIGN N_EVENTS SHARD_ID BIN_SPEC X509_PROXY [PAYLOAD]" >&2
  echo "  BIN_SPEC is all or a comma-separated list of bin IDs 0--7" >&2
  exit 2
fi

CAMPAIGN="$1"
N_EVENTS="$2"
SHARD_ID="$3"
BIN_SPEC="$4"
X509_PROXY="$5"

HH4B_REPO="${HH4B_REPO:-/uscms_data/d3/$USER/repos/hh-bbww-baselines}"
HH4B_STORE="${HH4B_STORE:-/uscms_data/d3/$USER/hh4b_delphes}"
SOURCE_PAYLOAD="${6:-$HH4B_STORE/condor_inputs/qcd_hardqcd_importance_inputs.tar.gz}"
EOS_BASE="${HH4B_EOS_BASE:-/store/user/$USER/hh4b_delphes/run2_13tev/frozen_v2}"

[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || {
  echo "ERROR: invalid campaign name: $CAMPAIGN" >&2
  exit 3
}

[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: N_EVENTS must be positive" >&2
  exit 3
}

[[ "$SHARD_ID" =~ ^[0-9]+$ ]] || {
  echo "ERROR: SHARD_ID must be nonnegative" >&2
  exit 3
}

test -s "$SOURCE_PAYLOAD" || {
  echo "ERROR: missing payload: $SOURCE_PAYLOAD" >&2
  exit 3
}

test -s "$X509_PROXY" || {
  echo "ERROR: missing X.509 proxy: $X509_PROXY" >&2
  exit 3
}

if command -v voms-proxy-info >/dev/null 2>&1; then
  voms-proxy-info -file "$X509_PROXY" -exists -valid 1:00 || {
    echo "ERROR: proxy has less than one hour remaining" >&2
    exit 3
  }
fi

if [[ -n "$(git -C "$HH4B_REPO" status --porcelain)" ]]; then
  echo "ERROR: commit or remove repository changes before preparing production" >&2
  exit 4
fi

if [[ "$BIN_SPEC" == "all" ]]; then
  BIN_IDS=(0 1 2 3 4 5 6 7)
else
  IFS=, read -r -a BIN_IDS <<< "$BIN_SPEC"
fi

declare -A SEEN_BINS=()

for BIN_ID in "${BIN_IDS[@]}"; do
  [[ "$BIN_ID" =~ ^[0-7]$ ]] || {
    echo "ERROR: invalid bin ID: $BIN_ID" >&2
    exit 3
  }

  if [[ -n "${SEEN_BINS[$BIN_ID]:-}" ]]; then
    echo "ERROR: duplicate bin ID: $BIN_ID" >&2
    exit 3
  fi

  SEEN_BINS[$BIN_ID]=1
done

SUBMIT_DIR="$HH4B_STORE/condor_submit/$CAMPAIGN"
LOG_DIR="$HH4B_STORE/condor_logs/$CAMPAIGN"
RETURN_DIR="$HH4B_STORE/condor_return/$CAMPAIGN/receipts"
EOS_DIR="$EOS_BASE/bundles/$CAMPAIGN"

for CAMPAIGN_PATH in "$SUBMIT_DIR" "$LOG_DIR" "$(dirname "$RETURN_DIR")"; do
  if [[ -e "$CAMPAIGN_PATH" ]]; then
    echo "ERROR: refusing to reuse existing campaign path: $CAMPAIGN_PATH" >&2
    exit 5
  fi
done

mkdir -p "$SUBMIT_DIR" "$LOG_DIR" "$RETURN_DIR"

SOURCE_PAYLOAD_SHA256=$(sha256sum "$SOURCE_PAYLOAD" | awk '{print $1}')
PAYLOAD_NAME="${CAMPAIGN}_inputs_${SOURCE_PAYLOAD_SHA256:0:16}.tar.gz"
CAMPAIGN_PAYLOAD="$SUBMIT_DIR/$PAYLOAD_NAME"

cp "$SOURCE_PAYLOAD" "$CAMPAIGN_PAYLOAD"
PAYLOAD_SHA256=$(sha256sum "$CAMPAIGN_PAYLOAD" | awk '{print $1}')

if [[ "$PAYLOAD_SHA256" != "$SOURCE_PAYLOAD_SHA256" ]]; then
  echo "ERROR: campaign payload copy hash mismatch" >&2
  exit 6
fi

WRAPPER="$HH4B_REPO/scripts/production/run_qcd_importance_bundle.sh"
WRAPPER_SHA256=$(sha256sum "$WRAPPER" | awk '{print $1}')
PAYLOAD_WRAPPER_SHA256=$(
  tar -xOf "$CAMPAIGN_PAYLOAD" payload/manifest.txt |
  awk -F= '$1 == "worker_wrapper_sha256" {print $2}'
)
PAYLOAD_GIT_HEAD=$(
  tar -xOf "$CAMPAIGN_PAYLOAD" payload/manifest.txt |
  awk -F= '$1 == "git_head" {print $2}'
)
GIT_HEAD=$(git -C "$HH4B_REPO" rev-parse HEAD)

if [[ -z "$PAYLOAD_WRAPPER_SHA256" || "$WRAPPER_SHA256" != "$PAYLOAD_WRAPPER_SHA256" ]]; then
  echo "ERROR: payload and worker wrapper hashes do not agree" >&2
  exit 6
fi

if [[ -z "$PAYLOAD_GIT_HEAD" || "$PAYLOAD_GIT_HEAD" != "$GIT_HEAD" ]]; then
  echo "ERROR: payload git head does not match the current checkout" >&2
  exit 6
fi

chmod 0444 "$CAMPAIGN_PAYLOAD"

CARD_SHA256=$(sha256sum "$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl" | awk '{print $1}')
SUBMIT_FILE="$SUBMIT_DIR/$CAMPAIGN.sub"
CAMPAIGN_MANIFEST="$SUBMIT_DIR/${CAMPAIGN}_manifest.csv"

cat > "$SUBMIT_FILE" <<EOF
universe = vanilla
executable = $WRAPPER
arguments = \$(bin_id) $SHARD_ID $N_EVENTS $CAMPAIGN \$(Cluster) $PAYLOAD_NAME $EOS_DIR $PAYLOAD_SHA256

output = $LOG_DIR/${CAMPAIGN}_\$(Cluster)_bin\$(bin_id).out
error = $LOG_DIR/${CAMPAIGN}_\$(Cluster)_bin\$(bin_id).err
log = $LOG_DIR/${CAMPAIGN}_\$(Cluster).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = $CAMPAIGN_PAYLOAD

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = $RETURN_DIR/${CAMPAIGN}_\$(Cluster)_bin\$(bin_id)_receipt.json"

request_cpus = 1
request_memory = 4GB
request_disk = 20GB

getenv = False
environment = "LC_ALL=C LANG=C"
use_x509userproxy = True
x509userproxy = $X509_PROXY

+JobBatchName = "$CAMPAIGN"
+DesiredOS = "EL9"

on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)

queue bin_id from (
EOF

for BIN_ID in "${BIN_IDS[@]}"; do
  printf '%s\n' "$BIN_ID" >> "$SUBMIT_FILE"
done

printf ')\n' >> "$SUBMIT_FILE"

printf '%s\n' \
  'campaign,bin_id,pthat_min_GeV,pthat_max_GeV,n_events,shard_id,seed,payload_sha256,wrapper_sha256,card_sha256,git_head,eos_directory' \
  > "$CAMPAIGN_MANIFEST"

PTHAT_MIN=(50 75 100 200 300 500 700 1000)
PTHAT_MAX=(75 100 200 300 500 700 1000 0)

for BIN_ID in "${BIN_IDS[@]}"; do
  SEED=$((1200000 + BIN_ID * 10000 + SHARD_ID))
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$CAMPAIGN" \
    "$BIN_ID" \
    "${PTHAT_MIN[$BIN_ID]}" \
    "${PTHAT_MAX[$BIN_ID]}" \
    "$N_EVENTS" \
    "$SHARD_ID" \
    "$SEED" \
    "$PAYLOAD_SHA256" \
    "$WRAPPER_SHA256" \
    "$CARD_SHA256" \
    "$GIT_HEAD" \
    "$EOS_DIR" \
    >> "$CAMPAIGN_MANIFEST"
done

echo "Prepared submit file: $SUBMIT_FILE"
echo "Campaign manifest: $CAMPAIGN_MANIFEST"
echo "Payload SHA-256: $PAYLOAD_SHA256"
echo "Worker wrapper SHA-256: $WRAPPER_SHA256"
echo "EOS directory: $EOS_DIR"
echo "Submit only once with: condor_submit $SUBMIT_FILE"
