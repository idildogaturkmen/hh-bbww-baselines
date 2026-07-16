#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

HH4B_REPO="${HH4B_REPO:-/uscms_data/d3/$USER/repos/hh-bbww-baselines}"
DELPHES_DIR="${DELPHES_DIR:-/uscms_data/d3/$USER/software/Delphes}"
OUTPUT_DIR="${1:-$HH4B_REPO/outputs/agent_runs/qcd_importance_20260716/compile_preflight}"
APPTAINER="${APPTAINER:-/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer}"
WORKER_IMAGE="${HH4B_WORKER_IMAGE:-/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel9}"

test -x "$APPTAINER" || {
  echo "ERROR: missing Apptainer executable: $APPTAINER" >&2
  exit 3
}

test -e "$WORKER_IMAGE" || {
  echo "ERROR: missing EL9 worker image: $WORKER_IMAGE" >&2
  exit 3
}

if [[ -e "$OUTPUT_DIR/job_receipt.json" || -e "$OUTPUT_DIR/qcd_compile_preflight_inputs.tar.gz" ]]; then
  echo "ERROR: refusing to overwrite an existing preflight: $OUTPUT_DIR" >&2
  exit 4
fi

mkdir -p "$OUTPUT_DIR"

PREFLIGHT_TARBALL="$OUTPUT_DIR/qcd_compile_preflight_inputs.tar.gz"

HH4B_REPO="$HH4B_REPO" \
DELPHES_DIR="$DELPHES_DIR" \
  "$HH4B_REPO/scripts/production/prepare_condor_qcd_importance_inputs.sh" \
  "$PREFLIGHT_TARBALL"

cp \
  "$HH4B_REPO/scripts/production/run_qcd_importance_bundle.sh" \
  "$OUTPUT_DIR/run_qcd_importance_bundle.sh"

chmod +x "$OUTPUT_DIR/run_qcd_importance_bundle.sh"
PREFLIGHT_PAYLOAD_SHA256=$(sha256sum "$PREFLIGHT_TARBALL" | awk '{print $1}')

(
  unset SOURCE
  "$APPTAINER" exec \
    --pid \
    --ipc \
    --contain \
    --cleanenv \
    --env LC_ALL=C \
    --env LANG=C \
    --bind /cvmfs \
    --home "$OUTPUT_DIR:/srv" \
    --pwd /srv \
    "$WORKER_IMAGE" \
    ./run_qcd_importance_bundle.sh \
    0 \
    900 \
    1 \
    qcd_compile_preflight_20260716 \
    local \
    qcd_compile_preflight_inputs.tar.gz \
    /store/user/unused/qcd_compile_preflight \
    "$PREFLIGHT_PAYLOAD_SHA256" \
    --compile-only
) 2>&1 | tee "$OUTPUT_DIR/preflight.log"

python3 - "$OUTPUT_DIR/job_receipt.json" <<'PY_RECEIPT'
from pathlib import Path
import json
import sys

receipt = json.loads(Path(sys.argv[1]).read_text())

if receipt["exit_status"] != 0:
    raise SystemExit("ERROR: compile preflight receipt is nonzero")

if receipt["stage"] != "compile_preflight_complete":
    raise SystemExit("ERROR: compile preflight receipt has the wrong stage")

for key in (
    "payload_sha256",
    "expected_payload_sha256",
    "delphes_card_sha256",
    "wrapper_sha256",
    "lcg_setup_sha256",
    "pythia_configuration",
    "hepmc3_configuration",
):
    if not receipt.get(key):
        raise SystemExit(f"ERROR: compile preflight receipt lacks {key}")

if receipt["payload_sha256"] != receipt["expected_payload_sha256"]:
    raise SystemExit("ERROR: compile preflight payload hashes disagree")

print(json.dumps(receipt, indent=2, sort_keys=True))
PY_RECEIPT

echo "EL9_COMPILE_PREFLIGHT_SUCCESS=$OUTPUT_DIR"
