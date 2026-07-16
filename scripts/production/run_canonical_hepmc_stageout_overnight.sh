#!/usr/bin/env bash
# Keep errexit and pipefail active, but enable nounset only after
# the external LCG environment has been sourced.
set -eo pipefail

export HH4B_REPO="/uscms_data/d3/$USER/repos/hh-bbww-baselines"
export HH4B_STORE="/uscms_data/d3/$USER/hh4b_delphes"

cd "$HH4B_REPO"
source scripts/delphes/setup_lpc_delphes_env.sh

# The LCG setup references variables that may initially be unset.
# It is now safe to enable strict undefined-variable checking.
set -u

export X509_USER_PROXY="/tmp/x509up_u$(id -u)"

export HH4B_EOS_HOST="root://cmseos.fnal.gov"
export HH4B_EOS_PRODUCTION="/store/user/$USER/hh4b_delphes/run2_13tev/frozen_v2"

export HH4B_DELPHES_CARD="$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"

export AUDIT_DIR="$HH4B_REPO/outputs/audits/hepmc_lineage_run2_2026_07_16"
export LINEAGE_MANIFEST="$AUDIT_DIR/canonical_frozen_v2_rerun_manifest.csv"

export TRANSFER_DIR="$HH4B_STORE/metadata/eos_stageout"
export TRANSFER_MANIFEST="$TRANSFER_DIR/canonical_hepmc_transfer.jsonl"
export VALIDATION_SUMMARY="$TRANSFER_DIR/canonical_hepmc_eos_validation.json"

LOG_DIR="$HH4B_STORE/logs/eos_stageout"
STAGEOUT_LOG="$LOG_DIR/canonical_hepmc_stageout.log"

EXPECTED_CARD_HASH="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

mkdir -p "$TRANSFER_DIR" "$LOG_DIR"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Repository: $HH4B_REPO"
echo "Local store: $HH4B_STORE"
echo "EOS production: $HH4B_EOS_PRODUCTION"
echo "Proxy: $X509_USER_PROXY"

if [[ ! -r "$X509_USER_PROXY" ]]; then
    echo "ERROR: missing proxy: $X509_USER_PROXY"
    exit 1
fi

PROXY_SECONDS=$(voms-proxy-info -timeleft)

echo "Proxy time remaining: $PROXY_SECONDS seconds"

if (( PROXY_SECONDS < 14400 )); then
    echo "ERROR: less than four hours remain on the proxy"
    exit 1
fi

OBSERVED_CARD_HASH=$(
    sha256sum "$HH4B_DELPHES_CARD" |
    awk '{print $1}'
)

echo "Expected card hash: $EXPECTED_CARD_HASH"
echo "Observed card hash: $OBSERVED_CARD_HASH"

if [[ "$OBSERVED_CARD_HASH" != "$EXPECTED_CARD_HASH" ]]; then
    echo "ERROR: frozen-v2 card checksum mismatch"
    exit 1
fi

if [[ ! -s "$LINEAGE_MANIFEST" ]]; then
    echo "ERROR: missing lineage manifest: $LINEAGE_MANIFEST"
    exit 1
fi

if [[ ! -x scripts/production/stageout_canonical_hepmc_to_eos.py ]]; then
    echo "ERROR: missing executable stage-out script"
    exit 1
fi

echo
echo "=== Starting canonical HepMC stage-out ==="

python3 scripts/production/stageout_canonical_hepmc_to_eos.py \
    --lineage-manifest "$LINEAGE_MANIFEST" \
    --transfer-manifest "$TRANSFER_MANIFEST" \
    2>&1 | tee "$STAGEOUT_LOG"

echo
echo "=== Independently validating EOS inventory ==="

python3 - <<'PY'
from pathlib import Path
import csv
import datetime as dt
import json
import os
import subprocess

lineage_path = Path(os.environ["LINEAGE_MANIFEST"])
transfer_path = Path(os.environ["TRANSFER_MANIFEST"])
validation_path = Path(os.environ["VALIDATION_SUMMARY"])

host = os.environ["HH4B_EOS_HOST"]
base = os.environ["HH4B_EOS_PRODUCTION"].rstrip("/")

with lineage_path.open(newline="", encoding="utf-8") as handle:
    lineage = list(csv.DictReader(handle))

records = [
    json.loads(line)
    for line in transfer_path.read_text().splitlines()
    if line.strip()
]

# Repeated tests or resumed transfers can produce more than one record.
# Retain the newest record for each remote object.
latest = {
    record["remote_lfn"]: record
    for record in records
}

expected = {
    f"{base}/hepmc/{row['filename']}": row
    for row in lineage
}

missing = sorted(set(expected) - set(latest))

if missing:
    raise SystemExit(
        "Missing transfer records:\n" + "\n".join(missing)
    )

verified_bytes = 0
status_counts = {}

for remote_lfn in sorted(expected):
    transfer = latest[remote_lfn]

    if transfer["sha256"] != expected[remote_lfn]["sha256"]:
        raise SystemExit(
            f"SHA-256 lineage mismatch: {remote_lfn}"
        )

    if transfer["adler32"] != transfer["remote_adler32"]:
        raise SystemExit(
            f"Recorded Adler-32 mismatch: {remote_lfn}"
        )

    result = subprocess.run(
        [
            "xrdfs",
            host,
            "query",
            "checksum",
            remote_lfn,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    observed = result.stdout.strip().split()[-1].lower()

    if observed != transfer["adler32"]:
        raise SystemExit(
            f"Live EOS checksum mismatch: {remote_lfn}\n"
            f"expected={transfer['adler32']} observed={observed}"
        )

    verified_bytes += int(transfer["size_bytes"])

    status = transfer["status"]
    status_counts[status] = status_counts.get(status, 0) + 1

summary = {
    "validated_at_utc": dt.datetime.now(
        dt.timezone.utc
    ).isoformat(),
    "expected_canonical_files": len(expected),
    "verified_eos_files": len(expected),
    "verified_events": sum(
        int(row["n_events"])
        for row in lineage
    ),
    "verified_size_bytes": verified_bytes,
    "verified_size_GB": verified_bytes / 1.0e9,
    "status_counts": status_counts,
    "lineage_manifest": str(lineage_path),
    "transfer_manifest": str(transfer_path),
    "eos_base": base,
    "result": "CANONICAL_EOS_TRANSFER_VALID",
}

validation_path.write_text(
    json.dumps(summary, indent=2) + "\n"
)

print(json.dumps(summary, indent=2))
print("CANONICAL EOS TRANSFER VALID")
PY

echo
echo "=== Copying lineage metadata to EOS ==="

REMOTE_METADATA_DIR="$HH4B_EOS_PRODUCTION/metadata/hepmc_lineage_run2_2026_07_16"

xrdfs "$HH4B_EOS_HOST" mkdir -p "$REMOTE_METADATA_DIR"

FILES_TO_COPY=(
    "$AUDIT_DIR/canonical_frozen_v2_rerun_manifest.csv"
    "$AUDIT_DIR/canonical_hepmc_summary.csv"
    "$AUDIT_DIR/exact_duplicate_hepmc_files.csv"
    "$AUDIT_DIR/summary.json"
    "$TRANSFER_MANIFEST"
    "$VALIDATION_SUMMARY"
    "$STAGEOUT_LOG"
)

for LOCAL_FILE in "${FILES_TO_COPY[@]}"; do
    if [[ ! -s "$LOCAL_FILE" ]]; then
        echo "ERROR: missing metadata file: $LOCAL_FILE"
        exit 1
    fi

    BASENAME=$(basename "$LOCAL_FILE")
    REMOTE_LFN="$REMOTE_METADATA_DIR/$BASENAME"

    echo "Staging metadata: $BASENAME"

    xrdcp \
        -f \
        --cksum adler32:print \
        "$LOCAL_FILE" \
        "${HH4B_EOS_HOST}/${REMOTE_LFN}"
done

COMPLETION_FILE="$TRANSFER_DIR/canonical_hepmc_stageout.complete"

{
    echo "status=complete"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "lineage_manifest=$LINEAGE_MANIFEST"
    echo "transfer_manifest=$TRANSFER_MANIFEST"
    echo "validation_summary=$VALIDATION_SUMMARY"
} > "$COMPLETION_FILE"

xrdcp \
    -f \
    --cksum adler32:print \
    "$COMPLETION_FILE" \
    "${HH4B_EOS_HOST}/${REMOTE_METADATA_DIR}/$(basename "$COMPLETION_FILE")"

echo
echo "=== Final EOS quota ==="
eosquota

echo
echo "OVERNIGHT CANONICAL HEPMC STAGE-OUT COMPLETE"
echo "Validation: $VALIDATION_SUMMARY"
echo "EOS metadata: $REMOTE_METADATA_DIR"
