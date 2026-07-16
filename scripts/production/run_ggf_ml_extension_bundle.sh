#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 LOCAL_SHARD N_EVENTS CAMPAIGN CLUSTER_ID INPUT_TARBALL EOS_BUNDLE_DIR"
  exit 1
fi

LOCAL_SHARD="$1"
N_EVENTS="$2"
CAMPAIGN="$3"
CLUSTER_ID="$4"
INPUT_TARBALL="$5"
EOS_BUNDLE_DIR="$6"

[[ "$LOCAL_SHARD" =~ ^[0-9]+$ ]] || { echo "ERROR: bad shard"; exit 1; }
[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: bad event count"; exit 1; }
[[ "$CAMPAIGN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: bad campaign"; exit 1; }
[[ "$EOS_BUNDLE_DIR" == /store/user/*/hh4b_delphes/* ]] || {
  echo "ERROR: unexpected EOS destination: $EOS_BUNDLE_DIR"
  exit 1
}

# The tested inner wrapper uses SEED=70000+SHARD_ID.  Shards 0--9 were
# previously used, so this extension uses inner shards 16000--16019,
# corresponding to fresh seeds 86000--86019.
INNER_SHARD=$((16000 + LOCAL_SHARD))
SEED=$((70000 + INNER_SHARD))
SHARD_PADDED=$(printf '%03d' "$LOCAL_SHARD")
TAG="HH4b_ggf_${CAMPAIGN}_shard${SHARD_PADDED}_seed${SEED}"

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"
if [[ "$(pwd -P)" != /srv* && "${ALLOW_NON_SRV_FOR_TESTING:-0}" != "1" ]]; then
  echo "ERROR: expected Condor scratch under /srv; got $(pwd -P)"
  exit 2
fi

RECEIPT="$SCRATCH/job_receipt.json"
STAGE="initializing"
REMOTE_BUNDLE=""
ADLER32=""

finalize() {
  rc=$?
  cat > "$RECEIPT" <<EOF
{
  "campaign": "$CAMPAIGN",
  "local_shard": $LOCAL_SHARD,
  "inner_shard": $INNER_SHARD,
  "n_events": $N_EVENTS,
  "seed": $SEED,
  "cluster_id": "$CLUSTER_ID",
  "stage": "$STAGE",
  "remote_bundle": "$REMOTE_BUNDLE",
  "adler32": "$ADLER32",
  "exit_status": $rc,
  "finished_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF
  trap - EXIT
  exit "$rc"
}
trap finalize EXIT

echo "HOST=$(hostname -f)"
echo "CAMPAIGN=$CAMPAIGN"
echo "LOCAL_SHARD=$LOCAL_SHARD"
echo "INNER_SHARD=$INNER_SHARD"
echo "SEED=$SEED"
echo "N_EVENTS=$N_EVENTS"
echo "EOS_BUNDLE_DIR=$EOS_BUNDLE_DIR"

if [ ! -s "$INPUT_TARBALL" ]; then
  echo "ERROR: missing payload: $INPUT_TARBALL"
  exit 3
fi

STAGE="checking_payload"
PAYLOAD_SHA256=$(sha256sum "$INPUT_TARBALL" | awk '{print $1}')
tar -xzf "$INPUT_TARBALL"

PAYLOAD_REPO="$SCRATCH/payload/repo"
CARD="$PAYLOAD_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
INNER_WRAPPER="$PAYLOAD_REPO/scripts/delphes/run_ggf_hh4b_shard_transfer.sh"
V2_RECO="$PAYLOAD_REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
EXPECTED_CARD_SHA256="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

if [ ! -s "$CARD" ] || [ ! -x "$INNER_WRAPPER" ] || [ ! -s "$V2_RECO" ]; then
  echo "ERROR: payload lacks the frozen card, ggF wrapper, or v2 reconstruction"
  exit 4
fi

CARD_SHA256=$(sha256sum "$CARD" | awk '{print $1}')
if [ "$CARD_SHA256" != "$EXPECTED_CARD_SHA256" ]; then
  echo "ERROR: frozen-v2 card hash mismatch"
  echo "expected=$EXPECTED_CARD_SHA256"
  echo "observed=$CARD_SHA256"
  exit 4
fi

export HH4B_DELPHES_CARD="$CARD"

STAGE="generating_and_reconstructing"
set +e
"$INNER_WRAPPER" \
  "$INNER_SHARD" \
  "$N_EVENTS" \
  "$CAMPAIGN" \
  "$CLUSTER_ID" \
  "$INPUT_TARBALL"
INNER_STATUS=$?
set -e

if [ "$INNER_STATUS" -ne 0 ]; then
  echo "ERROR: inner generation pipeline failed: $INNER_STATUS"
  exit "$INNER_STATUS"
fi

if [ ! -s "$SCRATCH/shard_output.tar.gz" ]; then
  echo "ERROR: missing returned reconstruction archive"
  exit 5
fi

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
if [ ! -r "$LCG_SETUP" ]; then
  echo "ERROR: LCG setup cannot be read"
  exit 5
fi
set +u
# shellcheck disable=SC1090
source "$LCG_SETUP"
set -u

for cmd in python3 xrdcp xrdfs tar; do
  command -v "$cmd" >/dev/null || {
    echo "ERROR: missing command after LCG setup: $cmd"
    exit 5
  }
done

HEPMC=$(find "$SCRATCH/work" -maxdepth 1 -type f -name '*.hepmc' -print -quit)
if [ -z "$HEPMC" ] || [ ! -s "$HEPMC" ]; then
  echo "ERROR: generated HepMC file was not retained"
  exit 5
fi

N_HEPMC=$(grep -c '^E ' "$HEPMC")
if [ "$N_HEPMC" -ne "$N_EVENTS" ]; then
  echo "ERROR: HepMC has $N_HEPMC events; expected $N_EVENTS"
  exit 6
fi

ARCHIVE_LIST="$SCRATCH/shard_output.contents.txt"
tar -tzf "$SCRATCH/shard_output.tar.gz" > "$ARCHIVE_LIST"
if ! grep -E '(^|/)root/.*[.]root$' "$ARCHIVE_LIST" >/dev/null; then
  echo "ERROR: reconstruction archive contains no ROOT file"
  exit 6
fi

STAGE="rebuilding_v2_parquet"
REBUILT="$SCRATCH/rebuilt_products"
mkdir -p "$REBUILT"
tar -xzf "$SCRATCH/shard_output.tar.gz" -C "$REBUILT"
ROOT_FILE=$(find "$REBUILT/root" -maxdepth 1 -type f -name '*.root' -print -quit)
if [ -z "$ROOT_FILE" ] || [ ! -s "$ROOT_FILE" ]; then
  echo "ERROR: ROOT file could not be extracted"
  exit 6
fi

N_ROOT=$(python3 - "$ROOT_FILE" <<'PY'
import sys
import uproot
with uproot.open(sys.argv[1]) as handle:
    print(handle["Delphes"].num_entries)
PY
)
if [ "$N_ROOT" -ne "$N_EVENTS" ]; then
  echo "ERROR: ROOT has $N_ROOT events; expected $N_EVENTS"
  exit 6
fi

python3 "$V2_RECO" \
  --input "$ROOT_FILE" \
  --out "$REBUILT/parquet/${TAG}_hh4b_candidates_v2.parquet" \
  --sample "$TAG"

REBUILT_ARCHIVE="$SCRATCH/${TAG}_reconstruction.tar.gz"
tar -C "$REBUILT" -czf "$REBUILT_ARCHIVE" .

HEPMC_SHA256=$(sha256sum "$HEPMC" | awk '{print $1}')
PRODUCTS_SHA256=$(sha256sum "$REBUILT_ARCHIVE" | awk '{print $1}')

STAGE="building_bundle"
BUNDLE_DIR="$SCRATCH/bundle"
mkdir -p "$BUNDLE_DIR/hepmc" "$BUNDLE_DIR/products" "$BUNDLE_DIR/metadata"
cp "$HEPMC" "$BUNDLE_DIR/hepmc/${TAG}_pythia8.hepmc"
cp "$REBUILT_ARCHIVE" "$BUNDLE_DIR/products/${TAG}_reconstruction.tar.gz"

cat > "$BUNDLE_DIR/metadata/${TAG}_provenance.json" <<EOF
{
  "sample": "HEFT ggF HH -> 4b ML-training approximation",
  "physics_use": "architecture and reconstruction training; not final precision ggF modeling",
  "campaign": "$CAMPAIGN",
  "local_shard": $LOCAL_SHARD,
  "inner_shard": $INNER_SHARD,
  "seed": $SEED,
  "n_events": $N_EVENTS,
  "cluster_id": "$CLUSTER_ID",
  "payload_sha256": "$PAYLOAD_SHA256",
  "delphes_card_sha256": "$CARD_SHA256",
  "hepmc_sha256": "$HEPMC_SHA256",
  "reconstruction_archive_sha256": "$PRODUCTS_SHA256",
  "immutable_split": "not_assigned",
  "created_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

BUNDLE="$SCRATCH/${TAG}_bundle.tar.gz"
tar -C "$BUNDLE_DIR" -czf "$BUNDLE" .
ADLER32=$(python3 - "$BUNDLE" <<'PY'
import sys
import zlib

checksum = 1
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        checksum = zlib.adler32(chunk, checksum)
print(f"{checksum & 0xffffffff:08x}")
PY
)

STAGE="staging_to_eos"
EOS_HOST="root://cmseos.fnal.gov"
REMOTE_BUNDLE="$EOS_BUNDLE_DIR/${TAG}_bundle.tar.gz"
REMOTE_TMP="${REMOTE_BUNDLE}.partial.${CLUSTER_ID}.${LOCAL_SHARD}.$$"
xrdfs "$EOS_HOST" mkdir -p "$EOS_BUNDLE_DIR"

set +e
EXISTING_REPLY=$(xrdfs "$EOS_HOST" query checksum "$REMOTE_BUNDLE" 2>/dev/null)
EXISTING_STATUS=$?
set -e
if [ "$EXISTING_STATUS" -eq 0 ]; then
  EXISTING_SUM=$(printf '%s\n' "$EXISTING_REPLY" | awk '{print tolower($NF)}')
  if [ "$EXISTING_SUM" = "$ADLER32" ]; then
    STAGE="complete_already_verified"
    exit 0
  fi
  echo "ERROR: destination exists with a different checksum"
  exit 7
fi

cleanup_partial() {
  xrdfs "$EOS_HOST" rm "$REMOTE_TMP" >/dev/null 2>&1 || true
}
trap cleanup_partial ERR

xrdcp -f --nopbar --cksum adler32:print "$BUNDLE" "${EOS_HOST}/${REMOTE_TMP}"
REMOTE_TMP_SUM=$(xrdfs "$EOS_HOST" query checksum "$REMOTE_TMP" | awk '{print tolower($NF)}')
if [ "$REMOTE_TMP_SUM" != "$ADLER32" ]; then
  echo "ERROR: temporary EOS checksum mismatch"
  exit 8
fi

xrdfs "$EOS_HOST" mv "$REMOTE_TMP" "$REMOTE_BUNDLE"
trap - ERR

REMOTE_FINAL_SUM=$(xrdfs "$EOS_HOST" query checksum "$REMOTE_BUNDLE" | awk '{print tolower($NF)}')
if [ "$REMOTE_FINAL_SUM" != "$ADLER32" ]; then
  echo "ERROR: final EOS checksum mismatch"
  exit 9
fi

STAGE="complete_copied_and_verified"
echo "SUCCESS: $REMOTE_BUNDLE"
echo "ADLER32: $ADLER32"
