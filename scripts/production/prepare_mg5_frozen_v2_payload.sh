#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 FAMILY PROCESS_DIR OUTPUT_TARBALL" >&2
  exit 2
fi

FAMILY="$1"
PROCESS_DIR="$2"
TARBALL="$3"

HH4B_REPO="${HH4B_REPO:-/uscms_data/d3/$USER/repos/hh-bbww-baselines}"
DELPHES_DIR="${DELPHES_DIR:-/uscms_data/d3/$USER/software/Delphes}"

[[ "$FAMILY" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]

test -d "$PROCESS_DIR"
test -x "$PROCESS_DIR/bin/generate_events"
test -s "$PROCESS_DIR/Cards/proc_card_mg5.dat"
test -s "$PROCESS_DIR/Cards/run_card.dat"

WORKER="$HH4B_REPO/scripts/production/run_mg5_frozen_v2_bundle.sh"
EVENT_SCRIPT="$HH4B_REPO/scripts/delphes/make_delphes_event_summary.py"
CANDIDATE_SCRIPT="$HH4B_REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
PARQUET_WRITER="$HH4B_REPO/scripts/delphes/write_parquet_from_pickle.py"
CONVERTER_SOURCE="$HH4B_REPO/scripts/production/lhe_to_hepmc3_seeded.cc"
CARD="$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"

test -x "$WORKER"
test -s "$EVENT_SCRIPT"
test -s "$CANDIDATE_SCRIPT"
test -s "$PARQUET_WRITER"
test -s "$CONVERTER_SOURCE"
test -s "$CARD"

EXPECTED_CARD_SHA256="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

CARD_SHA256="$(
  sha256sum "$CARD" |
  awk '{print $1}'
)"

test "$CARD_SHA256" = "$EXPECTED_CARD_SHA256"

for FILE in \
  DelphesHepMC3 \
  libDelphes.so \
  ClassesDict_rdict.pcm \
  ExRootAnalysisDict_rdict.pcm \
  ModulesDict_rdict.pcm \
  ModulesFastJetDict_rdict.pcm \
  DelphesEnv.sh
do
  test -s "$DELPHES_DIR/$FILE"
done

mkdir -p "$(dirname "$TARBALL")"

STAGE="$(
  mktemp -d "$(dirname "$TARBALL")/mg5_frozen_v2_payload.XXXXXX"
)"

cleanup() {
  rm -rf "$STAGE"
}

trap cleanup EXIT

PAYLOAD="$STAGE/payload"

mkdir -p \
  "$PAYLOAD/repo/scripts/delphes" \
  "$PAYLOAD/repo/scripts/production" \
  "$PAYLOAD/repo/cards/delphes" \
  "$PAYLOAD/Delphes" \
  "$PAYLOAD/mg5_template"

cp \
  "$EVENT_SCRIPT" \
  "$CANDIDATE_SCRIPT" \
  "$PARQUET_WRITER" \
  "$PAYLOAD/repo/scripts/delphes/"

cp \
  "$CONVERTER_SOURCE" \
  "$PAYLOAD/repo/scripts/production/"

cp \
  "$CARD" \
  "$PAYLOAD/repo/cards/delphes/"

for FILE in \
  DelphesHepMC3 \
  libDelphes.so \
  ClassesDict_rdict.pcm \
  ExRootAnalysisDict_rdict.pcm \
  ModulesDict_rdict.pcm \
  ModulesFastJetDict_rdict.pcm \
  DelphesEnv.sh
do
  cp \
    "$DELPHES_DIR/$FILE" \
    "$PAYLOAD/Delphes/"
done

tar \
  -C "$PROCESS_DIR" \
  --exclude='./Events' \
  --exclude='./HTML' \
  --exclude='./*.log' \
  --exclude='./run_*_debug.log' \
  -cf - \
  . |
tar \
  -C "$PAYLOAD/mg5_template" \
  -xf -

GIT_HEAD="$(
  git -C "$HH4B_REPO" rev-parse HEAD
)"

WORKER_SHA256="$(
  sha256sum "$WORKER" |
  awk '{print $1}'
)"

EVENT_SHA256="$(
  sha256sum "$EVENT_SCRIPT" |
  awk '{print $1}'
)"

CANDIDATE_SHA256="$(
  sha256sum "$CANDIDATE_SCRIPT" |
  awk '{print $1}'
)"

PARQUET_SHA256="$(
  sha256sum "$PARQUET_WRITER" |
  awk '{print $1}'
)"

CONVERTER_SHA256="$(
  sha256sum "$CONVERTER_SOURCE" |
  awk '{print $1}'
)"

PROCESS_DEFINITION="$(
  grep -E \
    '^[[:space:]]*(generate|add process)[[:space:]]' \
    "$PROCESS_DIR/Cards/proc_card_mg5.dat" |
  tr '\n' ' '
)"

cat > "$PAYLOAD/manifest.txt" <<EOF
schema_version=1
family=$FAMILY
operation=mg5_to_seeded_pythia_to_frozen_v2_delphes
created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
git_head=$GIT_HEAD
process_definition=$PROCESS_DEFINITION
card_sha256=$CARD_SHA256
worker_wrapper_sha256=$WORKER_SHA256
event_summary_script_sha256=$EVENT_SHA256
candidate_script_sha256=$CANDIDATE_SHA256
parquet_writer_sha256=$PARQUET_SHA256
seeded_converter_sha256=$CONVERTER_SHA256
EOF

TMP="$TARBALL.tmp"

tar \
  -C "$STAGE" \
  -czf "$TMP" \
  payload

mv "$TMP" "$TARBALL"

chmod 0444 "$TARBALL"

echo "MG5_FROZEN_V2_PAYLOAD_VALID"
echo "family=$FAMILY"
echo "payload=$TARBALL"
sha256sum "$TARBALL"
du -h "$TARBALL"
