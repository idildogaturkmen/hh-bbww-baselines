#!/usr/bin/env bash
set -euo pipefail

export HH4B_REPO="${HH4B_REPO:-/uscms_data/d3/$USER/repos/hh-bbww-baselines}"
export HH4B_STORE="${HH4B_STORE:-/uscms_data/d3/$USER/hh4b_delphes}"
export DELPHES_DIR="${DELPHES_DIR:-/uscms_data/d3/$USER/software/Delphes}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT_TARBALL]" >&2
  exit 1
fi

DEFAULT_INPUT_DIR="$HH4B_STORE/condor_inputs"
TARBALL="${1:-$DEFAULT_INPUT_DIR/qcd_hardqcd_importance_inputs.tar.gz}"
INPUT_DIR=$(dirname "$TARBALL")

CARD="$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
EXPECTED_HASH="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

mkdir -p "$INPUT_DIR"

OBSERVED_HASH=$(sha256sum "$CARD" | awk '{print $1}')

if [[ "$OBSERVED_HASH" != "$EXPECTED_HASH" ]]; then
  echo "ERROR: frozen-v2 card hash mismatch"
  exit 1
fi

STAGE=$(mktemp -d "$INPUT_DIR/qcd_payload.XXXXXX")

cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

PAYLOAD="$STAGE/payload"

mkdir -p \
  "$PAYLOAD/repo/scripts/production" \
  "$PAYLOAD/repo/scripts/delphes" \
  "$PAYLOAD/repo/cards/delphes" \
  "$PAYLOAD/Delphes"

cp \
  "$HH4B_REPO/scripts/production/generate_pythia8_hardqcd_hepmc3.cc" \
  "$HH4B_REPO/scripts/production/compile_pythia8_hepmc3.sh" \
  "$PAYLOAD/repo/scripts/production/"

cp \
  "$HH4B_REPO/scripts/delphes/make_delphes_event_summary.py" \
  "$HH4B_REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py" \
  "$HH4B_REPO/scripts/delphes/write_parquet_from_pickle.py" \
  "$PAYLOAD/repo/scripts/delphes/"

cp "$CARD" "$PAYLOAD/repo/cards/delphes/"

for FILE in \
  DelphesHepMC3 \
  libDelphes.so \
  ClassesDict_rdict.pcm \
  ExRootAnalysisDict_rdict.pcm \
  ModulesDict_rdict.pcm \
  ModulesFastJetDict_rdict.pcm \
  DelphesEnv.sh
do
  test -e "$DELPHES_DIR/$FILE" || {
    echo "ERROR: missing Delphes runtime file: $FILE"
    exit 2
  }

  cp "$DELPHES_DIR/$FILE" "$PAYLOAD/Delphes/"
done

GIT_HEAD=$(git -C "$HH4B_REPO" rev-parse HEAD)
GENERATOR_SHA=$(sha256sum "$HH4B_REPO/scripts/production/generate_pythia8_hardqcd_hepmc3.cc" | awk '{print $1}')
COMPILE_HELPER_SHA=$(sha256sum "$HH4B_REPO/scripts/production/compile_pythia8_hepmc3.sh" | awk '{print $1}')
WRAPPER_SHA=$(sha256sum "$HH4B_REPO/scripts/production/run_qcd_importance_bundle.sh" | awk '{print $1}')
PARQUET_WRITER_SHA=$(sha256sum "$HH4B_REPO/scripts/delphes/write_parquet_from_pickle.py" | awk '{print $1}')

cat > "$PAYLOAD/manifest.txt" <<EOF
sample=Pythia8_HardQCD_importance_pilot
created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
git_head=$GIT_HEAD
card_sha256=$OBSERVED_HASH
generator_source_sha256=$GENERATOR_SHA
compile_helper_sha256=$COMPILE_HELPER_SHA
worker_wrapper_sha256=$WRAPPER_SHA
parquet_writer_sha256=$PARQUET_WRITER_SHA
sqrt_s_GeV=13000
tune=Monash2013_TuneEE7_TunePP14
EOF

TMP="$TARBALL.tmp"

tar -C "$STAGE" -czf "$TMP" payload
mv "$TMP" "$TARBALL"

echo "Prepared: $TARBALL"
sha256sum "$TARBALL"
du -h "$TARBALL"
