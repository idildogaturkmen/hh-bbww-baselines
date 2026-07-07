#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

HH4B_REPO="${HH4B_REPO:-$DEFAULT_REPO}"
HH4B_STORE="${HH4B_STORE:-/uscms_data/d3/${USER}/hh4b_delphes}"
DELPHES_DIR="${DELPHES_DIR:-/uscms_data/d3/${USER}/software/Delphes}"
MG5_TEMPLATE="${MG5_TEMPLATE:-$HH4B_STORE/mg5/HH4b_smoke_vbf}"

CONDOR_INPUT_DIR="${CONDOR_INPUT_DIR:-$HH4B_STORE/condor_inputs}"
TARBALL="${TARBALL:-$CONDOR_INPUT_DIR/hh4b_vbf_condor_inputs.tar.gz}"

mkdir -p "$CONDOR_INPUT_DIR"
mkdir -p \
  "$HH4B_STORE/condor_return" \
  "$HH4B_STORE/condor_logs" \
  "$HH4B_STORE/root" \
  "$HH4B_STORE/parquet" \
  "$HH4B_STORE/metadata" \
  "$HH4B_STORE/logs"

STAGE_DIR="$(mktemp -d "$CONDOR_INPUT_DIR/payload_stage.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

PAYLOAD="$STAGE_DIR/payload"
REPO_PAYLOAD="$PAYLOAD/repo"

mkdir -p \
  "$REPO_PAYLOAD/scripts/delphes" \
  "$REPO_PAYLOAD/cards/delphes" \
  "$REPO_PAYLOAD/cards/mg5" \
  "$PAYLOAD/Delphes" \
  "$PAYLOAD/mg5_template"

if [ ! -d "$HH4B_REPO/.git" ]; then
  echo "ERROR: HH4B_REPO does not look like a repo: $HH4B_REPO"
  exit 2
fi

if [ ! -d "$MG5_TEMPLATE" ]; then
  echo "ERROR: missing MG5 template process: $MG5_TEMPLATE"
  exit 3
fi

if [ ! -x "$DELPHES_DIR/DelphesHepMC3" ]; then
  echo "ERROR: missing executable DelphesHepMC3 under DELPHES_DIR=$DELPHES_DIR"
  exit 4
fi

cp "$HH4B_REPO"/scripts/delphes/*.py "$REPO_PAYLOAD/scripts/delphes/"
cp "$HH4B_REPO"/scripts/delphes/*.sh "$REPO_PAYLOAD/scripts/delphes/"
cp "$HH4B_REPO"/scripts/delphes/*.cc "$REPO_PAYLOAD/scripts/delphes/"
cp "$HH4B_REPO"/cards/delphes/*.tcl "$REPO_PAYLOAD/cards/delphes/"
cp "$HH4B_REPO"/cards/mg5/*.mg5 "$REPO_PAYLOAD/cards/mg5/"

tar -C "$MG5_TEMPLATE" \
  --exclude './Events' \
  --exclude './HTML' \
  --exclude './*.log' \
  --exclude './run_*_debug.log' \
  -cf - . | tar -C "$PAYLOAD/mg5_template" -xf -

for file in \
  DelphesHepMC3 \
  libDelphes.so \
  ClassesDict_rdict.pcm \
  ExRootAnalysisDict_rdict.pcm \
  ModulesDict_rdict.pcm \
  ModulesFastJetDict_rdict.pcm \
  DelphesEnv.sh
do
  if [ ! -e "$DELPHES_DIR/$file" ]; then
    echo "ERROR: missing Delphes runtime file: $DELPHES_DIR/$file"
    exit 5
  fi
  cp "$DELPHES_DIR/$file" "$PAYLOAD/Delphes/"
done

cat > "$PAYLOAD/manifest.txt" <<EOF
HH4B Condor transfer payload
created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
repo=$HH4B_REPO
mg5_template=$MG5_TEMPLATE
delphes_dir=$DELPHES_DIR
delphes_runtime=minimal
sample=VBF HH -> 4b validation
EOF

TMP_TARBALL="$TARBALL.tmp"
tar -C "$STAGE_DIR" -czf "$TMP_TARBALL" payload
mv "$TMP_TARBALL" "$TARBALL"

echo "Prepared Condor input tarball:"
echo "  $TARBALL"
du -h "$TARBALL"
echo
echo "Payload manifest:"
sed -n '1,120p' "$PAYLOAD/manifest.txt"
