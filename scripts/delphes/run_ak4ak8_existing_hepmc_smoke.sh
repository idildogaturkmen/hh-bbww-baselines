#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"
: "${DELPHES_DIR:?Need DELPHES_DIR}"

CARD="$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl"
OUTDIR="$HH4B_STORE/root_ak4ak8_smoke"
LOGDIR="$HH4B_STORE/logs/ak4ak8_smoke"

mkdir -p "$OUTDIR" "$LOGDIR"

if [ ! -f "$CARD" ]; then
  echo "Missing card: $CARD"
  exit 1
fi

if [ ! -x "$DELPHES_DIR/DelphesHepMC3" ]; then
  echo "Missing executable: $DELPHES_DIR/DelphesHepMC3"
  exit 1
fi

echo "Using card: $CARD"
echo "Writing to:  $OUTDIR"

run_one () {
  local tag="$1"
  local hepmc="$2"
  local out="$OUTDIR/${tag}_ak4ak8_delphes.root"
  local log="$LOGDIR/${tag}_ak4ak8_delphes.log"

  if [ ! -f "$hepmc" ]; then
    echo "[skip] missing $hepmc"
    return 0
  fi

  echo
  echo "=== $tag ==="
  echo "Input:  $hepmc"
  echo "Output: $out"

  "$DELPHES_DIR/DelphesHepMC3" "$CARD" "$out" "$hepmc" 2>&1 | tee "$log"
}

run_one "ggf_hh4b_smoke20" \
  "$HH4B_STORE/hepmc/HH4b_ggf_loop_sm_hh4b_smoke_20_pythia8.hepmc"

run_one "vbf_hh4b_smoke1000" \
  "$HH4B_STORE/hepmc/HH4b_smoke_vbf_run_03_1000_pythia8.hepmc"

run_one "ttbar_10k" \
  "$HH4B_STORE/hepmc/ttbar_10k_pythia8.hepmc"

run_one "zbbbb_10k" \
  "$HH4B_STORE/hepmc/zbbbb_presel_10k_pythia8.hepmc"

run_one "qcd_bbbb_10k" \
  "$HH4B_STORE/hepmc/qcd_bbbb_presel_10k_pythia8.hepmc"

echo
echo "Done. Outputs:"
ls -lh "$OUTDIR"/*.root
