#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"
: "${HH4B_DELPHES_CARD:?Need HH4B_DELPHES_CARD}"

cd "$HH4B_REPO"

mkdir -p "$HH4B_STORE/logs/ak4ak8_bkg50k_v1"
mkdir -p "$HH4B_STORE/metadata/ak4ak8_bkg50k_v1"

check_storage() {
  USED_GB=$(du -s --block-size=1G "$HH4B_STORE" | awk '{print $1}' | sed 's/G//')
  echo "Storage used: ${USED_GB}G"
  quota -s || true

  if [[ "$USED_GB" -ge 190 ]]; then
    echo "ERROR: storage is at or above 190G; stopping to avoid quota problems."
    exit 20
  fi
}

run_one() {
  local PROCESS_DIR="$1"
  local CAMPAIGN="$2"
  local SHARD="$3"
  local SEED="$4"

  local TAG="${CAMPAIGN}_shard${SHARD}"
  local RUN_NAME="run_${CAMPAIGN}_${SHARD}"
  local LOG="$HH4B_STORE/logs/ak4ak8_bkg50k_v1/${TAG}_outer.log"
  local ROOT="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"
  local CAND="$HH4B_STORE/parquet/${TAG}_hh4b_candidates.parquet"

  echo
  echo "============================================================"
  echo "Running $TAG"
  echo "PROCESS_DIR=$PROCESS_DIR"
  echo "RUN_NAME=$RUN_NAME"
  echo "SEED=$SEED"
  echo "CARD=$HH4B_DELPHES_CARD"
  echo "ROOT=$ROOT"
  echo "============================================================"

  check_storage

  if [[ -s "$ROOT" && -s "$CAND" ]]; then
    echo "Already complete, skipping: $TAG"
    return 0
  fi

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    10000 \
    "$SEED" \
    "$TAG" \
    yes \
    2>&1 | tee "$LOG"

  if [[ ! -s "$ROOT" ]]; then
    echo "ERROR: missing ROOT after run: $ROOT"
    exit 30
  fi

  if [[ ! -s "$CAND" ]]; then
    echo "ERROR: missing candidate parquet after run: $CAND"
    exit 31
  fi

  echo "DONE $TAG"
  check_storage
}

echo "Starting AK4/AK8 QCD/Zbbbb extra50k v1"
echo "HH4B_DELPHES_CARD=$HH4B_DELPHES_CARD"
grep -n "set ParameterR" "$HH4B_DELPHES_CARD"

# 5 shards × 10k = 50k QCD bbbb
for i in 0 1 2 3 4; do
  SHARD=$(printf "%03d" "$i")
  SEED=$((717000 + i))
  run_one "$HH4B_STORE/mg5/QCD_bbbb_presel_smoke" "qcd_bbbb_ak4ak8_extra50k_v1" "$SHARD" "$SEED"
done

# 5 shards × 10k = 50k Zbbbb
for i in 0 1 2 3 4; do
  SHARD=$(printf "%03d" "$i")
  SEED=$((718000 + i))
  run_one "$HH4B_STORE/mg5/Zbbbb_presel_smoke" "zbbbb_ak4ak8_extra50k_v1" "$SHARD" "$SEED"
done

echo
echo "DONE all AK4/AK8 QCD/Zbbbb extra50k v1"
du -sh "$HH4B_STORE"
quota -s || true
