#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

PROCESS_DIR="$HH4B_STORE/mg5/Zbbbb_presel_smoke"
CAMPAIGN="zbbbb_presel_100k"

if [[ ! -f "$PROCESS_DIR/Cards/run_card.dat" ]]; then
  echo "ERROR: PROCESS_DIR does not look like an MG5 process dir: $PROCESS_DIR"
  exit 2
fi

cd "$HH4B_REPO"

mkdir -p "$HH4B_STORE/logs" "$HH4B_STORE/root" "$HH4B_STORE/hepmc" "$HH4B_STORE/parquet" "$HH4B_STORE/metadata"
BACKUP_DIR="$HH4B_STORE/recovery_backups/zbbbb_100k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "PROCESS_DIR=$PROCESS_DIR"
echo "CAMPAIGN=$CAMPAIGN"
echo "BACKUP_DIR=$BACKUP_DIR"

cp -p "$PROCESS_DIR/Cards/run_card.dat" "$BACKUP_DIR/run_card_before_recovery.dat"
cp -p "$PROCESS_DIR/Cards/proc_card_mg5.dat" "$BACKUP_DIR/proc_card_mg5.dat" || true

check_storage_guard() {
  local used_gb
  used_gb=$(du -s "$HH4B_STORE" | awk '{printf "%.0f", $1/1024/1024}')
  echo "Current HH4B_STORE usage: ${used_gb}G"
  if (( used_gb > 170 )); then
    echo "ERROR: HH4B_STORE usage exceeded 170G guardrail. Stopping."
    df -h "$HH4B_STORE"
    exit 9
  fi
}

for i in $(seq 1 9); do
  SHARD=$(printf "%03d" "$i")
  TAG="${CAMPAIGN}_shard${SHARD}"
  RUN_NAME="run_${CAMPAIGN}_${SHARD}"
  SEED=$((104000 + i))
  ROOT_FILE="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"

  echo
  echo "============================================================"
  echo "Recovering ROOT for $TAG"
  echo "RUN_NAME=$RUN_NAME"
  echo "SEED=$SEED"
  echo "ROOT_FILE=$ROOT_FILE"
  echo "============================================================"

  if [[ -s "$ROOT_FILE" ]]; then
    echo "ROOT already exists, skipping: $ROOT_FILE"
    continue
  fi

  for f in \
    "$HH4B_STORE/parquet/${TAG}_event_summary.parquet" \
    "$HH4B_STORE/parquet/${TAG}_hh4b_candidates.parquet" \
    "$HH4B_STORE/metadata/${TAG}_diagnostic.csv" \
    "$HH4B_STORE/metadata/${TAG}_summary.txt"
  do
    if [[ -f "$f" ]]; then
      cp -p "$f" "$BACKUP_DIR/"
    fi
  done

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    10000 \
    "$SEED" \
    "$TAG" \
    no \
    2>&1 | tee "$HH4B_STORE/logs/${TAG}_root_recovery_pipeline.log"

  rm -f "$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  rm -rf "$PROCESS_DIR/Events/$RUN_NAME"

  if [[ ! -s "$ROOT_FILE" ]]; then
    echo "ERROR: expected ROOT was not produced: $ROOT_FILE"
    exit 3
  fi

  echo "Kept ROOT: $ROOT_FILE"
  df -h "$HH4B_STORE"
  check_storage_guard
done

echo
echo "DONE recovering Zbbbb_100k ROOT shards 001-009"
