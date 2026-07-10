#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
CAMPAIGN_BACKUP_DIR="$HH4B_STORE/recovery_backups/qcd_iht_nominal_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$RUN_CARD" ]]; then
  echo "ERROR: missing QCD run card: $RUN_CARD"
  exit 2
fi

cd "$HH4B_REPO"
mkdir -p "$HH4B_STORE/logs" "$HH4B_STORE/root" "$HH4B_STORE/hepmc" "$HH4B_STORE/parquet" "$HH4B_STORE/metadata" "$CAMPAIGN_BACKUP_DIR"


check_storage_guard() {
  local used_gb
  used_gb=$(du -s "$HH4B_STORE" | awk '{printf "%.0f", $1/1024/1024}')
  echo "Current HH4B_STORE usage: ${used_gb}G"
  if (( used_gb > 170 )); then
    echo "ERROR: HH4B_STORE usage exceeded 170G guardrail. Stopping to avoid quota problems."
    df -h "$HH4B_STORE"
  check_storage_guard
    exit 9
  fi
}


echo "PROCESS_DIR=$PROCESS_DIR"
echo "RUN_CARD=$RUN_CARD"
echo "BACKUP_DIR=$CAMPAIGN_BACKUP_DIR"

cp -p "$RUN_CARD" "$CAMPAIGN_BACKUP_DIR/run_card_before_qcd_recovery.dat"
cp -p "$PROCESS_DIR/Cards/proc_card_mg5.dat" "$CAMPAIGN_BACKUP_DIR/proc_card_mg5.dat" || true

restore_run_card() {
  cp "$CAMPAIGN_BACKUP_DIR/run_card_before_qcd_recovery.dat" "$RUN_CARD"
}
trap restore_run_card EXIT

set_qcd_iht_card() {
  local IHTMIN="$1"
  local IHTMAX="$2"

  python3 - <<PY
from pathlib import Path
import re

p = Path("${RUN_CARD}")
txt = p.read_text()

def set_param(text, key, value, comment=""):
    pattern = rf"(?m)^.*=\\s*{re.escape(key)}\\b.*$"
    newline = f" {value} = {key}"
    if comment:
        newline += f" ! {comment}"
    if re.search(pattern, text):
        return re.sub(pattern, newline, text)
    return text + "\\n" + newline + "\\n"

settings = {
    "ptb": ("25.0", "minimum pt for b quarks"),
    "ptbmax": ("-1.0", "maximum pt for b quarks"),
    "etab": ("2.7", "maximum eta for b quarks"),
    "drbb": ("0.4", "minimum deltaR between b quarks"),
    "drbj": ("0.4", "minimum deltaR between b and light jets"),
    "ptj": ("20.0", "minimum pt for jets"),
    "etaj": ("5.0", "maximum eta for jets"),
    "ihtmin": ("${IHTMIN}", "inclusive HT minimum for all partons including b"),
    "ihtmax": ("${IHTMAX}", "inclusive HT maximum for all partons including b"),
    "ht2min": ("0.0", "minimum HT for two leading jets"),
    "ht3min": ("0.0", "minimum HT for three leading jets"),
    "ht4min": ("0.0", "minimum HT for four leading jets"),
    "ht2max": ("-1.0", "maximum HT for two leading jets"),
    "ht3max": ("-1.0", "maximum HT for three leading jets"),
    "ht4max": ("-1.0", "maximum HT for four leading jets"),
}

for key, (value, comment) in settings.items():
    txt = set_param(txt, key, value, comment)

p.write_text(txt)
PY

  echo "Updated QCD run-card slice settings:"
  grep -E "ptb|ptbmax|ihtmin|ihtmax|ht2min|ht3min|ht4min|ht2max|ht3max|ht4max" "$RUN_CARD" | head -80 || true
}

backup_existing_outputs() {
  local TAG="$1"
  for f in \
    "$HH4B_STORE/parquet/${TAG}_event_summary.parquet" \
    "$HH4B_STORE/parquet/${TAG}_hh4b_candidates.parquet" \
    "$HH4B_STORE/metadata/${TAG}_diagnostic.csv" \
    "$HH4B_STORE/metadata/${TAG}_summary.txt"
  do
    if [[ -f "$f" ]]; then
      cp -p "$f" "$CAMPAIGN_BACKUP_DIR/"
    fi
  done
}

run_one_qcd() {
  local TAG="$1"
  local RUN_NAME="$2"
  local N_EVENTS="$3"
  local SEED="$4"
  local IHTMIN="$5"
  local IHTMAX="$6"

  local ROOT_FILE="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"

  echo
  echo "============================================================"
  echo "Recovering QCD ROOT for $TAG"
  echo "RUN_NAME=$RUN_NAME"
  echo "N_EVENTS=$N_EVENTS"
  echo "SEED=$SEED"
  echo "IHTMIN=$IHTMIN"
  echo "IHTMAX=$IHTMAX"
  echo "ROOT_FILE=$ROOT_FILE"
  echo "============================================================"

  if [[ -s "$ROOT_FILE" ]]; then
    echo "ROOT already exists, skipping: $ROOT_FILE"
    return
  fi

  set_qcd_iht_card "$IHTMIN" "$IHTMAX"
  backup_existing_outputs "$TAG"

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$N_EVENTS" \
    "$SEED" \
    "$TAG" \
    no \
    2>&1 | tee "$HH4B_STORE/logs/${TAG}_qcd_root_recovery_pipeline.log"

  # Keep ROOT, delete bulky temporary files.
  rm -f "$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  rm -rf "$PROCESS_DIR/Events/$RUN_NAME"

  if [[ ! -s "$ROOT_FILE" ]]; then
    echo "ERROR: expected ROOT was not produced: $ROOT_FILE"
    exit 3
  fi

  echo "Kept ROOT: $ROOT_FILE"
  df -h "$HH4B_STORE"
  check_storage_guard
}

# Original 20k HT/iHT slices from run_qcd_bbbb_iht_slice_scan.sh.
run_one_qcd "qcd_bbbb_iht100to200_20000" "run_qcd_bbbb_iht100to200_20000" 20000 161000 100 200
run_one_qcd "qcd_bbbb_iht200to400_20000" "run_qcd_bbbb_iht200to400_20000" 20000 162000 200 400
run_one_qcd "qcd_bbbb_iht400to600_20000" "run_qcd_bbbb_iht400to600_20000" 20000 163000 400 600
run_one_qcd "qcd_bbbb_iht600plus_20000"  "run_qcd_bbbb_iht600plus_20000"  20000 164000 600 -1

# Extra 100k iHT200-400 campaign. Shard000 already exists, so this recovers missing shards 001-009.
for i in $(seq 0 9); do
  SHARD=$(printf "%03d" "$i")
  TAG="qcd_bbbb_iht200to400_extra100k_shard${SHARD}"
  RUN_NAME="run_qcd_bbbb_iht200to400_extra100k_${SHARD}"
  SEED=$((252000 + i))
  run_one_qcd "$TAG" "$RUN_NAME" 10000 "$SEED" 200 400
done

echo
echo "DONE recovering nominal QCD iHT ROOT files with ROOT kept."
