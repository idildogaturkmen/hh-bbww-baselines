#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

cd "$HH4B_REPO"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"

CAMPAIGN="qcd_bbbb_iht200to400_extra200k_rootkeep_v1"
N_SHARDS=20
EVENTS_PER_SHARD=10000
SEED0=852000

BACKUP_DIR="$HH4B_STORE/recovery_backups/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR" "$HH4B_STORE/logs" "$HH4B_STORE/root" "$HH4B_STORE/parquet" "$HH4B_STORE/metadata" "$HH4B_STORE/hepmc"

if [[ ! -f "$RUN_CARD" ]]; then
  echo "ERROR: missing run card: $RUN_CARD"
  exit 2
fi

cp -p "$RUN_CARD" "$BACKUP_DIR/run_card_before_${CAMPAIGN}.dat"
cp -p "$PROCESS_DIR/Cards/proc_card_mg5.dat" "$BACKUP_DIR/proc_card_mg5.dat" || true

restore_run_card() {
  cp "$BACKUP_DIR/run_card_before_${CAMPAIGN}.dat" "$RUN_CARD"
}
trap restore_run_card EXIT

used_gb() {
  du -s "$HH4B_STORE" | awk '{printf "%.0f", $1/1024/1024}'
}

check_storage_guard() {
  local used
  used=$(used_gb)
  echo "Current HH4B_STORE usage: ${used}G"
  df -h "$HH4B_STORE"
  if (( used > 195 )); then
    echo "ERROR: HH4B_STORE usage exceeded 195G guardrail. Stopping before hitting quota."
    exit 9
  fi
}

set_qcd_iht200to400_card() {
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
    "ihtmin": ("200", "inclusive HT minimum for all partons including b"),
    "ihtmax": ("400", "inclusive HT maximum for all partons including b"),
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

  echo "Using QCD bbbb iHT 200-400 run-card settings:"
  grep -E "ptb|ptbmax|ihtmin|ihtmax|ht2min|ht3min|ht4min|ht2max|ht3max|ht4max" "$RUN_CARD" | head -80 || true
}

MANIFEST="$HH4B_STORE/metadata/${CAMPAIGN}_manifest.csv"
SUMMARY="$HH4B_STORE/metadata/${CAMPAIGN}_summary.txt"

echo "campaign,shard,tag,run_name,seed,events_per_shard,root_file,event_parquet,candidate_parquet,root_kept,hepmc_deleted,lhe_deleted,status" > "$MANIFEST"

echo "============================================================"
echo "Campaign: $CAMPAIGN"
echo "N_SHARDS=$N_SHARDS"
echo "EVENTS_PER_SHARD=$EVENTS_PER_SHARD"
echo "SEED0=$SEED0"
echo "PROCESS_DIR=$PROCESS_DIR"
echo "RUN_CARD=$RUN_CARD"
echo "BACKUP_DIR=$BACKUP_DIR"
echo "============================================================"

check_storage_guard
set_qcd_iht200to400_card

for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD=$(printf "%03d" "$i")
  SEED=$((SEED0 + i))
  TAG="${CAMPAIGN}_shard${SHARD}"
  RUN_NAME="run_${CAMPAIGN}_${SHARD}"

  ROOT_FILE="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"
  HEPMC_FILE="$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  EVENT_PARQUET="$HH4B_STORE/parquet/${TAG}_event_summary.parquet"
  CAND_PARQUET="$HH4B_STORE/parquet/${TAG}_hh4b_candidates.parquet"
  LHE_DIR="$PROCESS_DIR/Events/$RUN_NAME"

  echo
  echo "============================================================"
  echo "Shard $SHARD / $CAMPAIGN"
  echo "TAG=$TAG"
  echo "RUN_NAME=$RUN_NAME"
  echo "SEED=$SEED"
  echo "ROOT_FILE=$ROOT_FILE"
  echo "============================================================"

  check_storage_guard

  if [[ -s "$ROOT_FILE" && -s "$EVENT_PARQUET" && -s "$CAND_PARQUET" ]]; then
    echo "All outputs already exist, skipping shard $SHARD."
    echo "$CAMPAIGN,$SHARD,$TAG,$RUN_NAME,$SEED,$EVENTS_PER_SHARD,$ROOT_FILE,$EVENT_PARQUET,$CAND_PARQUET,yes,unknown,unknown,skipped_existing" >> "$MANIFEST"
    continue
  fi

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$EVENTS_PER_SHARD" \
    "$SEED" \
    "$TAG" \
    no \
    2>&1 | tee "$HH4B_STORE/logs/${TAG}_pipeline.log"

  if [[ ! -s "$ROOT_FILE" ]]; then
    echo "ERROR: ROOT file was not produced: $ROOT_FILE"
    exit 3
  fi

  if [[ ! -s "$EVENT_PARQUET" ]]; then
    echo "ERROR: event parquet was not produced: $EVENT_PARQUET"
    exit 4
  fi

  if [[ ! -s "$CAND_PARQUET" ]]; then
    echo "ERROR: candidate parquet was not produced: $CAND_PARQUET"
    exit 5
  fi

  HEPMC_DELETED="no"
  if [[ -f "$HEPMC_FILE" ]]; then
    rm -f "$HEPMC_FILE"
    HEPMC_DELETED="yes"
  fi

  LHE_DELETED="no"
  if [[ -d "$LHE_DIR" ]]; then
    rm -rf "$LHE_DIR"
    LHE_DELETED="yes"
  fi

  echo "$CAMPAIGN,$SHARD,$TAG,$RUN_NAME,$SEED,$EVENTS_PER_SHARD,$ROOT_FILE,$EVENT_PARQUET,$CAND_PARQUET,yes,$HEPMC_DELETED,$LHE_DELETED,done" >> "$MANIFEST"

  echo "Kept ROOT: $ROOT_FILE"
  echo "Kept event parquet: $EVENT_PARQUET"
  echo "Kept candidate parquet: $CAND_PARQUET"
  echo "Deleted transient HEPMC: $HEPMC_DELETED"
  echo "Deleted transient LHE dir: $LHE_DELETED"
done

echo
echo "============================================================"
echo "Merging campaign parquets"
echo "============================================================"

python3 - <<PY
from pathlib import Path
import os
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
campaign = "${CAMPAIGN}"
n_shards = int("${N_SHARDS}")
events_per_shard = int("${EVENTS_PER_SHARD}")

event_files = sorted((store / "parquet").glob(f"{campaign}_shard*_event_summary.parquet"))
cand_files = sorted((store / "parquet").glob(f"{campaign}_shard*_hh4b_candidates.parquet"))
root_files = sorted((store / "root").glob(f"{campaign}_shard*_pythia8_delphes.root"))

if len(event_files) != n_shards:
    raise RuntimeError(f"Expected {n_shards} event parquet files, found {len(event_files)}")
if len(cand_files) != n_shards:
    raise RuntimeError(f"Expected {n_shards} candidate parquet files, found {len(cand_files)}")
if len(root_files) != n_shards:
    raise RuntimeError(f"Expected {n_shards} ROOT files, found {len(root_files)}")

events = pd.concat([pd.read_parquet(p) for p in event_files], ignore_index=True)
cands = pd.concat([pd.read_parquet(p) for p in cand_files], ignore_index=True)

event_out = store / "parquet" / f"{campaign}_merged_event_summary.parquet"
cand_out = store / "parquet" / f"{campaign}_merged_hh4b_candidates.parquet"

events.to_parquet(event_out, index=False)
cands.to_parquet(cand_out, index=False)

lines = []
lines.append(f"campaign: {campaign}")
lines.append(f"shards: {n_shards}")
lines.append(f"events_per_shard: {events_per_shard}")
lines.append(f"total generated events: {len(events)}")
lines.append(f"candidate rows: {len(cands)}")
lines.append(f"root files kept: {len(root_files)}")
lines.append(f"events with >=4 selected jets: {int((events['n_jet_pt30_eta25'] >= 4).sum())}")
lines.append(f"events with >=4 selected b-tagged jets: {int((events['n_bjet_pt30_eta25'] >= 4).sum())}")

if "event_cross_section_pb" in events.columns:
    lines.append(f"median xsec pb: {events['event_cross_section_pb'].median()}")

if len(cands):
    for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]:
        if col in cands.columns:
            lines.append(f"median {col}: {cands[col].median()}")

lines.append(f"merged event summary: {event_out}")
lines.append(f"merged candidates: {cand_out}")

summary = store / "metadata" / f"{campaign}_summary.txt"
summary.write_text("\\n".join(lines) + "\\n")

print("\\n".join(lines))
print("Wrote:", summary)
PY

echo
echo "DONE campaign $CAMPAIGN"
echo "Manifest: $MANIFEST"
echo "Summary: $SUMMARY"
check_storage_guard
