#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 CAMPAIGN PROCESS_DIR N_SHARDS EVENTS_PER_SHARD SEED0 APPLY_B_CUTS KEEP_ROOT_SHARDS DELETE_LHE"
  echo "Example:"
  echo "  $0 qcd_bbbb_presel_100k \$HH4B_STORE/mg5/QCD_bbbb_presel_smoke 10 10000 103000 yes 1 yes"
  exit 1
fi

CAMPAIGN="$1"
PROCESS_DIR="$2"
N_SHARDS="$3"
EVENTS_PER_SHARD="$4"
SEED0="$5"
APPLY_B_CUTS="$6"
KEEP_ROOT_SHARDS="$7"
DELETE_LHE="$8"

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

cd "$HH4B_REPO"

MANIFEST="$HH4B_STORE/metadata/${CAMPAIGN}_manifest.csv"
SUMMARY="$HH4B_STORE/metadata/${CAMPAIGN}_summary.txt"

echo "campaign,shard,tag,run_name,seed,events_per_shard,event_parquet,candidate_parquet,root_kept,hepmc_deleted,lhe_deleted" > "$MANIFEST"

for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD=$(printf "%03d" "$i")
  SEED=$((SEED0 + i))
  TAG="${CAMPAIGN}_shard${SHARD}"
  RUN_NAME="run_${CAMPAIGN}_${SHARD}"

  echo
  echo "============================================================"
  echo "Starting shard $SHARD / campaign $CAMPAIGN"
  echo "TAG=$TAG"
  echo "RUN_NAME=$RUN_NAME"
  echo "SEED=$SEED"
  echo "============================================================"

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$EVENTS_PER_SHARD" \
    "$SEED" \
    "$TAG" \
    "$APPLY_B_CUTS"

  ROOT_FILE="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"
  HEPMC_FILE="$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  EVENT_PARQUET="$HH4B_STORE/parquet/${TAG}_event_summary.parquet"
  CAND_PARQUET="$HH4B_STORE/parquet/${TAG}_hh4b_candidates.parquet"
  LHE_DIR="$PROCESS_DIR/Events/$RUN_NAME"

  ROOT_KEPT="yes"
  if (( i >= KEEP_ROOT_SHARDS )); then
    rm -f "$ROOT_FILE"
    ROOT_KEPT="no"
  fi

  HEPMC_DELETED="no"
  if [[ -f "$HEPMC_FILE" ]]; then
    rm -f "$HEPMC_FILE"
    HEPMC_DELETED="yes"
  fi

  LHE_DELETED="no"
  if [[ "$DELETE_LHE" == "yes" && -d "$LHE_DIR" ]]; then
    rm -rf "$LHE_DIR"
    LHE_DELETED="yes"
  fi

  echo "$CAMPAIGN,$SHARD,$TAG,$RUN_NAME,$SEED,$EVENTS_PER_SHARD,$EVENT_PARQUET,$CAND_PARQUET,$ROOT_KEPT,$HEPMC_DELETED,$LHE_DELETED" >> "$MANIFEST"

  echo "Finished shard $SHARD"
done

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

if len(event_files) != n_shards:
    raise RuntimeError(f"Expected {n_shards} event parquet files, found {len(event_files)}")
if len(cand_files) != n_shards:
    raise RuntimeError(f"Expected {n_shards} candidate parquet files, found {len(cand_files)}")

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
lines.append(f"events with >=4 selected jets: {int((events['n_jet_pt30_eta25'] >= 4).sum())}")
lines.append(f"events with >=4 selected b-tagged jets: {int((events['n_bjet_pt30_eta25'] >= 4).sum())}")
lines.append(f"median xsec pb: {events['event_cross_section_pb'].median()}")

if len(cands):
    for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]:
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
