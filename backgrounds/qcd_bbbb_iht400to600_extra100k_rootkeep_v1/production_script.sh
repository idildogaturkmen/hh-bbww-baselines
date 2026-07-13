#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

cd "$HH4B_REPO"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
BACKUP_RUN_CARD="${RUN_CARD}.before_iht400to600_extra100k_rootkeep_v1"

CAMPAIGN="qcd_bbbb_iht400to600_extra100k_rootkeep_v1"
N_SHARDS=10
EVENTS_PER_SHARD=10000
SEED0=852000

cp "$RUN_CARD" "$BACKUP_RUN_CARD"

restore_run_card() {
  cp "$BACKUP_RUN_CARD" "$RUN_CARD"
}
trap restore_run_card EXIT

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

    "ihtmin": ("400", "inclusive HT minimum for all partons including b"),
    "ihtmax": ("600", "inclusive HT maximum for all partons including b"),

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

echo "Using QCD bbbb iHT400to600 run-card settings:"
grep -E "ptb|ptbmax|ihtmin|ihtmax|ht2min|ht3min|ht4min|ht2max|ht3max|ht4max" "$RUN_CARD" | head -80 || true

for IDX in $(seq 0 $((N_SHARDS - 1))); do
  SHARD=$(printf "%03d" "$IDX")
  RUN_NAME="run_${CAMPAIGN}_${SHARD}"
  TAG="${CAMPAIGN}_shard${SHARD}"
  SEED=$((SEED0 + IDX))

  echo
  echo "=== Starting shard ${SHARD}/${N_SHARDS} seed=${SEED} ==="

  scripts/delphes/run_local_background_sample_v2.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$EVENTS_PER_SHARD" \
    "$SEED" \
    "$TAG" \
    yes

  echo "=== Finished shard ${SHARD} ==="
done

python3 - <<PY
import os
from pathlib import Path
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
campaign = "${CAMPAIGN}"

event_paths = sorted((store / "parquet").glob(f"{campaign}_shard*_event_summary.parquet"))
cand_paths = sorted((store / "parquet").glob(f"{campaign}_shard*_hh4b_candidates.parquet"))

if not event_paths:
    raise RuntimeError("No event summary parquet files found")
if not cand_paths:
    raise RuntimeError("No candidate parquet files found")

events = pd.concat([pd.read_parquet(p) for p in event_paths], ignore_index=True)
cands = pd.concat([pd.read_parquet(p) for p in cand_paths], ignore_index=True)

eps = 1e-9
if len(cands):
    cands["pt_sum4"] = cands["j1_pt"] + cands["j2_pt"] + cands["j3_pt"] + cands["j4_pt"]
    cands["ht_over_mhh"] = cands["ht_candidate_jets"] / np.maximum(cands["mhh"], eps)
    cands["pt_asym_12"] = np.abs(cands["j1_pt"] - cands["j2_pt"]) / np.maximum(cands["j1_pt"] + cands["j2_pt"], eps)
    cands["pt_asym_34"] = np.abs(cands["j3_pt"] - cands["j4_pt"]) / np.maximum(cands["j3_pt"] + cands["j4_pt"], eps)
    cands["avg_drbb"] = 0.5 * (cands["drbb1"] + cands["drbb2"])
    cands["max_drbb"] = np.maximum(cands["drbb1"], cands["drbb2"])
    cands["min_drbb"] = np.minimum(cands["drbb1"], cands["drbb2"])
    cands["analysis_sample"] = campaign

events_out = store / "parquet" / f"{campaign}_merged_event_summary.parquet"
cands_out = store / "parquet" / f"{campaign}_merged_hh4b_candidates_bdtv2_ready.parquet"

events.to_parquet(events_out, index=False)
cands.to_parquet(cands_out, index=False)

lines = []
lines.append(f"campaign: {campaign}")
lines.append(f"shards: ${N_SHARDS}")
lines.append(f"events_per_shard: ${EVENTS_PER_SHARD}")
lines.append(f"total generated events: {len(events)}")
lines.append(f"candidate rows: {len(cands)}")
lines.append(f"events with >=4 selected jets: {int((events['n_jet_pt30_eta25'] >= 4).sum())}")
lines.append(f"events with >=4 selected b-tagged jets: {int((events['n_bjet_pt30_eta25'] >= 4).sum())}")
lines.append(f"median xsec pb: {events['event_cross_section_pb'].median()}")

if len(cands):
    for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]:
        lines.append(f"median {col}: {cands[col].median()}")

lines.append(f"merged event summary: {events_out}")
lines.append(f"merged candidates: {cands_out}")

out = store / "metadata" / f"{campaign}_summary.txt"
out.write_text("\\n".join(lines) + "\\n")

print("\\n".join(lines))
print("Wrote:", out)
PY

echo
echo "DONE ${CAMPAIGN}"
