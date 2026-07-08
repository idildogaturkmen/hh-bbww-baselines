#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 PROCESS_DIR RUN_NAME N_EVENTS SEED OUTPUT_TAG APPLY_B_CUTS"
  echo "Example: $0 \$HH4B_STORE/mg5/QCD_bbbb_presel_smoke run_03_10000 10000 93003 qcd_bbbb_presel_10k yes"
  exit 1
fi

PROCESS_DIR="$1"
RUN_NAME="$2"
N_EVENTS="$3"
SEED="$4"
OUTPUT_TAG="$5"
APPLY_B_CUTS="$6"

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

echo "PROCESS_DIR=$PROCESS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "N_EVENTS=$N_EVENTS"
echo "SEED=$SEED"
echo "OUTPUT_TAG=$OUTPUT_TAG"
echo "APPLY_B_CUTS=$APPLY_B_CUTS"

cd "$PROCESS_DIR"

python3 - <<PY
from pathlib import Path
import re

p = Path("Cards/run_card.dat")
txt = p.read_text()

def set_param(text, key, value, comment=""):
    pattern = rf"(?m)^.*=\\s*{re.escape(key)}\\b.*$"
    newline = f" {value} = {key}"
    if comment:
        newline += f" ! {comment}"
    if re.search(pattern, text):
        text = re.sub(pattern, newline, text)
    else:
        text += "\\n" + newline + "\\n"
    return text

settings = {
    "nevents": ("${N_EVENTS}", "Number of unweighted events"),
    "iseed": ("${SEED}", "Random seed"),
}

if "${APPLY_B_CUTS}".lower() in ("yes", "true", "1"):
    settings.update({
        "ptb": ("25.0", "minimum pt for b quarks"),
        "etab": ("2.7", "maximum eta for b quarks"),
        "drbb": ("0.4", "minimum deltaR between b quarks"),
        "drbj": ("0.4", "minimum deltaR between b and light jets"),
        "ptj": ("20.0", "minimum pt for jets"),
        "etaj": ("5.0", "maximum eta for jets"),
    })

for key, (value, comment) in settings.items():
    txt = set_param(txt, key, value, comment)

p.write_text(txt)
PY

echo "Updated run card:"
grep -E "nevents|iseed|ptb|etab|drbb|drbj|ptj|etaj" Cards/run_card.dat | head -80 || true

LOG="$HH4B_STORE/logs/${OUTPUT_TAG}_generate_events.log"
./bin/generate_events "$RUN_NAME" -f | tee "$LOG"

LHE="$PROCESS_DIR/Events/$RUN_NAME/unweighted_events.lhe.gz"
if [[ ! -s "$LHE" ]]; then
  echo "ERROR: missing LHE: $LHE"
  exit 2
fi

N_FOUND=$(zgrep -c "<event>" "$LHE")
echo "LHE=$LHE"
echo "N_FOUND=$N_FOUND"

if [[ "$N_FOUND" != "$N_EVENTS" ]]; then
  echo "ERROR: expected $N_EVENTS events but found $N_FOUND"
  exit 3
fi

cd "$HH4B_REPO"

scripts/delphes/run_existing_lhe_pythia_delphes.sh \
  "$OUTPUT_TAG" \
  "$LHE" \
  "$N_EVENTS" \
  "$OUTPUT_TAG" \
  | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_pipeline.log"

ROOT="$HH4B_STORE/root/${OUTPUT_TAG}_pythia8_delphes.root"
CAND="$HH4B_STORE/parquet/${OUTPUT_TAG}_hh4b_candidates.parquet"

python3 scripts/delphes/reconstruct_hh4b_candidates.py \
  --input "$ROOT" \
  --out "$CAND" \
  --sample "$OUTPUT_TAG" \
  | tee "$HH4B_STORE/logs/${OUTPUT_TAG}_hh4b_candidates.log"

python3 - <<PY
from pathlib import Path
import os
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
tag = "${OUTPUT_TAG}"

event_path = store / "parquet" / f"{tag}_event_summary.parquet"
cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"

ev = pd.read_parquet(event_path)
cand = pd.read_parquet(cand_path)

lines = []
lines.append(f"sample: {tag}")
lines.append(f"events: {len(ev)}")
lines.append(f"candidate rows: {len(cand)}")
lines.append(f"cross-section median pb: {ev['event_cross_section_pb'].median()}")
lines.append(f"events with >=4 selected jets: {int((ev['n_jet_pt30_eta25'] >= 4).sum())}")
lines.append(f"events with >=4 selected b-tagged jets: {int((ev['n_bjet_pt30_eta25'] >= 4).sum())}")

if len(cand):
    for c in ["mbb1","mbb2","avg_mbb","delta_mbb","mhh","n_selected_bjets"]:
        lines.append(f"median {c}: {cand[c].median()}")

out = store / "metadata" / f"{tag}_summary.txt"
out.write_text("\\n".join(lines) + "\\n")
print("\\n".join(lines))
print("Wrote:", out)
PY

echo "DONE $OUTPUT_TAG"
