#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 N_EVENTS"
  echo "Runs first-pass non-overlapping QCD bbbb ptb slices:"
  echo "  25-40, 40-60, 60-80, >80 GeV"
  exit 1
fi

N_EVENTS="$1"

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
BACKUP_RUN_CARD="${RUN_CARD}.before_ptb_slice_scan"

cp "$RUN_CARD" "$BACKUP_RUN_CARD"
restore_run_card() {
  cp "$BACKUP_RUN_CARD" "$RUN_CARD"
}
trap restore_run_card EXIT

cd "$HH4B_REPO"

# Format: label ptb ptbmax seed_offset
SLICES=(
  "ptb25to40 25 40 131000"
  "ptb40to60 40 60 132000"
  "ptb60to80 60 80 133000"
  "ptb80plus 80 -1 134000"
)

for entry in "${SLICES[@]}"; do
  read -r LABEL PTB PTBMAX SEED0 <<< "$entry"
  TAG="qcd_bbbb_${LABEL}_${N_EVENTS}"
  RUN_NAME="run_${TAG}"
  SEED=$((SEED0 + N_EVENTS))

  echo
  echo "============================================================"
  echo "QCD bbbb ptb slice"
  echo "LABEL=$LABEL"
  echo "PTB=$PTB"
  echo "PTBMAX=$PTBMAX"
  echo "TAG=$TAG"
  echo "RUN_NAME=$RUN_NAME"
  echo "SEED=$SEED"
  echo "============================================================"

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
        text = re.sub(pattern, newline, text)
    else:
        text += "\\n" + newline + "\\n"
    return text

settings = {
    "ptb": ("${PTB}", "minimum pt for b quarks; ptb slice"),
    "ptbmax": ("${PTBMAX}", "maximum pt for b quarks; ptb slice"),
    "etab": ("2.7", "maximum eta for b quarks"),
    "drbb": ("0.4", "minimum deltaR between b quarks"),
    "drbj": ("0.4", "minimum deltaR between b and light jets"),
    "ptj": ("20.0", "minimum pt for jets"),
    "etaj": ("5.0", "maximum eta for jets"),
}

for key, (value, comment) in settings.items():
    txt = set_param(txt, key, value, comment)

p.write_text(txt)
PY

  grep -E "ptb|ptbmax|etab|drbb|drbj|ptj|etaj" "$RUN_CARD" | head -80 || true

  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$N_EVENTS" \
    "$SEED" \
    "$TAG" \
    no

  rm -f "$HH4B_STORE/root/${TAG}_pythia8_delphes.root"
  rm -f "$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  rm -rf "$PROCESS_DIR/Events/$RUN_NAME"

  echo "Finished and cleaned bulky files for $TAG"
done

QCD_SLICE_N_EVENTS="$N_EVENTS" python3 - <<'PY'
from pathlib import Path
import os
import re
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
n_events = int(os.environ["QCD_SLICE_N_EVENTS"])

rows = []
for event_path in sorted((store / "parquet").glob(f"qcd_bbbb_ptb*_{n_events}_event_summary.parquet")):
    tag = event_path.name.replace("_event_summary.parquet", "")
    cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"
    if not cand_path.exists():
        continue

    ev = pd.read_parquet(event_path)
    cand = pd.read_parquet(cand_path)

    label = tag.replace(f"qcd_bbbb_", "").replace(f"_{n_events}", "")

    rows.append({
        "tag": tag,
        "slice_label": label,
        "n_generated": len(ev),
        "xsec_pb": float(ev["event_cross_section_pb"].median()),
        "n_ge4_jets": int((ev["n_jet_pt30_eta25"] >= 4).sum()),
        "n_ge4_btags": int((ev["n_bjet_pt30_eta25"] >= 4).sum()),
        "candidate_rows": len(cand),
        "candidate_eff": len(cand) / len(ev) if len(ev) else 0,
        "effective_candidate_xsec_pb": float(ev["event_cross_section_pb"].median()) * len(cand) / len(ev) if len(ev) else 0,
        "median_mbb1": cand["mbb1"].median() if len(cand) else None,
        "median_mbb2": cand["mbb2"].median() if len(cand) else None,
        "median_avg_mbb": cand["avg_mbb"].median() if len(cand) else None,
        "median_delta_mbb": cand["delta_mbb"].median() if len(cand) else None,
        "median_mhh": cand["mhh"].median() if len(cand) else None,
    })

df = pd.DataFrame(rows)
out = store / "metadata" / f"qcd_bbbb_ptb_slice_scan_{n_events}.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("Wrote:", out)
PY
