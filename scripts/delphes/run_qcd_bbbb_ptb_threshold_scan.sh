#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 N_EVENTS PTB1 [PTB2 PTB3 ...]"
  echo "Example: $0 2000 25 40 60 80"
  exit 1
fi

N_EVENTS="$1"
shift
PTB_VALUES=("$@")

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
BACKUP_RUN_CARD="${RUN_CARD}.before_ptb_scan"
cp "$RUN_CARD" "$BACKUP_RUN_CARD"
restore_run_card() {
  cp "$BACKUP_RUN_CARD" "$RUN_CARD"
}
trap restore_run_card EXIT

cd "$HH4B_REPO"

for PTB in "${PTB_VALUES[@]}"; do
  PTB_TAG=$(echo "$PTB" | sed 's/\./p/g')
  TAG="qcd_bbbb_ptb${PTB_TAG}_${N_EVENTS}"
  RUN_NAME="run_${TAG}"
  SEED=$((120000 + ${PTB_TAG//p/} + N_EVENTS))

  echo
  echo "============================================================"
  echo "QCD bbbb pT threshold scan"
  echo "PTB=$PTB"
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
    "ptb": ("${PTB}", "minimum pt for b quarks; threshold scan"),
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

  echo "Run-card cut settings:"
  grep -E "ptb|etab|drbb|drbj|ptj|etaj" "$RUN_CARD" | head -80 || true

  # Use APPLY_B_CUTS=no so run_local_background_sample does not reset ptb to 25.
  scripts/delphes/run_local_background_sample.sh \
    "$PROCESS_DIR" \
    "$RUN_NAME" \
    "$N_EVENTS" \
    "$SEED" \
    "$TAG" \
    no

  ROOT_FILE="$HH4B_STORE/root/${TAG}_pythia8_delphes.root"
  HEPMC_FILE="$HH4B_STORE/hepmc/${TAG}_pythia8.hepmc"
  LHE_DIR="$PROCESS_DIR/Events/$RUN_NAME"

  # Keep parquet and summary, delete bulky temporary files.
  rm -f "$ROOT_FILE"
  rm -f "$HEPMC_FILE"
  rm -rf "$LHE_DIR"

  echo "Finished and cleaned bulky files for $TAG"
done

python3 - <<'PY'
from pathlib import Path
import os
import re
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
n_events = int("${N_EVENTS}")

rows = []
for cand_path in sorted((store / "parquet").glob(f"qcd_bbbb_ptb*_{n_events}_hh4b_candidates.parquet")):
    tag = cand_path.name.replace("_hh4b_candidates.parquet", "")
    event_path = store / "parquet" / f"{tag}_event_summary.parquet"
    if not event_path.exists():
        continue

    m = re.search(r"ptb([0-9p]+)_", tag)
    ptb = m.group(1).replace("p", ".") if m else "unknown"

    ev = pd.read_parquet(event_path)
    cand = pd.read_parquet(cand_path)

    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    n_cand = len(cand)
    n_ge4j = int((ev["n_jet_pt30_eta25"] >= 4).sum())
    n_ge4b = int((ev["n_bjet_pt30_eta25"] >= 4).sum())

    rows.append({
        "tag": tag,
        "ptb_min": float(ptb),
        "n_generated": n_gen,
        "xsec_pb": xsec,
        "n_ge4_jets": n_ge4j,
        "n_ge4_btags": n_ge4b,
        "candidate_rows": n_cand,
        "candidate_eff": n_cand / n_gen if n_gen else 0.0,
        "effective_candidate_xsec_pb": xsec * n_cand / n_gen if n_gen else 0.0,
        "median_mbb1": cand["mbb1"].median() if n_cand else None,
        "median_mbb2": cand["mbb2"].median() if n_cand else None,
        "median_avg_mbb": cand["avg_mbb"].median() if n_cand else None,
        "median_mhh": cand["mhh"].median() if n_cand else None,
    })

df = pd.DataFrame(rows).sort_values("ptb_min")
out = store / "metadata" / f"qcd_bbbb_ptb_threshold_scan_{n_events}.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("Wrote:", out)
PY
