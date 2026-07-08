#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 N_EVENTS"
  echo "Runs first-pass non-overlapping QCD bbbb inclusive-HT slices:"
  echo "  100-200, 200-400, 400-600, >600 GeV"
  exit 1
fi

N_EVENTS="$1"

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
BACKUP_RUN_CARD="${RUN_CARD}.before_iht_slice_scan"

cp "$RUN_CARD" "$BACKUP_RUN_CARD"
restore_run_card() {
  cp "$BACKUP_RUN_CARD" "$RUN_CARD"
}
trap restore_run_card EXIT

cd "$HH4B_REPO"

# Format: label ihtmin ihtmax seed_offset
# Baseline ptb remains 25 GeV. These are inclusive HT slices over all partons, including b quarks.
SLICES=(
  "iht100to200 100 200 141000"
  "iht200to400 200 400 142000"
  "iht400to600 400 600 143000"
  "iht600plus 600 -1 144000"
)

for entry in "${SLICES[@]}"; do
  read -r LABEL IHTMIN IHTMAX SEED0 <<< "$entry"
  TAG="qcd_bbbb_${LABEL}_${N_EVENTS}"
  RUN_NAME="run_${TAG}"
  SEED=$((SEED0 + N_EVENTS))

  echo
  echo "============================================================"
  echo "QCD bbbb inclusive-HT slice"
  echo "LABEL=$LABEL"
  echo "IHTMIN=$IHTMIN"
  echo "IHTMAX=$IHTMAX"
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

  grep -E "ptb|ptbmax|ihtmin|ihtmax|ht2min|ht3min|ht4min|ht2max|ht3max|ht4max" "$RUN_CARD" | head -80 || true

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

QCD_IHT_N_EVENTS="$N_EVENTS" python3 - <<'PY'
from pathlib import Path
import os
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
n_events = int(os.environ["QCD_IHT_N_EVENTS"])

rows = []
for event_path in sorted((store / "parquet").glob(f"qcd_bbbb_iht*_{n_events}_event_summary.parquet")):
    tag = event_path.name.replace("_event_summary.parquet", "")
    cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"
    if not cand_path.exists():
        continue

    ev = pd.read_parquet(event_path)
    cand = pd.read_parquet(cand_path)
    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    n_cand = len(cand)

    rows.append({
        "tag": tag,
        "slice_label": tag.replace("qcd_bbbb_", "").replace(f"_{n_events}", ""),
        "n_generated": n_gen,
        "xsec_pb": xsec,
        "n_ge4_jets": int((ev["n_jet_pt30_eta25"] >= 4).sum()),
        "n_ge4_btags": int((ev["n_bjet_pt30_eta25"] >= 4).sum()),
        "candidate_rows": n_cand,
        "candidate_eff": n_cand / n_gen if n_gen else 0.0,
        "effective_candidate_xsec_pb": xsec * n_cand / n_gen if n_gen else 0.0,
        "median_mbb1": cand["mbb1"].median() if n_cand else None,
        "median_mbb2": cand["mbb2"].median() if n_cand else None,
        "median_avg_mbb": cand["avg_mbb"].median() if n_cand else None,
        "median_delta_mbb": cand["delta_mbb"].median() if n_cand else None,
        "median_mhh": cand["mhh"].median() if n_cand else None,
        "median_ht_pt30_eta25": ev["ht_pt30_eta25"].median() if "ht_pt30_eta25" in ev.columns else None,
    })

df = pd.DataFrame(rows)
out = store / "metadata" / f"qcd_bbbb_iht_slice_scan_{n_events}.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("Wrote:", out)
PY
