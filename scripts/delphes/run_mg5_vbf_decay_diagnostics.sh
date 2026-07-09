#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"
: "${HH4B_SOFTWARE:?Need HH4B_SOFTWARE}"

MG5_BIN="${MG5_BIN:-$(command -v mg5_aMC || find "$HH4B_SOFTWARE" -path "*/bin/mg5_aMC" | head -1)}"
NEVENTS="${NEVENTS:-2000}"

if [[ -z "$MG5_BIN" ]]; then
  echo "ERROR: Could not find mg5_aMC"
  exit 1
fi

CARD_DIR="$HH4B_REPO/scripts/delphes/xsec_checks"
OUT_BASE="$HH4B_STORE/mg5_vbf_decay_diagnostics"
SUMMARY="$HH4B_STORE/metadata/mg5_vbf_decay_diagnostics.csv"

mkdir -p "$CARD_DIR" "$OUT_BASE" "$HH4B_STORE/metadata" "$HH4B_STORE/logs"

write_card () {
  local card="$1"
  local process="$2"
  local outdir="$3"

  cat > "$card" <<EOF
import model sm
define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~
$process
output $outdir -f
EOF
}

patch_run_card () {
  local run_card="$1"

  python3 - "$run_card" "$NEVENTS" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
nevents = sys.argv[2]

settings = {
    "nevents": nevents,
    "ebeam1": "6500.0",
    "ebeam2": "6500.0",
    "pdlabel": "nn23lo1",
    "lhaid": "230000",
    "cut_decays": "False",
}

lines = path.read_text().splitlines()
out = []
for line in lines:
    new = line
    for key, value in settings.items():
        pat = r"^(\s*)(.*?)(\s*=\s*" + re.escape(key) + r"\b.*)$"
        m = re.match(pat, line)
        if m:
            new = f"{m.group(1)}{value}{m.group(3)}"
            break
    out.append(new)

path.write_text("\n".join(out) + "\n")
PY
}

run_check () {
  local name="$1"
  local process="$2"

  local outdir="$OUT_BASE/$name"
  local card="$CARD_DIR/${name}.mg5"

  echo
  echo "============================================================"
  echo "Running $name"
  echo "$process"
  echo "============================================================"

  rm -rf "$outdir"
  write_card "$card" "$process" "$outdir"

  "$MG5_BIN" "$card" 2>&1 | tee "$HH4B_STORE/logs/${name}_output.log"

  patch_run_card "$outdir/Cards/run_card.dat"

  (
    cd "$outdir"
    ./bin/generate_events -f
  ) 2>&1 | tee "$HH4B_STORE/logs/${name}_generate_events.log"

  banner=$(find "$outdir/Events" -name "*banner.txt" | sort | tail -1)
  xsec_pb=$(grep -h "Integrated weight (pb)" "$banner" | tail -1 | awk -F: '{print $2}' | xargs)
  xsec_fb=$(python3 - <<PY
print(float("$xsec_pb") * 1000.0)
PY
)

  echo "$name,$xsec_pb,$xsec_fb,$process,$banner" >> "$SUMMARY"
}

rm -rf "$OUT_BASE"
mkdir -p "$OUT_BASE"

echo "sample,xsec_pb,xsec_fb,process,banner" > "$SUMMARY"

# Single-Higgs VBF sanity check
run_check "check_vbf_hjj_stable_sm" \
  "generate p p > h j j QCD=0"

run_check "check_vbf_hjj_bb_sm" \
  "generate p p > h j j QCD=0, h > b b~"

# HHjj syntax variants
run_check "check_vbf_hhjj_stable_sm" \
  "generate p p > h h j j QCD=0"

run_check "check_vbf_hhjj_decay_single_syntax_sm" \
  "generate p p > h h j j QCD=0, h > b b~"

run_check "check_vbf_hhjj_decay_double_syntax_sm" \
  "generate p p > h h j j QCD=0, (h > b b~), (h > b b~)"

echo
echo "Wrote: $SUMMARY"
cat "$SUMMARY"

python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

p = Path(os.environ["HH4B_STORE"]) / "metadata/mg5_vbf_decay_diagnostics.csv"
df = pd.read_csv(p)

print()
print(df[["sample", "xsec_pb", "xsec_fb"]].to_string(index=False))

def get(name):
    return float(df.loc[df["sample"] == name, "xsec_fb"].iloc[0])

print()
print("Ratios:")
print("VBF Hjj bb / stable:", get("check_vbf_hjj_bb_sm") / get("check_vbf_hjj_stable_sm"))
print("VBF HHjj single-syntax / stable:", get("check_vbf_hhjj_decay_single_syntax_sm") / get("check_vbf_hhjj_stable_sm"))
print("VBF HHjj double-syntax / stable:", get("check_vbf_hhjj_decay_double_syntax_sm") / get("check_vbf_hhjj_stable_sm"))
PY
