#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"
: "${HH4B_SOFTWARE:?Need HH4B_SOFTWARE}"

MG5_BIN="${MG5_BIN:-$(command -v mg5_aMC || find "$HH4B_SOFTWARE" -path "*/bin/mg5_aMC" | head -1)}"
NEVENTS="${NEVENTS:-2000}"

if [[ -z "$MG5_BIN" ]]; then
  echo "ERROR: Could not find mg5_aMC. Set MG5_BIN manually."
  exit 1
fi

echo "Using MG5: $MG5_BIN"
echo "NEVENTS=$NEVENTS"

CARD_DIR="$HH4B_REPO/scripts/delphes/xsec_checks"
OUT_BASE="$HH4B_STORE/mg5_xsec_checks_v2"
SUMMARY="$HH4B_STORE/metadata/mg5_xsec_sanity_checks_v2.csv"

mkdir -p "$CARD_DIR" "$OUT_BASE" "$HH4B_STORE/metadata" "$HH4B_STORE/logs"

write_output_card () {
  local card="$1"
  local model="$2"
  local process="$3"
  local outdir="$4"
  local pdef="$5"

  cat > "$card" <<EOF
import model $model
$pdef
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

text = path.read_text().splitlines()
out = []

for line in text:
    new_line = line
    for key, value in settings.items():
        # Match lines like: "  10000 = nevents ! comment"
        pattern = r"^(\s*)(.*?)(\s*=\s*" + re.escape(key) + r"\b.*)$"
        m = re.match(pattern, line)
        if m:
            new_line = f"{m.group(1)}{value}{m.group(3)}"
            break
    out.append(new_line)

path.write_text("\n".join(out) + "\n")
print(f"Patched {path}")
PY
}

run_check () {
  local name="$1"
  local model="$2"
  local process="$3"
  local pdef="$4"

  local outdir="$OUT_BASE/$name"
  local card="$CARD_DIR/${name}.mg5"

  echo
  echo "============================================================"
  echo "Preparing $name"
  echo "Process: $process"
  echo "Output: $outdir"
  echo "============================================================"

  rm -rf "$outdir"
  write_output_card "$card" "$model" "$process" "$outdir" "$pdef"

  echo
  echo "--- MG5 output step ---"
  "$MG5_BIN" "$card" 2>&1 | tee "$HH4B_STORE/logs/${name}_output.log"

  echo
  echo "--- Patch run card ---"
  patch_run_card "$outdir/Cards/run_card.dat"

  echo
  echo "--- Selected run card lines ---"
  grep -n "nevents\|ebeam\|pdlabel\|lhaid\|cut_decays" "$outdir/Cards/run_card.dat" || true

  echo
  echo "--- Generate events ---"
  (
    cd "$outdir"
    ./bin/generate_events -f
  ) 2>&1 | tee "$HH4B_STORE/logs/${name}_generate_events.log"

  local banner
  banner=$(find "$outdir/Events" -name "*banner.txt" | sort | tail -1)

  if [[ -z "$banner" ]]; then
    echo "ERROR: no banner found for $name"
    exit 1
  fi

  local xsec_pb
  xsec_pb=$(grep -h "Integrated weight (pb)" "$banner" | tail -1 | awk -F: '{print $2}' | xargs)

  if [[ -z "$xsec_pb" ]]; then
    echo "ERROR: could not parse cross section for $name from $banner"
    exit 1
  fi

  local xsec_fb
  xsec_fb=$(python3 - <<PY
x = float("$xsec_pb")
print(x * 1000.0)
PY
)

  echo "$name,$xsec_pb,$xsec_fb,$banner" >> "$SUMMARY"
}

rm -rf "$OUT_BASE"
mkdir -p "$OUT_BASE"
echo "sample,xsec_pb,xsec_fb,banner" > "$SUMMARY"

# 1. A simple common SM sanity check: on-shell Z to ee.
run_check \
  "check_z_to_ee_sm" \
  "sm" \
  "generate p p > z, z > e+ e-" \
  "define p = g u c d s u~ c~ d~ s~"

# 2. Single-Higgs ggF in HEFT, stable H.
run_check \
  "check_ggf_h_stable_heft" \
  "$HH4B_STORE/mg5_models/heft" \
  "generate p p > h" \
  "define p = g"

# 3. Single-Higgs ggF in HEFT, H->bb.
run_check \
  "check_ggf_h_bb_heft" \
  "$HH4B_STORE/mg5_models/heft" \
  "generate p p > h, h > b b~" \
  "define p = g"

# 4. HH ggF in HEFT, stable HH.
run_check \
  "check_ggf_hh_stable_heft" \
  "$HH4B_STORE/mg5_models/heft" \
  "generate p p > h h" \
  "define p = g"

# 5. HH ggF in HEFT, HH->4b.
run_check \
  "check_ggf_hh_4b_heft" \
  "$HH4B_STORE/mg5_models/heft" \
  "generate p p > h h, (h > b b~), (h > b b~)" \
  "define p = g"

# 6. Electroweak/VBF-like HHjj in SM, stable HH.
run_check \
  "check_vbf_hhjj_stable_sm" \
  "sm" \
  "generate p p > h h j j QCD=0" \
  "define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~"

# 7. Electroweak/VBF-like HHjj in SM, HH->4b.
run_check \
  "check_vbf_hhjj_4b_sm" \
  "sm" \
  "generate p p > h h j j QCD=0, (h > b b~), (h > b b~)" \
  "define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~"

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
cat "$SUMMARY"

python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

summary = Path(os.environ["HH4B_STORE"]) / "metadata/mg5_xsec_sanity_checks_v2.csv"
df = pd.read_csv(summary)

print()
print("Cross-section summary:")
print(df[["sample", "xsec_pb", "xsec_fb"]].to_string(index=False))

def get(name):
    return float(df.loc[df["sample"] == name, "xsec_fb"].iloc[0])

print()
print("Derived ratios:")
checks = [
    ("check_ggf_h_bb_heft", "check_ggf_h_stable_heft", "ggF H->bb / stable H"),
    ("check_ggf_hh_4b_heft", "check_ggf_hh_stable_heft", "ggF HH->4b / stable HH"),
    ("check_vbf_hhjj_4b_sm", "check_vbf_hhjj_stable_sm", "VBF HHjj->4b / stable HHjj"),
]

for num, den, label in checks:
    if num in set(df["sample"]) and den in set(df["sample"]):
        r = get(num) / get(den)
        print(f"{label}: {r:.4f}")

print()
print("Reference BR(H->bb)^2 using 0.5824:")
print(f"{0.5824**2:.4f}")
PY

echo
echo "Wrote summary: $SUMMARY"
