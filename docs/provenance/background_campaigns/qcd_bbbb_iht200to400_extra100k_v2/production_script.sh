#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

cd "$HH4B_REPO"

PROCESS_DIR="$HH4B_STORE/mg5/QCD_bbbb_presel_smoke"
RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
BACKUP_RUN_CARD="${RUN_CARD}.before_iht200to400_extra100k"

CAMPAIGN="qcd_bbbb_iht200to400_extra100k_v2"
N_SHARDS=10
EVENTS_PER_SHARD=10000
SEED0=452000

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

    # Target slice
    "ihtmin": ("200", "inclusive HT minimum for all partons including b"),
    "ihtmax": ("400", "inclusive HT maximum for all partons including b"),

    # Disable other HT slicing requirements
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

echo "Using QCD bbbb iht200to400 run-card settings:"
grep -E "ptb|ptbmax|ihtmin|ihtmax|ht2min|ht3min|ht4min|ht2max|ht3max|ht4max" "$RUN_CARD" | head -80 || true

scripts/delphes/run_storage_safe_local_campaign.sh \
  "$CAMPAIGN" \
  "$PROCESS_DIR" \
  "$N_SHARDS" \
  "$EVENTS_PER_SHARD" \
  "$SEED0" \
  no \
  1 \
  yes

echo
echo "DONE $CAMPAIGN"
