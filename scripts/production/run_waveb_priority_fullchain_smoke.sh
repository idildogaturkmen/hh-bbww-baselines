#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 FAMILY" >&2
  echo "Families: tttt tth_hbb ttz_zbb vbf_hbb" >&2
  exit 2
fi

FAMILY="$1"

case "$FAMILY" in
  tttt|tth_hbb|ttz_zbb|vbf_hbb)
    ;;
  *)
    echo "ERROR: unsupported family: $FAMILY" >&2
    exit 2
    ;;
esac

REPO="/uscms_data/d3/$USER/repos/hh-bbww-baselines"
STORE="/uscms_data/d3/$USER/hh4b_delphes"
DELPHES="/uscms_data/d3/$USER/software/Delphes"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

CARD="$REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
EXPECTED_CARD_SHA256="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

PLAN="$REPO/metadata/production_plans/waveb_priority_smoke_templates_20260721.tsv"
LHE_SUMMARY="$REPO/outputs/agent_runs/waveb_priority_lhe_smokes_20260721/$FAMILY/lhe_validation.json"

CONVERTER_SOURCE="$REPO/scripts/production/lhe_to_hepmc3_seeded.cc"
COMPILE_SCRIPT="$REPO/scripts/production/compile_pythia8_hepmc3.sh"
EVENT_SCRIPT="$REPO/scripts/delphes/make_delphes_event_summary.py"
CANDIDATE_SCRIPT="$REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py"

FULLCHAIN_ROOT="$STORE/waveb_fullchain_smokes_20260721"
RUNTIME_DIR="$FULLCHAIN_ROOT/runtime"
OUT="$FULLCHAIN_ROOT/$FAMILY"

TAG="waveb_${FAMILY}_fullchain_smoke100_20260721"

LOCAL_HEAD="$(
  git -C "$REPO" rev-parse HEAD
)"

REMOTE_HEAD="$(
  git -C "$REPO" rev-parse origin/delphes-hh4b-production
)"

DIRTY_COUNT="$(
  git -C "$REPO" status --porcelain |
  wc -l
)"

echo "family=$FAMILY"
echo "local_head=$LOCAL_HEAD"
echo "remote_head=$REMOTE_HEAD"
echo "dirty_entries=$DIRTY_COUNT"

test "$LOCAL_HEAD" = "$REMOTE_HEAD"
test "$DIRTY_COUNT" -eq 0

test -s "$PLAN"
test -s "$LHE_SUMMARY"
test -s "$CONVERTER_SOURCE"
test -x "$COMPILE_SCRIPT"
test -s "$EVENT_SCRIPT"
test -s "$CANDIDATE_SCRIPT"
test -x "$DELPHES/DelphesHepMC3"
test -s "$CARD"
test -s "$LCG_SETUP"

CARD_SHA256="$(
  sha256sum "$CARD" |
  awk '{print $1}'
)"

echo "delphes_card_sha256=$CARD_SHA256"

test "$CARD_SHA256" = "$EXPECTED_CARD_SHA256"

readarray -t INPUT_RECORD < <(
  python3 - "$LHE_SUMMARY" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
record = json.loads(path.read_text())

lhe_file = Path(record["lhe_file"])
lhe_events = int(record["lhe_events"])
failures = int(record["topology"]["failing_event_count"])

if not lhe_file.is_file() or lhe_file.stat().st_size == 0:
    raise SystemExit("ERROR: missing LHE file")

if lhe_events != 100:
    raise SystemExit("ERROR: LHE validation did not record 100 events")

if failures != 0:
    raise SystemExit("ERROR: LHE topology validation did not pass")

print(lhe_file)
print(lhe_events)
PY
)

LHE_GZ="${INPUT_RECORD[0]}"
EXPECTED_EVENTS="${INPUT_RECORD[1]}"

MG5_SEED="$(
  python3 - "$PLAN" "$FAMILY" <<'PY'
import csv
from pathlib import Path
import sys

plan = Path(sys.argv[1])
family = sys.argv[2]

with plan.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

matches = [
    row
    for row in rows
    if row["family"] == family
]

if len(matches) != 1:
    raise SystemExit("ERROR: family not found exactly once in plan")

row = matches[0]

if row["dataset_split"] == "test":
    raise SystemExit("ERROR: final test data are not authorized")

print(int(row["seed"]))
PY
)"

PYTHIA_SEED="$(
  python3 - "$MG5_SEED" <<'PY'
import sys

seed = int(sys.argv[1])
value = (seed * 37 + 17) % 900_000_000

if value == 0:
    value = 1

print(value)
PY
)"

echo "lhe_file=$LHE_GZ"
echo "expected_events=$EXPECTED_EVENTS"
echo "mg5_seed=$MG5_SEED"
echo "pythia_seed=$PYTHIA_SEED"

if [[ -e "$OUT" ]]; then
  echo "ERROR: refusing to overwrite existing output: $OUT" >&2
  exit 3
fi

mkdir -p \
  "$RUNTIME_DIR" \
  "$OUT/source" \
  "$OUT/hepmc" \
  "$OUT/root" \
  "$OUT/parquet" \
  "$OUT/logs" \
  "$OUT/metadata"

set +u
source "$LCG_SETUP"
set -u

unset SOURCE 2>/dev/null || true
hash -r

for COMMAND in \
  python3 \
  g++ \
  pythia8-config \
  HepMC3-config \
  sha256sum \
  gzip
do
  command -v "$COMMAND" >/dev/null
done

CONVERTER="$RUNTIME_DIR/lhe_to_hepmc3_seeded"
SOURCE_SHA_FILE="$RUNTIME_DIR/lhe_to_hepmc3_seeded.source.sha256"
COMPILE_ARGUMENTS="$RUNTIME_DIR/lhe_to_hepmc3_seeded.compile_args.txt"

CONVERTER_SOURCE_SHA256="$(
  sha256sum "$CONVERTER_SOURCE" |
  awk '{print $1}'
)"

RECOMPILE=1

if [[ -x "$CONVERTER" && -s "$SOURCE_SHA_FILE" ]]; then
  RECORDED_SOURCE_SHA256="$(
    cat "$SOURCE_SHA_FILE"
  )"

  if [[ "$RECORDED_SOURCE_SHA256" = "$CONVERTER_SOURCE_SHA256" ]]; then
    RECOMPILE=0
  fi
fi

if [[ "$RECOMPILE" -eq 1 ]]; then
  rm -f \
    "$CONVERTER" \
    "$SOURCE_SHA_FILE" \
    "$COMPILE_ARGUMENTS"

  "$COMPILE_SCRIPT" \
    "$CONVERTER_SOURCE" \
    "$CONVERTER" \
    "$COMPILE_ARGUMENTS" \
    2>&1 |
  tee "$RUNTIME_DIR/compile_converter.log"

  printf '%s\n' "$CONVERTER_SOURCE_SHA256" \
    > "$SOURCE_SHA_FILE"
fi

test -x "$CONVERTER"

CONVERTER_SHA256="$(
  sha256sum "$CONVERTER" |
  awk '{print $1}'
)"

LHE="$OUT/source/${TAG}.lhe"
HEPMC="$OUT/hepmc/${TAG}.hepmc"
ROOT_FILE="$OUT/root/${TAG}_delphes.root"
EVENT_SUMMARY="$OUT/parquet/${TAG}_event_summary.parquet"
CANDIDATES="$OUT/parquet/${TAG}_hh4b_candidates_v2.parquet"

gzip -cd "$LHE_GZ" > "$LHE"

N_LHE="$(
  grep -c '<event>' "$LHE" ||
  true
)"

echo "lhe_events=$N_LHE"

test "$N_LHE" -eq "$EXPECTED_EVENTS"

"$CONVERTER" \
  "$LHE" \
  "$HEPMC" \
  "$EXPECTED_EVENTS" \
  "$PYTHIA_SEED" \
  2>&1 |
tee "$OUT/logs/${TAG}_pythia_hepmc.log"

N_HEPMC="$(
  grep -c '^E ' "$HEPMC" ||
  true
)"

echo "hepmc_events=$N_HEPMC"

test "$N_HEPMC" -eq "$EXPECTED_EVENTS"

export LD_LIBRARY_PATH="$DELPHES:${LD_LIBRARY_PATH:-}"

"$DELPHES/DelphesHepMC3" \
  "$CARD" \
  "$ROOT_FILE" \
  "$HEPMC" \
  2>&1 |
tee "$OUT/logs/${TAG}_delphes.log"

N_ROOT="$(
  python3 -c \
    'import sys, uproot; print(uproot.open(sys.argv[1])["Delphes"].num_entries)' \
    "$ROOT_FILE"
)"

echo "root_events=$N_ROOT"

test "$N_ROOT" -eq "$EXPECTED_EVENTS"

python3 \
  "$EVENT_SCRIPT" \
  --input "$ROOT_FILE" \
  --outdir "$OUT/parquet" \
  --sample "$TAG" \
  2>&1 |
tee "$OUT/logs/${TAG}_event_summary.log"

test -s "$EVENT_SUMMARY"

EVENT_ROWS="$(
  python3 -c \
    'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$EVENT_SUMMARY"
)"

echo "event_summary_rows=$EVENT_ROWS"

test "$EVENT_ROWS" -eq "$EXPECTED_EVENTS"

set +e

python3 \
  "$CANDIDATE_SCRIPT" \
  --input "$ROOT_FILE" \
  --out "$CANDIDATES" \
  --sample "$TAG" \
  2>&1 |
tee "$OUT/logs/${TAG}_candidates.log"

CANDIDATE_STATUS=${PIPESTATUS[0]}

set -e

if [[ "$CANDIDATE_STATUS" -ne 0 ]]; then
  if [[ "$CANDIDATE_STATUS" -ne 134 ]]; then
    echo "ERROR: candidate reconstruction failed" >&2
    exit "$CANDIDATE_STATUS"
  fi

  python3 - \
    "$EVENT_SUMMARY" \
    "$CANDIDATES" \
    "$EXPECTED_EVENTS" <<'PY'
from pathlib import Path
import os
import sys

import pandas as pd

event_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
expected_events = int(sys.argv[3])

if not event_path.is_file() or event_path.stat().st_size == 0:
    raise SystemExit("ERROR: missing event summary")

if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
    raise SystemExit("ERROR: missing candidate Parquet")

events = pd.read_parquet(event_path)
candidates = pd.read_parquet(candidate_path)

if len(events) != expected_events:
    raise SystemExit("ERROR: wrong event-summary row count")

if "n_bjet_pt30_eta25" not in events.columns:
    raise SystemExit("ERROR: missing b-jet multiplicity column")

if int((events["n_bjet_pt30_eta25"] >= 4).sum()) != 0:
    raise SystemExit(
        "ERROR: exit 134 occurred despite >=4b events"
    )

if len(candidates) != 0:
    raise SystemExit(
        "ERROR: exit 134 occurred with nonempty candidates"
    )

os.write(
    1,
    b"RECOVERED_VALID_EMPTY_CANDIDATE_TEARDOWN_EXIT_134\n",
)

os._exit(0)
PY
fi

test -s "$CANDIDATES"

N_CANDIDATES="$(
  python3 -c \
    'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$CANDIDATES"
)"

echo "candidate_rows=$N_CANDIDATES"

python3 - \
  "$OUT/metadata/fullchain_validation.json" \
  "$FAMILY" \
  "$TAG" \
  "$LOCAL_HEAD" \
  "$MG5_SEED" \
  "$PYTHIA_SEED" \
  "$CARD_SHA256" \
  "$CONVERTER_SOURCE_SHA256" \
  "$CONVERTER_SHA256" \
  "$LHE_GZ" \
  "$LHE" \
  "$HEPMC" \
  "$ROOT_FILE" \
  "$EVENT_SUMMARY" \
  "$CANDIDATES" \
  "$N_LHE" \
  "$N_HEPMC" \
  "$N_ROOT" \
  "$EVENT_ROWS" \
  "$N_CANDIDATES" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

(
    output,
    family,
    tag,
    git_head,
    mg5_seed,
    pythia_seed,
    card_sha,
    converter_source_sha,
    converter_sha,
    source_lhe_gz,
    lhe,
    hepmc,
    root_file,
    event_summary,
    candidates,
    n_lhe,
    n_hepmc,
    n_root,
    event_rows,
    candidate_rows,
) = sys.argv[1:]


def checksum(path_string):
    path = Path(path_string)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


record = {
    "schema_version": 1,
    "family": family,
    "tag": tag,
    "git_head": git_head,
    "mg5_seed": int(mg5_seed),
    "pythia_seed": int(pythia_seed),
    "delphes_card_sha256": card_sha,
    "converter_source_sha256": converter_source_sha,
    "converter_binary_sha256": converter_sha,
    "source_lhe_gz": source_lhe_gz,
    "lhe_events": int(n_lhe),
    "hepmc_events": int(n_hepmc),
    "root_events": int(n_root),
    "event_summary_rows": int(event_rows),
    "candidate_rows": int(candidate_rows),
    "checksums": {
        "lhe": checksum(lhe),
        "hepmc": checksum(hepmc),
        "root": checksum(root_file),
        "event_summary": checksum(event_summary),
        "candidates": checksum(candidates),
    },
    "local_fullchain_smoke_pass": True,
    "authorized_for_condor_scaleout": False,
    "authorized_for_final_test": False,
}

Path(output).write_text(
    json.dumps(
        record,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(f"summary={output}")
PY

echo "WAVEB_FULLCHAIN_SMOKE_COMPLETE_AND_VALID"
