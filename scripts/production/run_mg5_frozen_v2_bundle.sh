#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$#" -ne 14 ]]; then
  echo "Usage:" >&2
  echo "  $0 CAMPAIGN FAMILY TARGET_TAG SHARD_ID N_EVENTS SEED DATASET_SPLIT SPLIT_SALT RUN_CARD_PROFILE DATASET_ROLE INPUT_TARBALL EOS_DIR EXPECTED_PAYLOAD_SHA256 CLUSTER_ID" >&2
  exit 2
fi

CAMPAIGN="$1"
FAMILY="$2"
TARGET_TAG="$3"
SHARD_ID="$4"
N_EVENTS="$5"
SEED="$6"
DATASET_SPLIT="$7"
SPLIT_SALT="$8"
RUN_CARD_PROFILE="$9"
DATASET_ROLE="${10}"
INPUT_TARBALL="${11}"
EOS_DIR="${12}"
EXPECTED_PAYLOAD_SHA256="${13}"
CLUSTER_ID="${14}"

[[ "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
[[ "$FAMILY" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
[[ "$TARGET_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
[[ "$SHARD_ID" =~ ^[0-9]+$ ]]
[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]]
[[ "$SEED" =~ ^[1-9][0-9]*$ ]]
[[ "$DATASET_SPLIT" =~ ^(train|validation)$ ]]
[[ "$SPLIT_SALT" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$RUN_CARD_PROFILE" =~ ^(bbbb_general|bbbb_iht400to600|zbbbb_general|preserve_template)$ ]]
[[ "$DATASET_ROLE" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$INPUT_TARBALL" != */* ]]
[[ "$EOS_DIR" == /store/user/* ]]
[[ "$EXPECTED_PAYLOAD_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$CLUSTER_ID" =~ ^[0-9]+$ ]]

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

RECEIPT="$SCRATCH/job_receipt.json"

STAGE="initializing"
PAYLOAD_SHA256=""
PAYLOAD_GIT_HEAD=""
CARD_SHA256=""
SOURCE_LHE_SHA256=""
BANNER_SHA256=""
ROOT_SHA256=""
EVENT_SUMMARY_SHA256=""
CANDIDATE_SHA256=""
BUNDLE_SHA256=""
BUNDLE_ADLER32=""
BUNDLE_SIZE_BYTES=0
REMOTE_BUNDLE=""
GENERATOR_XSEC_PB=0
N_LHE=0
N_HEPMC=0
N_ROOT=0
N_CANDIDATES=0
PYTHIA_SEED=0

finalize() {
  STATUS=$?

  cat > "$RECEIPT" <<EOF
{
  "schema_version": 1,
  "campaign": "$CAMPAIGN",
  "family": "$FAMILY",
  "target_tag": "$TARGET_TAG",
  "shard_id": $SHARD_ID,
  "n_events": $N_EVENTS,
  "seed": $SEED,
  "pythia_seed": $PYTHIA_SEED,
  "dataset_split": "$DATASET_SPLIT",
  "split_assignment_unit": "whole_shard",
  "split_salt": "$SPLIT_SALT",
  "run_card_profile": "$RUN_CARD_PROFILE",
  "dataset_role": "$DATASET_ROLE",
  "cluster_id": "$CLUSTER_ID",
  "stage": "$STAGE",
  "payload_sha256": "$PAYLOAD_SHA256",
  "expected_payload_sha256": "$EXPECTED_PAYLOAD_SHA256",
  "payload_git_head": "$PAYLOAD_GIT_HEAD",
  "delphes_card_sha256": "$CARD_SHA256",
  "source_lhe_sha256": "$SOURCE_LHE_SHA256",
  "banner_sha256": "$BANNER_SHA256",
  "root_sha256": "$ROOT_SHA256",
  "event_summary_sha256": "$EVENT_SUMMARY_SHA256",
  "candidate_sha256": "$CANDIDATE_SHA256",
  "bundle_sha256": "$BUNDLE_SHA256",
  "bundle_adler32": "$BUNDLE_ADLER32",
  "bundle_size_bytes": $BUNDLE_SIZE_BYTES,
  "remote_bundle": "$REMOTE_BUNDLE",
  "generator_xsec_pb": $GENERATOR_XSEC_PB,
  "lhe_events": $N_LHE,
  "hepmc_events": $N_HEPMC,
  "root_events": $N_ROOT,
  "candidate_rows": $N_CANDIDATES,
  "exit_status": $STATUS,
  "finished_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

  trap - EXIT
  exit "$STATUS"
}

trap finalize EXIT

test -s "$INPUT_TARBALL"

PAYLOAD_SHA256="$(
  sha256sum "$INPUT_TARBALL" |
  awk '{print $1}'
)"

if [[ "$PAYLOAD_SHA256" != "$EXPECTED_PAYLOAD_SHA256" ]]; then
  echo "ERROR: payload SHA-256 mismatch" >&2
  exit 3
fi

tar -xzf "$INPUT_TARBALL"

PAYLOAD="$SCRATCH/payload"
REPO="$PAYLOAD/repo"
DELPHES="$PAYLOAD/Delphes"
MG5_TEMPLATE="$PAYLOAD/mg5_template"
MANIFEST="$PAYLOAD/manifest.txt"

test -s "$MANIFEST"

PAYLOAD_FAMILY="$(
  awk -F= '$1 == "family" {print $2}' "$MANIFEST"
)"

PAYLOAD_GIT_HEAD="$(
  awk -F= '$1 == "git_head" {print $2}' "$MANIFEST"
)"

EXPECTED_WORKER_SHA256="$(
  awk -F= '$1 == "worker_wrapper_sha256" {print $2}' "$MANIFEST"
)"

EXPECTED_EVENT_SHA256="$(
  awk -F= '$1 == "event_summary_script_sha256" {print $2}' "$MANIFEST"
)"

EXPECTED_CANDIDATE_SHA256="$(
  awk -F= '$1 == "candidate_script_sha256" {print $2}' "$MANIFEST"
)"

EXPECTED_PARQUET_SHA256="$(
  awk -F= '$1 == "parquet_writer_sha256" {print $2}' "$MANIFEST"
)"

EXPECTED_CONVERTER_SHA256="$(
  awk -F= '$1 == "seeded_converter_sha256" {print $2}' "$MANIFEST"
)"

test "$PAYLOAD_FAMILY" = "$FAMILY"

test "$(
  sha256sum "$0" |
  awk '{print $1}'
)" = "$EXPECTED_WORKER_SHA256"

EVENT_SCRIPT="$REPO/scripts/delphes/make_delphes_event_summary.py"
CANDIDATE_SCRIPT="$REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
PARQUET_WRITER="$REPO/scripts/delphes/write_parquet_from_pickle.py"
CONVERTER_SOURCE="$REPO/scripts/production/lhe_to_hepmc3_seeded.cc"

test "$(
  sha256sum "$EVENT_SCRIPT" |
  awk '{print $1}'
)" = "$EXPECTED_EVENT_SHA256"

test "$(
  sha256sum "$CANDIDATE_SCRIPT" |
  awk '{print $1}'
)" = "$EXPECTED_CANDIDATE_SHA256"

test "$(
  sha256sum "$PARQUET_WRITER" |
  awk '{print $1}'
)" = "$EXPECTED_PARQUET_SHA256"

test "$(
  sha256sum "$CONVERTER_SOURCE" |
  awk '{print $1}'
)" = "$EXPECTED_CONVERTER_SHA256"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

test -r "$LCG_SETUP"

set +u
source "$LCG_SETUP"
set -u

unset SOURCE || true
hash -r

export LD_LIBRARY_PATH="$DELPHES:${LD_LIBRARY_PATH:-}"

for COMMAND in \
  python3 \
  g++ \
  pythia8-config \
  HepMC3-config \
  gzip \
  tar \
  sha256sum \
  xrdcp \
  xrdfs
do
  command -v "$COMMAND" >/dev/null
done

test -r "${X509_USER_PROXY:-}"

openssl x509 \
  -in "$X509_USER_PROXY" \
  -noout \
  -checkend 3600 >/dev/null

CARD="$REPO/cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"

EXPECTED_CARD_SHA256="1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"

CARD_SHA256="$(
  sha256sum "$CARD" |
  awk '{print $1}'
)"

test "$CARD_SHA256" = "$EXPECTED_CARD_SHA256"
test -x "$DELPHES/DelphesHepMC3"
test -x "$MG5_TEMPLATE/bin/generate_events"

OUT="$SCRATCH/output"
WORK="$SCRATCH/work_${TARGET_TAG}"

mkdir -p \
  "$OUT/root" \
  "$OUT/parquet" \
  "$OUT/metadata" \
  "$OUT/logs" \
  "$OUT/cards" \
  "$OUT/source"

cp -a "$MG5_TEMPLATE" "$WORK"

RUN_NAME="run_${TARGET_TAG}"

python3 - \
  "$WORK/Cards/run_card.dat" \
  "$N_EVENTS" \
  "$SEED" \
  "$RUN_CARD_PROFILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
n_events = int(sys.argv[2])
seed = int(sys.argv[3])
profile = sys.argv[4]

text = path.read_text()


def set_parameter(
    source: str,
    key: str,
    value: str,
    comment: str,
) -> str:
    pattern = rf"(?m)^.*=\s*{re.escape(key)}\b.*$"
    replacement = f"  {value} = {key} ! {comment}"

    if re.search(pattern, source):
        return re.sub(
            pattern,
            replacement,
            source,
            count=1,
        )

    return source.rstrip() + "\n" + replacement + "\n"


settings = {
    "nevents": (
        str(n_events),
        "Number of unweighted events",
    ),
    "iseed": (
        str(seed),
        "MG5 random seed",
    ),
    "ebeam1": (
        "6500.0",
        "beam 1 energy in GeV",
    ),
    "ebeam2": (
        "6500.0",
        "beam 2 energy in GeV",
    ),
}

heavy_flavor_settings = {
    "ptb": ("25.0", "minimum pT for b quarks"),
    "etab": ("2.7", "maximum eta for b quarks"),
    "drbb": ("0.4", "minimum deltaR between b quarks"),
    "drbj": (
        "0.4",
        "minimum deltaR between b quarks and light jets",
    ),
    "ptj": ("20.0", "minimum pT for jets"),
    "etaj": ("5.0", "maximum eta for jets"),
}

if profile == "bbbb_general":
    settings.update(heavy_flavor_settings)
    settings.update({
        "ihtmin": ("0.0", "inclusive partonic HT minimum"),
        "ihtmax": ("-1.0", "inclusive partonic HT maximum"),
    })
elif profile == "bbbb_iht400to600":
    settings.update(heavy_flavor_settings)
    settings.update({
        "ihtmin": ("400.0", "inclusive partonic HT minimum"),
        "ihtmax": ("600.0", "inclusive partonic HT maximum"),
    })
elif profile == "zbbbb_general":
    settings.update(heavy_flavor_settings)
    settings.update({
        "ihtmin": ("0.0", "inclusive partonic HT minimum"),
        "ihtmax": ("-1.0", "inclusive partonic HT maximum"),
        "cut_decays": (
            "False",
            "do not apply production cuts to decay products",
        ),
    })
elif profile == "preserve_template":
    pass
else:
    raise SystemExit(
        f"ERROR: unsupported run-card profile: {profile}"
    )

for key, (value, comment) in settings.items():
    text = set_parameter(
        text,
        key,
        value,
        comment,
    )

path.write_text(text.rstrip() + "\n")
PY

cp \
  "$WORK/Cards/proc_card_mg5.dat" \
  "$OUT/cards/proc_card_mg5.dat"

cp \
  "$WORK/Cards/run_card.dat" \
  "$OUT/cards/run_card.dat"

find "$WORK/Cards" \
  -maxdepth 1 \
  -type f \
  \( \
    -name 'pythia8_card*.dat' -o \
    -name 'pythia_card*.dat' \
  \) \
  -exec cp {} "$OUT/cards/" \;

STAGE="mg5_generation"

cd "$WORK"

set +e

./bin/generate_events "$RUN_NAME" -f \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_mg5.log"

MG5_STATUS=${PIPESTATUS[0]}

set -e

if [[ "$MG5_STATUS" -ne 0 ]]; then
  echo "ERROR: MG5 failed with status $MG5_STATUS" >&2
  exit 4
fi

LHE_GZ="$WORK/Events/$RUN_NAME/unweighted_events.lhe.gz"

test -s "$LHE_GZ"

N_LHE="$(
  gzip -cd "$LHE_GZ" |
  grep -c '<event>' \
  || true
)"

if [[ "$N_LHE" -ne "$N_EVENTS" ]]; then
  echo "ERROR: LHE event-count mismatch: $N_LHE != $N_EVENTS" >&2
  exit 5
fi

BANNER="$(
  find "$WORK/Events/$RUN_NAME" \
    -maxdepth 1 \
    -type f \
    -name '*_banner.txt' \
    -print \
    -quit
)"

test -s "$BANNER"

cp "$LHE_GZ" "$OUT/source/unweighted_events.lhe.gz"
cp "$BANNER" "$OUT/metadata/generation_banner.txt"

SOURCE_LHE_SHA256="$(
  sha256sum "$OUT/source/unweighted_events.lhe.gz" |
  awk '{print $1}'
)"

BANNER_SHA256="$(
  sha256sum "$OUT/metadata/generation_banner.txt" |
  awk '{print $1}'
)"

cd "$SCRATCH"

LHE="$SCRATCH/${TARGET_TAG}.lhe"
HEPMC="$SCRATCH/${TARGET_TAG}.hepmc"
CONVERTER="$SCRATCH/lhe_to_hepmc3_seeded"

STAGE="compile_converter"

g++ \
  -O2 \
  -std=c++17 \
  "$CONVERTER_SOURCE" \
  -o "$CONVERTER" \
  $(pythia8-config --cxxflags) \
  $(HepMC3-config --cxxflags) \
  $(pythia8-config --libs) \
  $(HepMC3-config --libs) \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_compile_converter.log"

PYTHIA_SEED=$(
  python3 - "$SEED" <<'PY'
import sys

seed = int(sys.argv[1])
pythia_seed = (seed * 37 + 17) % 900_000_000

if pythia_seed == 0:
    pythia_seed = 1

print(pythia_seed)
PY
)

gzip -cd "$LHE_GZ" > "$LHE"

STAGE="pythia_hepmc"

"$CONVERTER" \
  "$LHE" \
  "$HEPMC" \
  "$N_EVENTS" \
  "$PYTHIA_SEED" \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_pythia_hepmc.log"

N_HEPMC="$(
  grep -c '^E ' "$HEPMC" \
  || true
)"

if [[ "$N_HEPMC" -ne "$N_EVENTS" ]]; then
  echo "ERROR: HepMC event-count mismatch: $N_HEPMC != $N_EVENTS" >&2
  exit 6
fi

ROOT_FILE="$OUT/root/${TARGET_TAG}_delphes.root"
EVENT_SUMMARY="$OUT/parquet/${TARGET_TAG}_event_summary.parquet"
CANDIDATES="$OUT/parquet/${TARGET_TAG}_hh4b_candidates_v2.parquet"

STAGE="delphes"

"$DELPHES/DelphesHepMC3" \
  "$CARD" \
  "$ROOT_FILE" \
  "$HEPMC" \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_delphes.log"

N_ROOT="$(
  python3 -c \
    'import sys, uproot; print(uproot.open(sys.argv[1])["Delphes"].num_entries)' \
    "$ROOT_FILE"
)"

if [[ "$N_ROOT" -ne "$N_EVENTS" ]]; then
  echo "ERROR: ROOT event-count mismatch: $N_ROOT != $N_EVENTS" >&2
  exit 7
fi

STAGE="event_summary"

python3 \
  "$EVENT_SCRIPT" \
  --input "$ROOT_FILE" \
  --outdir "$OUT/parquet" \
  --sample "$TARGET_TAG" \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_event_summary.log"

test -s "$EVENT_SUMMARY"

EVENT_ROWS="$(
  python3 -c \
    'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' \
    "$EVENT_SUMMARY"
)"

if [[ "$EVENT_ROWS" -ne "$N_EVENTS" ]]; then
  echo "ERROR: event-summary row-count mismatch" >&2
  exit 8
fi

STAGE="candidate_reconstruction"

set +e

python3 \
  "$CANDIDATE_SCRIPT" \
  --input "$ROOT_FILE" \
  --out "$CANDIDATES" \
  --sample "$TARGET_TAG" \
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_candidates.log"

RECONSTRUCTION_STATUS=${PIPESTATUS[0]}

set -e

if [[ "$RECONSTRUCTION_STATUS" -ne 0 ]]; then
  if [[ "$RECONSTRUCTION_STATUS" -ne 134 ]]; then
    echo "ERROR: candidate reconstruction failed" >&2
    exit "$RECONSTRUCTION_STATUS"
  fi

  python3 - \
    "$EVENT_SUMMARY" \
    "$CANDIDATES" \
    "$N_EVENTS" <<'PY'
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

GENERATOR_XSEC_PB="$(
  python3 - "$EVENT_SUMMARY" <<'PY'
import pandas as pd
import sys

frame = pd.read_parquet(sys.argv[1])

if "event_cross_section_pb" not in frame.columns:
    raise SystemExit(
        "ERROR: event summary lacks event_cross_section_pb"
    )

value = float(frame["event_cross_section_pb"].median())

if not value > 0:
    raise SystemExit(
        "ERROR: generator cross section is not positive"
    )

print(repr(value))
PY
)"

ROOT_SHA256="$(
  sha256sum "$ROOT_FILE" |
  awk '{print $1}'
)"

EVENT_SUMMARY_SHA256="$(
  sha256sum "$EVENT_SUMMARY" |
  awk '{print $1}'
)"

CANDIDATE_SHA256="$(
  sha256sum "$CANDIDATES" |
  awk '{print $1}'
)"

PROVENANCE="$OUT/metadata/${TARGET_TAG}_provenance.json"

python3 - \
  "$PROVENANCE" \
  "$OUT/cards/proc_card_mg5.dat" \
  "$CAMPAIGN" \
  "$FAMILY" \
  "$TARGET_TAG" \
  "$SHARD_ID" \
  "$N_EVENTS" \
  "$SEED" \
  "$PYTHIA_SEED" \
  "$DATASET_SPLIT" \
  "$SPLIT_SALT" \
  "$RUN_CARD_PROFILE" \
  "$DATASET_ROLE" \
  "$GENERATOR_XSEC_PB" \
  "$PAYLOAD_SHA256" \
  "$PAYLOAD_GIT_HEAD" \
  "$CARD_SHA256" \
  "$SOURCE_LHE_SHA256" \
  "$BANNER_SHA256" \
  "$ROOT_SHA256" \
  "$EVENT_SUMMARY_SHA256" \
  "$CANDIDATE_SHA256" \
  "$N_LHE" \
  "$N_HEPMC" \
  "$N_ROOT" \
  "$N_CANDIDATES" <<'PY'
from pathlib import Path
import json
import re
import sys

(
    output_path,
    process_card_path,
    campaign,
    family,
    target_tag,
    shard_id,
    n_events,
    seed,
    pythia_seed,
    dataset_split,
    split_salt,
    run_card_profile,
    dataset_role,
    generator_xsec_pb,
    payload_sha256,
    payload_git_head,
    card_sha256,
    source_lhe_sha256,
    banner_sha256,
    root_sha256,
    event_summary_sha256,
    candidate_sha256,
    lhe_events,
    hepmc_events,
    root_events,
    candidate_rows,
) = sys.argv[1:]

process_lines = []

for line in Path(process_card_path).read_text().splitlines():
    stripped = line.strip()

    if re.match(r"^(generate|add process)\s+", stripped):
        process_lines.append(stripped)

record = {
    "schema_version": 1,
    "campaign": campaign,
    "family": family,
    "target_tag": target_tag,
    "process_definition": " | ".join(process_lines),
    "shard_id": int(shard_id),
    "n_events": int(n_events),
    "seed": int(seed),
    "pythia_seed": int(pythia_seed),
    "dataset_split": dataset_split,
    "split_assignment_unit": "whole_shard",
    "split_salt": split_salt,
    "run_card_profile": run_card_profile,
    "dataset_role": dataset_role,
    "generator_xsec_pb": float(generator_xsec_pb),
    "normalization_status": (
        "generator_level_provisional_not_final_higher_order"
    ),
    "payload_sha256": payload_sha256,
    "payload_git_head": payload_git_head,
    "delphes_card_sha256": card_sha256,
    "source_lhe_sha256": source_lhe_sha256,
    "banner_sha256": banner_sha256,
    "root_sha256": root_sha256,
    "event_summary_sha256": event_summary_sha256,
    "candidate_sha256": candidate_sha256,
    "lhe_events": int(lhe_events),
    "hepmc_events": int(hepmc_events),
    "root_events": int(root_events),
    "candidate_rows": int(candidate_rows),
}

Path(output_path).write_text(
    json.dumps(
        record,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

rm -f "$LHE" "$HEPMC"
rm -rf "$WORK"

STAGE="bundling"

BUNDLE="$SCRATCH/${TARGET_TAG}_bundle.tar.gz"

tar \
  -C "$OUT" \
  -czf "$BUNDLE" \
  .

BUNDLE_SIZE_BYTES="$(
  stat -c '%s' "$BUNDLE"
)"

BUNDLE_SHA256="$(
  sha256sum "$BUNDLE" |
  awk '{print $1}'
)"

BUNDLE_ADLER32="$(
  python3 - "$BUNDLE" <<'PY'
import sys
import zlib

checksum = 1

with open(sys.argv[1], "rb") as handle:
    for block in iter(
        lambda: handle.read(8 * 1024 * 1024),
        b"",
    ):
        checksum = zlib.adler32(
            block,
            checksum,
        )

print(f"{checksum & 0xffffffff:08x}")
PY
)"

EOS_HOST="root://cmseos.fnal.gov"

REMOTE_BUNDLE="$EOS_DIR/${TARGET_TAG}_bundle.tar.gz"
REMOTE_TMP="${REMOTE_BUNDLE}.partial.${CLUSTER_ID}.$$"

if xrdfs "$EOS_HOST" stat "$REMOTE_BUNDLE" >/dev/null 2>&1; then
  echo "ERROR: refusing to overwrite existing EOS bundle" >&2
  exit 9
fi

STAGE="stageout"

xrdfs "$EOS_HOST" mkdir -p "$EOS_DIR"

xrdcp \
  -f \
  --nopbar \
  --cksum adler32:print \
  "$BUNDLE" \
  "${EOS_HOST}/${REMOTE_TMP}"

REMOTE_TMP_ADLER="$(
  xrdfs "$EOS_HOST" query checksum "$REMOTE_TMP" |
  awk '{print tolower($NF)}'
)"

if [[ "$REMOTE_TMP_ADLER" != "$BUNDLE_ADLER32" ]]; then
  echo "ERROR: temporary EOS checksum mismatch" >&2
  exit 10
fi

xrdfs "$EOS_HOST" mv \
  "$REMOTE_TMP" \
  "$REMOTE_BUNDLE"

REMOTE_ADLER="$(
  xrdfs "$EOS_HOST" query checksum "$REMOTE_BUNDLE" |
  awk '{print tolower($NF)}'
)"

if [[ "$REMOTE_ADLER" != "$BUNDLE_ADLER32" ]]; then
  echo "ERROR: final EOS checksum mismatch" >&2
  exit 11
fi

REMOTE_SIZE="$(
  xrdfs "$EOS_HOST" stat "$REMOTE_BUNDLE" |
  awk '
    $1 == "Size:" {
      print $2
      exit
    }
  '
)"

if [[ "$REMOTE_SIZE" -ne "$BUNDLE_SIZE_BYTES" ]]; then
  echo "ERROR: final EOS size mismatch" >&2
  exit 12
fi

STAGE="complete_copied_and_verified"

echo "SUCCESS: $REMOTE_BUNDLE"
echo "BUNDLE_SHA256: $BUNDLE_SHA256"
echo "ADLER32: $BUNDLE_ADLER32"
echo "BUNDLE_SIZE_BYTES: $BUNDLE_SIZE_BYTES"
echo "GENERATOR_XSEC_PB: $GENERATOR_XSEC_PB"
echo "ROOT_EVENTS: $N_ROOT"
echo "CANDIDATE_ROWS: $N_CANDIDATES"
