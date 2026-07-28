#!/usr/bin/env bash
# Local exact-image reconstruction-only recovery for the seven retained
# HH4b ttbar Delphes ROOT files.
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 MEMBER_NAME SEED" >&2
  exit 64
fi

MEMBER="$1"
SEED="$2"

case "${MEMBER}:${SEED}" in
  ttbar_100k_shard001:105001)
    EXPECTED_ROOT_SHA="cd812441e240f0fd7b137129aadf4def052907f44830784879ca4e5a0d6f395c"
    EXPECTED_CANDIDATE_SHA="f42a8bb61007940b71c3e83870cb234d39806dc7f6ac361adbe20842e3118db3"
    ;;
  ttbar_100k_shard002:105002)
    EXPECTED_ROOT_SHA="3328999ac5bc867fc29b29e220d382d468076a4f85b8f6c8193679a702992d61"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  ttbar_100k_shard004:105004)
    EXPECTED_ROOT_SHA="fc7d89902695b64ba84ee39c183dd89b549840e32cf1de9f086f96df7b2dad4c"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  ttbar_100k_shard005:105005)
    EXPECTED_ROOT_SHA="5a405c983c1fea331dfd853caef09b05f1c092b543d0d80dedf7f303fac03111"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  ttbar_100k_shard007:105007)
    EXPECTED_ROOT_SHA="181529274199a5f38c795590dd471ed3d8e0dbe4676d5678c3b744ebf09643a2"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  ttbar_100k_shard008:105008)
    EXPECTED_ROOT_SHA="04f3142f14807b77601b735a4901bfe411102bcdc29f17c27189a75869800c50"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  ttbar_100k_shard009:105009)
    EXPECTED_ROOT_SHA="62fa589354ecfb8415331393e83c01475a2fe19f30d96bf46cad11bdfca2d50f"
    EXPECTED_CANDIDATE_SHA=""
    ;;
  *)
    echo "ERROR: member and seed do not match the frozen recovery contract" >&2
    exit 65
    ;;
esac

SOURCE_CAMPAIGN="hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1"
RECOVERY_CAMPAIGN="hh4b_ttbar7_reconstruction_only_recovery_20260728_v1"

SOURCE_RETURN_BASE="/uscms_data/d3/$USER/hh4b_delphes/condor_return/$SOURCE_CAMPAIGN"
INPUT_BASE="/uscms_data/d3/$USER/hh4b_delphes/condor_inputs/$SOURCE_CAMPAIGN"
OUTPUT_BASE="/uscms_data/d3/$USER/hh4b_delphes/reconstruction_return/$RECOVERY_CAMPAIGN"

PAYLOAD="$INPUT_BASE/hh4b_ttbar7_exact_regeneration_scaleout_inputs.tar.gz"
PYENV="$INPUT_BASE/python39_site_packages.tar.gz"

EXPECTED_PAYLOAD_SHA="76b1f09935baa378d9d410607aa81cc56e72e084390a069e3ce434e377a054df"
EXPECTED_PYENV_SHA="6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"

IMAGE="/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel9"

ROOT_PATH="$SOURCE_RETURN_BASE/members/$MEMBER/root/${MEMBER}_pythia8_delphes.root"

MEMBER_OUTPUT="$OUTPUT_BASE/members/$MEMBER"
PARQUET_DIR="$MEMBER_OUTPUT/parquet"
LOG_DIR="$MEMBER_OUTPUT/logs"
RECEIPT_DIR="$MEMBER_OUTPUT/receipts"
CHECKSUM_DIR="$MEMBER_OUTPUT/checksums"

mkdir -p \
  "$PARQUET_DIR" \
  "$LOG_DIR" \
  "$RECEIPT_DIR" \
  "$CHECKSUM_DIR"

if find "$MEMBER_OUTPUT" -type f -print -quit | grep -q .; then
  echo "ERROR: output area for $MEMBER already contains files." >&2
  exit 66
fi

TMP=""

finalize() {
  status=$?

  trap - EXIT
  set +e

  {
    echo "exit_status=$status"
    echo "campaign=$RECOVERY_CAMPAIGN"
    echo "member=$MEMBER"
    echo "seed=$SEED"
    echo "reconstruction_only=true"
    echo "madgraph_rerun=false"
    echo "pythia_rerun=false"
    echo "delphes_rerun=false"
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$RECEIPT_DIR/${MEMBER}_final_status.txt"

  if [[ -n "$TMP" ]]; then
    rm -rf "$TMP"
  fi

  exit "$status"
}

trap finalize EXIT

for REQUIRED in \
  "$PAYLOAD" \
  "$PYENV" \
  "$ROOT_PATH" \
  "$IMAGE"
do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "ERROR: required input is absent: $REQUIRED" >&2
    exit 67
  fi
done

if [[ "$(sha256sum "$PAYLOAD" | awk '{print $1}')" != "$EXPECTED_PAYLOAD_SHA" ]]; then
  echo "ERROR: payload SHA256 mismatch" >&2
  exit 68
fi

if [[ "$(sha256sum "$PYENV" | awk '{print $1}')" != "$EXPECTED_PYENV_SHA" ]]; then
  echo "ERROR: portable environment SHA256 mismatch" >&2
  exit 68
fi

if [[ "$(sha256sum "$ROOT_PATH" | awk '{print $1}')" != "$EXPECTED_ROOT_SHA" ]]; then
  echo "ERROR: retained ROOT SHA256 mismatch" >&2
  exit 68
fi

APPTAINER="$(
  command -v apptainer \
  || command -v singularity \
  || true
)"

if [[ -z "$APPTAINER" ]]; then
  echo "ERROR: no Apptainer/Singularity command found" >&2
  exit 69
fi

TMP="$(
  mktemp -d \
    "/uscms_data/d3/$USER/hh4b_delphes/${RECOVERY_CAMPAIGN}.${MEMBER}.XXXXXX"
)"

mkdir -p \
  "$TMP/environment"

tar -xzf "$PAYLOAD" \
  -C "$TMP"

tar -xzf "$PYENV" \
  -C "$TMP/environment"

test -d "$TMP/payload"
test -d "$TMP/environment/site-packages"

ROOT_PARENT="$(dirname "$ROOT_PATH")"
ROOT_NAME="$(basename "$ROOT_PATH")"
OUTPUT_ABS="$(readlink -f "$MEMBER_OUTPUT")"

"$APPTAINER" exec \
  --cleanenv \
  --bind /cvmfs:/cvmfs:ro \
  --bind "$TMP:/work" \
  --bind "$ROOT_PARENT:/input:ro" \
  --bind "$OUTPUT_ABS:/output" \
  "$IMAGE" \
  bash -lc '
set -Eeuo pipefail

export LC_ALL=C
export LANG=C

MEMBER="$1"
SEED="$2"
ROOT_NAME="$3"
EXPECTED_ROOT_SHA="$4"
EXPECTED_CANDIDATE_SHA="$5"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

test -r "$LCG_SETUP"

CLEAN_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

set +u
source "$LCG_SETUP"
set -u

if [[ -z "${PYTHONHOME:-}" ]]; then
  echo "ERROR: expected inherited LCG PYTHONHOME is absent"
  exit 70
fi

printf "%s\n" "$PYTHONHOME" \
  > "/output/logs/${MEMBER}_inherited_pythonhome.txt"

PYTHON_SITE="/work/environment/site-packages"
BOOTSTRAP="/work/payload/bootstrap"
PORTABLE_PYTHON="/usr/bin/python3"

BUILDER="/work/payload/repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
SCHEMA="/work/payload/repo/schema/canonical72_columns.tsv"

ROOT_PATH="/input/$ROOT_NAME"
SAMPLE="${MEMBER}_pythia8_delphes"
CANDIDATE="/output/parquet/${MEMBER}_pythia8_delphes_canonical72.parquet"
RECEIPT="/output/receipts/${MEMBER}_reconstruction_receipt.json"

run_portable_python() {
  env \
    -u PYTHONHOME \
    -u PYTHONPATH \
    LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$PYTHON_SITE:$BOOTSTRAP" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    "$PORTABLE_PYTHON" "$@"
}

run_portable_python - <<'"'"'PYENV'"'"'
import os
import sys

import awkward
import numpy
import pandas
import pyarrow
import uproot

assert "PYTHONHOME" not in os.environ
assert sys.version_info[:2] == (3, 9)

assert numpy.__version__ == "1.26.4"
assert awkward.__version__ == "2.6.4"
assert uproot.__version__ == "5.3.7"
assert pandas.__version__ == "2.2.2"
assert pyarrow.__version__ == "15.0.0"

print("RECONSTRUCTION_ONLY_PORTABLE_ENVIRONMENT_PASS")
PYENV

run_portable_python "$BUILDER" \
  --input "$ROOT_PATH" \
  --out "$CANDIDATE" \
  --sample "$SAMPLE" \
  --target-mass 125.0 \
  --jet-pt-min 30.0 \
  --jet-eta-max 2.5 \
  --btag-min 0.0 \
  --max-bjets-for-pairing 8 \
  --higgs-ordering pt \
  2>&1 | tee "/output/logs/${MEMBER}_canonical72.log"

run_portable_python - \
  "$ROOT_PATH" \
  "$CANDIDATE" \
  "$SCHEMA" \
  "$MEMBER" \
  "$SEED" \
  "$EXPECTED_ROOT_SHA" \
  "$EXPECTED_CANDIDATE_SHA" \
  "$RECEIPT" <<'"'"'PYVALIDATE'"'"'
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import uproot


(
    root_arg,
    candidate_arg,
    schema_arg,
    member,
    seed_arg,
    expected_root_sha,
    expected_candidate_sha,
    receipt_arg,
) = sys.argv[1:]

root_path = Path(root_arg)
candidate_path = Path(candidate_arg)
schema_path = Path(schema_arg)
receipt_path = Path(receipt_arg)
seed = int(seed_arg)
sample = f"{member}_pythia8_delphes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


root_sha = sha256(root_path)

if root_sha != expected_root_sha:
    raise SystemExit(
        "ERROR: bound ROOT SHA256 mismatch"
    )

with uproot.open(root_path) as source:
    entries = int(source["Delphes"].num_entries)

if entries != 10000:
    raise SystemExit(
        f"ERROR: Delphes entries={entries}, expected 10000"
    )

schema_rows = [
    line.split("\t")
    for line in schema_path.read_text(
        encoding="utf-8"
    ).splitlines()[1:]
    if line.strip()
]

expected_columns = [
    row[1]
    for row in schema_rows
]

expected_types = {
    row[1]: row[2]
    for row in schema_rows
}

table = pq.read_table(candidate_path)

if table.column_names != expected_columns:
    raise SystemExit(
        "ERROR: canonical column order mismatch"
    )

if len(table.column_names) != 72:
    raise SystemExit(
        "ERROR: canonical table does not have 72 columns"
    )

observed_types = {
    field.name: str(field.type)
    for field in table.schema
}

if observed_types != expected_types:
    raise SystemExit(
        "ERROR: canonical Arrow type mismatch"
    )

frame = table.to_pandas()

if frame.duplicated(
    ["sample", "event"]
).any():
    raise SystemExit(
        "ERROR: duplicate sample/event keys"
    )

if len(frame) and set(frame["sample"]) != {
    sample
}:
    raise SystemExit(
        "ERROR: sample identity mismatch"
    )

for column in frame.select_dtypes(
    include=[np.number]
).columns:
    if not np.isfinite(
        frame[column].to_numpy()
    ).all():
        raise SystemExit(
            f"ERROR: nonfinite values in {column}"
        )

candidate_sha = sha256(candidate_path)

if (
    expected_candidate_sha
    and candidate_sha != expected_candidate_sha
):
    raise SystemExit(
        "ERROR: canary candidate SHA256 mismatch"
    )

record = {
    "schema_version": 1,
    "timestamp_utc":
        datetime.now(timezone.utc).isoformat(),
    "status":
        "hh4b_ttbar7_reconstruction_only_member_pass",
    "campaign":
        "hh4b_ttbar7_reconstruction_only_recovery_20260728_v1",
    "member": member,
    "seed": seed,
    "sample": sample,
    "root_entries": entries,
    "root_sha256": root_sha,
    "candidate_rows": table.num_rows,
    "candidate_columns": len(table.column_names),
    "candidate_sha256": candidate_sha,
    "schema_names_order_exact": True,
    "schema_types_exact": True,
    "duplicate_candidate_keys": 0,
    "finite_numeric_values": True,
    "pythonhome_explicitly_unset": (
        "PYTHONHOME" not in os.environ
    ),
    "reconstruction_only": True,
    "madgraph_rerun": False,
    "pythia_rerun": False,
    "delphes_rerun": False,
}

receipt_path.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(json.dumps(record, indent=2, sort_keys=True))
print("RECONSTRUCTION_ONLY_MEMBER_VALIDATION_PASS")
PYVALIDATE
' _ \
  "$MEMBER" \
  "$SEED" \
  "$ROOT_NAME" \
  "$EXPECTED_ROOT_SHA" \
  "$EXPECTED_CANDIDATE_SHA"

(
  cd "$MEMBER_OUTPUT"

  find parquet logs receipts \
    -type f \
    ! -name "${MEMBER}_final_status.txt" \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "checksums/${MEMBER}_SHA256SUMS"

  sha256sum \
    -c "checksums/${MEMBER}_SHA256SUMS"
)

echo "RECONSTRUCTION_ONLY_MEMBER_COMPLETE=$MEMBER"
