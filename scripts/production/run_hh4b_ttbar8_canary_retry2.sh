#!/usr/bin/env bash
# Reconstruction-only worker for the HH4b ttbar8 canary retry2 contract.
set -Eeuo pipefail

EXPECTED_ROOT_NAME="ttbar_100k_shard003_pythia8_delphes.root"
EXPECTED_ROOT_SHA256="127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e"
EXPECTED_ROOT_BYTES="913864894"
EXPECTED_MEMBER="ttbar_100k_shard003"
EXPECTED_SEED="105003"
EXPECTED_SAMPLE="ttbar_100k_shard003_pythia8_delphes"
EXPECTED_EVENTS="10000"

PAYLOAD_TARBALL_NAME="hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
PAYLOAD_TARBALL_SHA256="3cc059b75cb63d27b57c51dfbd6643397fcb794e8e184ce689b4d905f53fad97"

PYENV_TARBALL_NAME="python39_site_packages.tar.gz"
PYENV_TARBALL_SHA256="6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"

BUILDER_SHA256="4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57"
WRITER_SHA256="abef7e4f5d1b82fe72837834b0b0794b59bb31ff16032b1b5a7f1519a6edbcfb"
POLICY_SHA256="4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68"
SCHEMA_SHA256="e9a12517772f77e311e37b09bc27df372735fee8dd48b176a0a82ada01913421"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 ROOT_NAME ROOT_SHA256 MEMBER SEED" >&2
  exit 64
fi

ROOT_NAME="$1"
ROOT_SHA256="$2"
MEMBER="$3"
SEED="$4"

if [[ "$ROOT_NAME" != "$EXPECTED_ROOT_NAME" \
   || "$ROOT_SHA256" != "$EXPECTED_ROOT_SHA256" \
   || "$MEMBER" != "$EXPECTED_MEMBER" \
   || "$SEED" != "$EXPECTED_SEED" ]]
then
  echo "ERROR: arguments do not match frozen retry2 identity" >&2
  exit 65
fi

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
OUTDIR="$SCRATCH/output"

CANDIDATE="$OUTDIR/parquet/${MEMBER}_pythia8_delphes_canonical72.parquet"
RECON_STDOUT="$OUTDIR/logs/${MEMBER}_reconstruction.stdout"
RECON_STDERR="$OUTDIR/logs/${MEMBER}_reconstruction.stderr"
CANONICAL_LOG="$OUTDIR/logs/${MEMBER}_canonical72.log"
ENVIRONMENT_JSON="$OUTDIR/logs/${MEMBER}_environment.json"
FINAL_STATUS="$OUTDIR/receipts/${MEMBER}_final_status.txt"
RECEIPT="$OUTDIR/receipts/${MEMBER}_receipt.json"
CHECKSUMS="$OUTDIR/checksums/${MEMBER}_SHA256SUMS"

CANDIDATE_ROWS="null"
PAYLOAD_VALIDATED="false"
PYTHON_ENV_VALIDATED="false"
ROOT_VALIDATED="false"
SCHEMA_VALIDATED="false"

mkdir -p "$OUTDIR"/{parquet,logs,receipts,checksums,audits}

finalize() {
  local payload_status=$?
  local outcome="failure"
  local candidate_exists="false"
  local candidate_sha256=""

  trap - EXIT
  set +e

  if [[ -s "$CANDIDATE" ]]; then
    candidate_exists="true"
    candidate_sha256="$(
      sha256sum "$CANDIDATE" 2>/dev/null | awk '{print $1}'
    )"
  elif [[ "$payload_status" -eq 0 ]]; then
    payload_status=90
  fi

  if [[ "$payload_status" -eq 0 ]]; then
    outcome="success"
  fi

  {
    echo "status=$outcome"
    echo "exit_status=$payload_status"
    echo "retry_campaign=hh4b_ttbar8_exact_regeneration_canary_retry2_20260727_v1"
    echo "retry_mode=canonical72_reconstruction_only_from_returned_root"
    echo "source_job=3654710.0"
    echo "failed_retry1_job=3654939.0"
    echo "member=$MEMBER"
    echo "seed=$SEED"
    echo "input_root_sha256=$EXPECTED_ROOT_SHA256"
    echo "payload_validated=$PAYLOAD_VALIDATED"
    echo "python_environment_validated=$PYTHON_ENV_VALIDATED"
    echo "root_validated=$ROOT_VALIDATED"
    echo "schema_validated=$SCHEMA_VALIDATED"
    echo "candidate_exists=$candidate_exists"
    echo "candidate_rows=$CANDIDATE_ROWS"
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$FINAL_STATUS"

  cat > "$RECEIPT" <<EOF
{
  "candidate_exists": $candidate_exists,
  "candidate_rows": $CANDIDATE_ROWS,
  "candidate_sha256": "$candidate_sha256",
  "environment_validated": $PYTHON_ENV_VALIDATED,
  "exit_status": $payload_status,
  "input_root_bytes": $EXPECTED_ROOT_BYTES,
  "input_root_entries": $EXPECTED_EVENTS,
  "input_root_name": "$EXPECTED_ROOT_NAME",
  "input_root_sha256": "$EXPECTED_ROOT_SHA256",
  "member": "$MEMBER",
  "outcome": "$outcome",
  "payload_validated": $PAYLOAD_VALIDATED,
  "python_environment_sha256": "$PYENV_TARBALL_SHA256",
  "retry_campaign": "hh4b_ttbar8_exact_regeneration_canary_retry2_20260727_v1",
  "retry_mode": "canonical72_reconstruction_only_from_returned_root",
  "root_validated": $ROOT_VALIDATED,
  "schema_columns": 72,
  "schema_validated": $SCHEMA_VALIDATED,
  "seed_provenance": $SEED,
  "source_job": "3654710.0",
  "failed_retry1_job": "3654939.0"
}
EOF

  (
    cd "$OUTDIR" || exit 0
    find parquet logs receipts audits -type f -print0 \
      | sort -z \
      | xargs -0 -r sha256sum \
      > "$CHECKSUMS"
  )

  exit "$payload_status"
}

trap finalize EXIT

if [[ "$(realpath "$SCRATCH")" != "/srv" ]]; then
  echo "ERROR: frozen worker scratch must resolve to /srv" >&2
  exit 66
fi

cd "$SCRATCH"

for INPUT in \
  "$PAYLOAD_TARBALL_NAME" \
  "$PYENV_TARBALL_NAME" \
  "$ROOT_NAME"
do
  if [[ ! -s "$INPUT" ]]; then
    echo "ERROR: missing retry2 input: $INPUT" >&2
    exit 67
  fi
done

test "$(
  sha256sum "$PAYLOAD_TARBALL_NAME" | awk '{print $1}'
)" = "$PAYLOAD_TARBALL_SHA256"

test "$(
  sha256sum "$PYENV_TARBALL_NAME" | awk '{print $1}'
)" = "$PYENV_TARBALL_SHA256"

observed_root_bytes="$(stat --printf='%s' "$ROOT_NAME")"
observed_root_sha256="$(sha256sum "$ROOT_NAME" | awk '{print $1}')"

if [[ "$observed_root_bytes" != "$EXPECTED_ROOT_BYTES" \
   || "$observed_root_sha256" != "$EXPECTED_ROOT_SHA256" ]]
then
  echo "ERROR: returned ROOT immutable identity mismatch" >&2
  exit 68
fi

rm -rf /srv/payload /srv/python39_env
tar -xzf "$PAYLOAD_TARBALL_NAME"
mkdir -p /srv/python39_env
tar -xzf "$PYENV_TARBALL_NAME" -C /srv/python39_env

PAYLOAD="/srv/payload"
REPO_PAYLOAD="$PAYLOAD/repo"
PYTHON_SITE="/srv/python39_env/site-packages"
PYTHON_BIN="/usr/bin/python3"

BUILDER="$REPO_PAYLOAD/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
WRITER="$REPO_PAYLOAD/scripts/delphes/write_parquet_from_pickle.py"
POLICY="$REPO_PAYLOAD/configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml"
SCHEMA="$REPO_PAYLOAD/schema/canonical72_columns.tsv"

for REQUIRED in \
  "$BUILDER" \
  "$WRITER" \
  "$POLICY" \
  "$SCHEMA" \
  "$PAYLOAD/bootstrap/sitecustomize.py" \
  "$PAYLOAD/SHA256SUMS" \
  "$PYTHON_SITE"
do
  if [[ ! -e "$REQUIRED" ]]; then
    echo "ERROR: missing extracted retry2 dependency: $REQUIRED" >&2
    exit 69
  fi
done

(
  cd "$PAYLOAD"
  sha256sum -c SHA256SUMS
)

test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$BUILDER_SHA256"
test "$(sha256sum "$WRITER" | awk '{print $1}')" = "$WRITER_SHA256"
test "$(sha256sum "$POLICY" | awk '{print $1}')" = "$POLICY_SHA256"
test "$(sha256sum "$SCHEMA" | awk '{print $1}')" = "$SCHEMA_SHA256"

PAYLOAD_VALIDATED="true"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PYTHON_SITE:$PAYLOAD/bootstrap"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LC_ALL=C
export LANG=C

"$PYTHON_BIN" - "$ENVIRONMENT_JSON" <<'PY'
import json
import platform
import sys

import awkward
import numpy
import pandas
import pyarrow
import uproot

expected = {
    "numpy": "1.26.4",
    "awkward": "2.6.4",
    "uproot": "5.3.7",
    "pandas": "2.2.2",
    "pyarrow": "15.0.0",
}

observed = {
    "numpy": numpy.__version__,
    "awkward": awkward.__version__,
    "uproot": uproot.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
}

if observed != expected:
    raise SystemExit(
        f"ERROR: retry2 Python environment mismatch: {observed}"
    )

record = {
    **observed,
    "parquet_backend": "pyarrow",
    "platform": platform.platform(),
    "python": sys.version.split()[0],
}

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

PYTHON_ENV_VALIDATED="true"

"$PYTHON_BIN" - \
  "$ROOT_NAME" \
  "$OUTDIR/audits/root_input_audit.json" <<'PY'
import json
import sys

import uproot
from uproot.source.futures import TrivialExecutor

path, output = sys.argv[1:]
required = [
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
    "Jet.Mass",
    "Jet.BTag",
    "Jet.Flavor",
]

executor = TrivialExecutor()

with uproot.open(
    path,
    decompression_executor=executor,
    interpretation_executor=executor,
) as source:
    tree = source["Delphes"]
    entries = int(tree.num_entries)
    present = {name: name in tree for name in required}

if entries != 10000 or not all(present.values()):
    raise SystemExit(
        "ERROR: returned ROOT tree, entry, or branch mismatch"
    )

record = {
    "entries": entries,
    "readable": True,
    "required_branches": present,
    "tree": "Delphes",
}

with open(output, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

ROOT_VALIDATED="true"

set +e

timeout --signal=TERM --kill-after=60s 45m \
  "$PYTHON_BIN" "$BUILDER" \
    --input "$ROOT_NAME" \
    --out "$CANDIDATE" \
    --sample "$EXPECTED_SAMPLE" \
    --target-mass 125.0 \
    --jet-pt-min 30.0 \
    --jet-eta-max 2.5 \
    --btag-min 0.0 \
    --max-bjets-for-pairing 8 \
    --higgs-ordering pt \
    > "$RECON_STDOUT" \
    2> "$RECON_STDERR"

reconstruction_status=$?
set -e

cat "$RECON_STDOUT" "$RECON_STDERR" > "$CANONICAL_LOG"

if [[ "$reconstruction_status" -ne 0 ]]; then
  echo "ERROR: canonical reconstruction exited $reconstruction_status" >&2
  exit "$reconstruction_status"
fi

if [[ ! -s "$CANDIDATE" ]]; then
  echo "ERROR: canonical reconstruction produced no Parquet" >&2
  exit 72
fi

CANDIDATE_ROWS="$(
"$PYTHON_BIN" - \
  "$CANDIDATE" \
  "$SCHEMA" \
  "$EXPECTED_SAMPLE" \
  "$OUTDIR/audits/schema_audit.json" \
  "$OUTDIR/audits/entry_candidate_accounting.json" <<'PY'
import csv
import json
import sys

import numpy as np
import pyarrow.parquet as pq

(
    candidate,
    schema_path,
    sample,
    schema_audit_path,
    accounting_path,
) = sys.argv[1:]

with open(schema_path, newline="", encoding="utf-8") as handle:
    schema_rows = list(csv.DictReader(handle, delimiter="\t"))

expected_names = [row["column_name"] for row in schema_rows]
expected_types = {
    row["column_name"]: row["arrow_type"]
    for row in schema_rows
}

table = pq.read_table(candidate)
observed_types = {
    field.name: str(field.type)
    for field in table.schema
}

if len(expected_names) != 72 or table.column_names != expected_names:
    raise SystemExit(
        "ERROR: exact canonical 72-column name/order mismatch"
    )

if observed_types != expected_types:
    raise SystemExit(
        "ERROR: canonical Arrow logical-type mismatch"
    )

frame = table.to_pandas()

duplicates = int(
    frame.duplicated(["sample", "event"]).sum()
)

if duplicates:
    raise SystemExit("ERROR: duplicate candidate keys")

if len(frame) and set(frame["sample"]) != {sample}:
    raise SystemExit("ERROR: source/member identity mismatch")

for column in frame.select_dtypes(include=[np.number]).columns:
    if not np.isfinite(frame[column].to_numpy()).all():
        raise SystemExit(
            f"ERROR: nonfinite required values in {column}"
        )

schema_audit = {
    "arrow_types_exact": True,
    "column_count": len(table.column_names),
    "column_names_and_order_exact": True,
    "parquet_readable": True,
}

accounting = {
    "candidate_rows": len(frame),
    "duplicate_candidate_keys": duplicates,
    "root_entries": 10000,
    "sample_identity": sample,
    "source_job": "3654710.0",
    "failed_retry1_job": "3654939.0",
}

for path, record in (
    (schema_audit_path, schema_audit),
    (accounting_path, accounting),
):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

print(len(frame))
PY
)"

SCHEMA_VALIDATED="true"

echo \
  "Retry2 reconstruction complete for $MEMBER with $CANDIDATE_ROWS candidates"
