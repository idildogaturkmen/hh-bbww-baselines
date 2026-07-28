#!/usr/bin/env bash
# Reconstruction-only worker for the frozen HH4b ttbar8 canary retry contract.
set -Eeuo pipefail

EXPECTED_ROOT_NAME="ttbar_100k_shard003_pythia8_delphes.root"
EXPECTED_ROOT_SHA256="127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e"
EXPECTED_ROOT_BYTES="913864894"
EXPECTED_MEMBER="ttbar_100k_shard003"
EXPECTED_SEED="105003"
EXPECTED_SAMPLE="ttbar_100k_shard003_pythia8_delphes"
EXPECTED_EVENTS="10000"
PAYLOAD_TARBALL_NAME="hh4b_ttbar8_exact_regeneration_canary_retry1_payload.tar.gz"
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
BUILDER_SHA256="4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57"
WRITER_SHA256="abef7e4f5d1b82fe72837834b0b0794b59bb31ff16032b1b5a7f1519a6edbcfb"
POLICY_SHA256="4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68"
SCHEMA_FILE_SHA256="e9a12517772f77e311e37b09bc27df372735fee8dd48b176a0a82ada01913421"

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
   || "$SEED" != "$EXPECTED_SEED" ]]; then
  echo "ERROR: arguments do not match the frozen retry identity" >&2
  exit 65
fi

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
OUTDIR="$SCRATCH/output"
CANDIDATE="$OUTDIR/parquet/${MEMBER}_pythia8_delphes_canonical72.parquet"
RECON_STDOUT="$OUTDIR/logs/${MEMBER}_reconstruction.stdout"
RECON_STDERR="$OUTDIR/logs/${MEMBER}_reconstruction.stderr"
CANONICAL_LOG="$OUTDIR/logs/${MEMBER}_canonical72.log"
FINAL_STATUS="$OUTDIR/receipts/${MEMBER}_final_status.txt"
RECEIPT="$OUTDIR/receipts/${MEMBER}_receipt.json"
CHECKSUMS="$OUTDIR/checksums/${MEMBER}_SHA256SUMS"
CANDIDATE_ROWS="null"
PAYLOAD_VALIDATED="false"
ROOT_VALIDATED="false"
ENVIRONMENT_VALIDATED="false"
SCHEMA_VALIDATED="false"

# The directory exists before any payload, input, environment, or reconstruction
# check so the EXIT trap always has a transferable destination.
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
    candidate_sha256="$(sha256sum "$CANDIDATE" 2>/dev/null | awk '{print $1}')"
  elif [[ "$payload_status" -eq 0 ]]; then
    payload_status=90
  fi
  if [[ "$payload_status" -eq 0 ]]; then
    outcome="success"
  fi

  {
    echo "status=$outcome"
    echo "exit_status=$payload_status"
    echo "retry_campaign=hh4b_ttbar8_exact_regeneration_canary_retry1_20260727_v1"
    echo "retry_mode=canonical72_reconstruction_only_from_returned_root"
    echo "source_job=3654710.0"
    echo "member=$MEMBER"
    echo "seed=$SEED"
    echo "input_root_sha256=$EXPECTED_ROOT_SHA256"
    echo "payload_validated=$PAYLOAD_VALIDATED"
    echo "root_validated=$ROOT_VALIDATED"
    echo "environment_validated=$ENVIRONMENT_VALIDATED"
    echo "schema_validated=$SCHEMA_VALIDATED"
    echo "candidate_exists=$candidate_exists"
    echo "candidate_rows=$CANDIDATE_ROWS"
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$FINAL_STATUS"

  printf '{\n' > "$RECEIPT"
  printf '  "candidate_exists": %s,\n' "$candidate_exists" >> "$RECEIPT"
  printf '  "candidate_rows": %s,\n' "$CANDIDATE_ROWS" >> "$RECEIPT"
  printf '  "candidate_sha256": "%s",\n' "$candidate_sha256" >> "$RECEIPT"
  printf '  "checksum_manifest": "checksums/%s_SHA256SUMS",\n' "$MEMBER" >> "$RECEIPT"
  printf '  "duplicate_candidate_keys": %s,\n' "$([[ "$SCHEMA_VALIDATED" == "true" ]] && echo 0 || echo null)" >> "$RECEIPT"
  printf '  "environment_report": "logs/%s_environment.json",\n' "$MEMBER" >> "$RECEIPT"
  printf '  "environment_validated": %s,\n' "$ENVIRONMENT_VALIDATED" >> "$RECEIPT"
  printf '  "exit_status": %s,\n' "$payload_status" >> "$RECEIPT"
  printf '  "input_root_bytes": %s,\n' "$EXPECTED_ROOT_BYTES" >> "$RECEIPT"
  printf '  "input_root_entries": %s,\n' "$EXPECTED_EVENTS" >> "$RECEIPT"
  printf '  "input_root_name": "%s",\n' "$EXPECTED_ROOT_NAME" >> "$RECEIPT"
  printf '  "input_root_sha256": "%s",\n' "$EXPECTED_ROOT_SHA256" >> "$RECEIPT"
  printf '  "member": "%s",\n' "$MEMBER" >> "$RECEIPT"
  printf '  "outcome": "%s",\n' "$outcome" >> "$RECEIPT"
  printf '  "payload_validated": %s,\n' "$PAYLOAD_VALIDATED" >> "$RECEIPT"
  printf '  "policy_sha256": "%s",\n' "$POLICY_SHA256" >> "$RECEIPT"
  printf '  "retry_campaign": "hh4b_ttbar8_exact_regeneration_canary_retry1_20260727_v1",\n' >> "$RECEIPT"
  printf '  "retry_mode": "canonical72_reconstruction_only_from_returned_root",\n' >> "$RECEIPT"
  printf '  "root_validated": %s,\n' "$ROOT_VALIDATED" >> "$RECEIPT"
  printf '  "schema_columns": 72,\n' >> "$RECEIPT"
  printf '  "schema_validated": %s,\n' "$SCHEMA_VALIDATED" >> "$RECEIPT"
  printf '  "seed_provenance": %s,\n' "$SEED" >> "$RECEIPT"
  printf '  "source_job": "3654710.0",\n' >> "$RECEIPT"
  printf '  "source_member": "%s",\n' "$MEMBER" >> "$RECEIPT"
  printf '  "writer_sha256": "%s"\n' "$WRITER_SHA256" >> "$RECEIPT"
  printf '}\n' >> "$RECEIPT"

  (
    cd "$OUTDIR" || exit 0
    find parquet logs receipts audits -type f \
      ! -path "checksums/${MEMBER}_SHA256SUMS" -print0 \
      | sort -z \
      | xargs -0 -r sha256sum > "$CHECKSUMS"
  )
  exit "$payload_status"
}
trap finalize EXIT

if [[ "$(realpath "$SCRATCH")" != "/srv" ]]; then
  echo "ERROR: frozen worker scratch must be /srv, observed: $(realpath "$SCRATCH")" >&2
  exit 66
fi
cd "$SCRATCH"

if [[ ! -s "$PAYLOAD_TARBALL_NAME" ]]; then
  echo "ERROR: missing retry payload: $PAYLOAD_TARBALL_NAME" >&2
  exit 67
fi
if [[ ! -s "$ROOT_NAME" ]]; then
  echo "ERROR: missing returned ROOT input: $ROOT_NAME" >&2
  exit 68
fi
if [[ ! -r "$LCG_SETUP" ]]; then
  echo "ERROR: missing frozen LCG environment: $LCG_SETUP" >&2
  exit 69
fi

tar -xzf "$PAYLOAD_TARBALL_NAME"
PAYLOAD="/srv/payload"
REPO_PAYLOAD="$PAYLOAD/repo"
BUILDER="$REPO_PAYLOAD/scripts/delphes/reconstruct_hh4b_candidates_v2.py"
WRITER="$REPO_PAYLOAD/scripts/delphes/write_parquet_from_pickle.py"
POLICY="$REPO_PAYLOAD/configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml"
SCHEMA="$REPO_PAYLOAD/schema/canonical72_columns.tsv"
PAYLOAD_WRAPPER="$REPO_PAYLOAD/scripts/production/run_hh4b_ttbar8_canary_retry1.sh"

# These exact /srv paths are part of the retry contract.
for required_path in \
  "$BUILDER" \
  "$WRITER" \
  "$POLICY" \
  "$SCHEMA" \
  "$PAYLOAD_WRAPPER" \
  "$PAYLOAD/bootstrap/sitecustomize.py" \
  "$PAYLOAD/SHA256SUMS"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: required extracted payload path absent: $required_path" >&2
    exit 70
  fi
done
(cd "$PAYLOAD" && sha256sum -c SHA256SUMS)
test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$BUILDER_SHA256"
test "$(sha256sum "$WRITER" | awk '{print $1}')" = "$WRITER_SHA256"
test "$(sha256sum "$POLICY" | awk '{print $1}')" = "$POLICY_SHA256"
test "$(sha256sum "$SCHEMA" | awk '{print $1}')" = "$SCHEMA_FILE_SHA256"
PAYLOAD_VALIDATED="true"

observed_root_bytes="$(stat --printf='%s' "$ROOT_NAME")"
observed_root_sha256="$(sha256sum "$ROOT_NAME" | awk '{print $1}')"
if [[ "$observed_root_bytes" != "$EXPECTED_ROOT_BYTES" \
   || "$observed_root_sha256" != "$EXPECTED_ROOT_SHA256" ]]; then
  echo "ERROR: returned ROOT immutable identity mismatch" >&2
  exit 71
fi

set +u
# shellcheck disable=SC1090
source "$LCG_SETUP"
set -u
export PYTHONPATH="$PAYLOAD/bootstrap"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export LC_ALL=C
export LANG=C

python3 - "$ROOT_NAME" "$OUTDIR/audits/root_input_audit.json" <<'PY'
import json
import sys

import uproot

path, output = sys.argv[1:]
required = ["Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag", "Jet.Flavor"]
with uproot.open(path) as source:
    tree = source["Delphes"]
    entries = int(tree.num_entries)
    present = {name: name in tree for name in required}
if entries != 10000 or not all(present.values()):
    raise SystemExit("ERROR: returned ROOT tree, entry, or branch contract mismatch")
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

python3 - "$OUTDIR/logs/${MEMBER}_environment.json" <<'PY'
import json
import platform
import sys

import awkward
import numpy
import pandas
import pyarrow
import uproot

record = {
    "awkward": awkward.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "parquet_backend": "pyarrow",
    "platform": platform.platform(),
    "pyarrow": pyarrow.__version__,
    "python": sys.version.split()[0],
    "uproot": uproot.__version__,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
ENVIRONMENT_VALIDATED="true"

# The policy is used explicitly by exact path/checksum verification above and
# by spelling out every frozen policy choice below. The protected builder has
# no --policy CLI option, so no numerical implementation is altered.
set +e
timeout --signal=TERM --kill-after=60s 45m \
  python3 "$BUILDER" \
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
python3 - "$CANDIDATE" "$SCHEMA" "$EXPECTED_SAMPLE" \
  "$OUTDIR/audits/schema_audit.json" \
  "$OUTDIR/audits/entry_candidate_accounting.json" <<'PY'
import csv
import json
import sys

import numpy as np
import pyarrow.parquet as pq

candidate, schema_path, sample, schema_audit_path, accounting_path = sys.argv[1:]
with open(schema_path, newline="", encoding="utf-8") as handle:
    schema_rows = list(csv.DictReader(handle, delimiter="\t"))
expected_names = [row["column_name"] for row in schema_rows]
expected_types = {row["column_name"]: row["arrow_type"] for row in schema_rows}
table = pq.read_table(candidate)
observed_types = {field.name: str(field.type) for field in table.schema}
if len(expected_names) != 72 or table.column_names != expected_names:
    raise SystemExit("ERROR: exact canonical 72-column name/order mismatch")
if observed_types != expected_types:
    raise SystemExit("ERROR: canonical Arrow logical-type mismatch")
frame = table.to_pandas()
duplicates = int(frame.duplicated(["sample", "event"]).sum())
if duplicates:
    raise SystemExit("ERROR: duplicate candidate keys")
if len(frame) and set(frame["sample"]) != {sample}:
    raise SystemExit("ERROR: source/member identity mismatch")
for column in frame.select_dtypes(include=[np.number]).columns:
    if not np.isfinite(frame[column].to_numpy()).all():
        raise SystemExit(f"ERROR: nonfinite required values in {column}")
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

echo "Reconstruction-only retry complete for $MEMBER with $CANDIDATE_ROWS candidates"
