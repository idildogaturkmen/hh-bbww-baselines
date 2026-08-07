#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 JOB_INDEX OUTER_FOLD CATEGORY_ID STRUCTURE_ID EXPECTED_HEAD" >&2
    exit 64
fi

JOB_INDEX="$1"
OUTER_FOLD="$2"
CATEGORY_ID="$3"
STRUCTURE_ID="$4"
EXPECTED_HEAD="$5"

EXPECTED_PAYLOAD_SHA="90c02073602bb47e978660b9cf8e6105794b77219f2fd1c0337860cab06bc708"
EXPECTED_RUNTIME_SHA="4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

WORKER_LOG="$SCRATCH/worker.log"
RESULT_BUNDLE="$SCRATCH/result_bundle.tar.gz"
RECEIPT="$SCRATCH/job_receipt.json"
WORK_ROOT="$SCRATCH/work"
PAYLOAD="$SCRATCH/payload"
RUNTIME="$SCRATCH/runtime"
OUTPUT_ROOT="$WORK_ROOT/results"

WORKER_RC=not_run
FINALIZED=FALSE

finalize() {
    local shell_rc="$?"
    set +e

    if [[ "$FINALIZED" == "TRUE" ]]; then
        exit "$shell_rc"
    fi
    FINALIZED=TRUE

    [[ -f "$WORKER_LOG" ]] || : > "$WORKER_LOG"

    if [[ -d "$WORK_ROOT" ]]; then
        tar -C "$WORK_ROOT" -czf "$RESULT_BUNDLE" .
    else
        mkdir -p "$WORK_ROOT"
        cp "$WORKER_LOG" "$WORK_ROOT/worker.log"
        tar -C "$WORK_ROOT" -czf "$RESULT_BUNDLE" .
    fi

    RECEIPT_PYTHON=""
    if [[ -x "$RUNTIME/bin/python" ]]; then
        RECEIPT_PYTHON="$RUNTIME/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        RECEIPT_PYTHON="$(command -v python3)"
    fi

    if [[ -n "$RECEIPT_PYTHON" ]]; then
        RECEIPT_VALUE="$RECEIPT"         RESULT_BUNDLE_VALUE="$RESULT_BUNDLE"         WORKER_LOG_VALUE="$WORKER_LOG"         JOB_INDEX_VALUE="$JOB_INDEX"         OUTER_FOLD_VALUE="$OUTER_FOLD"         CATEGORY_ID_VALUE="$CATEGORY_ID"         STRUCTURE_ID_VALUE="$STRUCTURE_ID"         EXPECTED_HEAD_VALUE="$EXPECTED_HEAD"         WORKER_RC_VALUE="$WORKER_RC"         EXPECTED_PAYLOAD_SHA_VALUE="$EXPECTED_PAYLOAD_SHA"         EXPECTED_RUNTIME_SHA_VALUE="$EXPECTED_RUNTIME_SHA"         "$RECEIPT_PYTHON" - <<'PY_RECEIPT'
from pathlib import Path
import hashlib
import json
import os


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


receipt = Path(os.environ["RECEIPT_VALUE"])
bundle = Path(os.environ["RESULT_BUNDLE_VALUE"])
worker_log = Path(os.environ["WORKER_LOG_VALUE"])

payload = {
    "schema_version": 1,
    "status": (
        "pilot_job_complete"
        if os.environ["WORKER_RC_VALUE"] == "0"
        else "pilot_job_failed"
    ),
    "job_index": int(os.environ["JOB_INDEX_VALUE"]),
    "outer_fold": int(os.environ["OUTER_FOLD_VALUE"]),
    "category_id": os.environ["CATEGORY_ID_VALUE"],
    "structure_id": os.environ["STRUCTURE_ID_VALUE"],
    "expected_repository_head": os.environ["EXPECTED_HEAD_VALUE"],
    "worker_rc": os.environ["WORKER_RC_VALUE"],
    "expected_payload_archive_sha256": (
        os.environ["EXPECTED_PAYLOAD_SHA_VALUE"]
    ),
    "expected_runtime_archive_sha256": (
        os.environ["EXPECTED_RUNTIME_SHA_VALUE"]
    ),
    "result_bundle_sha256": sha(bundle) if bundle.is_file() else None,
    "worker_log_sha256": sha(worker_log) if worker_log.is_file() else None,
    "validation_payloads_opened": 0,
    "test_payloads_opened": 0,
    "full_270_job_submission_authorized": False,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY_RECEIPT
    else
        BUNDLE_SHA="$(sha256sum "$RESULT_BUNDLE" | awk '{print $1}')"
        WORKER_LOG_SHA="$(sha256sum "$WORKER_LOG" | awk '{print $1}')"
        STATUS="pilot_job_failed"
        if [[ "$WORKER_RC" == "0" ]]; then
            STATUS="pilot_job_complete"
        fi
        cat > "$RECEIPT" <<JSON_RECEIPT
{
  "schema_version": 1,
  "status": "$STATUS",
  "job_index": $JOB_INDEX,
  "outer_fold": $OUTER_FOLD,
  "category_id": "$CATEGORY_ID",
  "structure_id": "$STRUCTURE_ID",
  "expected_repository_head": "$EXPECTED_HEAD",
  "worker_rc": "$WORKER_RC",
  "expected_payload_archive_sha256": "$EXPECTED_PAYLOAD_SHA",
  "expected_runtime_archive_sha256": "$EXPECTED_RUNTIME_SHA",
  "result_bundle_sha256": "$BUNDLE_SHA",
  "worker_log_sha256": "$WORKER_LOG_SHA",
  "validation_payloads_opened": 0,
  "test_payloads_opened": 0,
  "full_270_job_submission_authorized": false
}
JSON_RECEIPT
    fi

    if [[ "$WORKER_RC" == "not_run" ]]; then
        exit "$shell_rc"
    fi
    exit "$WORKER_RC"
}
trap finalize EXIT

[[ -f hh4b_multivariate_cut_payload_v8_r1.tar.gz ]]
[[ -f hh4b_python_runtime_v8_r1.tar.gz ]]

ACTUAL_PAYLOAD_SHA="$(sha256sum hh4b_multivariate_cut_payload_v8_r1.tar.gz | awk '{print $1}')"
ACTUAL_RUNTIME_SHA="$(sha256sum hh4b_python_runtime_v8_r1.tar.gz | awk '{print $1}')"

[[ "$ACTUAL_PAYLOAD_SHA" == "$EXPECTED_PAYLOAD_SHA" ]]
[[ "$ACTUAL_RUNTIME_SHA" == "$EXPECTED_RUNTIME_SHA" ]]

mkdir -p "$RUNTIME" "$WORK_ROOT"
tar -C "$SCRATCH" -xzf hh4b_multivariate_cut_payload_v8_r1.tar.gz
tar -C "$RUNTIME" -xzf hh4b_python_runtime_v8_r1.tar.gz

(
    cd "$PAYLOAD"
    sha256sum -c SHA256SUMS
)

PYTHON="$RUNTIME/bin/python"
[[ -x "$PYTHON" ]]

"$PYTHON" - <<'PY_IMPORTS' >> "$WORKER_LOG" 2>&1
import numpy
import pandas
import pyarrow
print("TRANSFERRED_IMPORTS=PASS")
print("NUMPY_VERSION=" + numpy.__version__)
print("PANDAS_VERSION=" + pandas.__version__)
print("PYARROW_VERSION=" + pyarrow.__version__)
PY_IMPORTS

ACTIVE_CONFIG="$WORK_ROOT/runtime_config.json"

PAYLOAD_VALUE="$PAYLOAD" RUNTIME_VALUE="$RUNTIME" ACTIVE_CONFIG_VALUE="$ACTIVE_CONFIG" "$PYTHON" - <<'PY_CONFIG'
from pathlib import Path
import os

payload = os.environ["PAYLOAD_VALUE"]
runtime = os.environ["RUNTIME_VALUE"]
template = Path(payload) / "config" / "runtime_config.template.json"
active = Path(os.environ["ACTIVE_CONFIG_VALUE"])

text = template.read_text()
text = text.replace("__PAYLOAD_ROOT__", payload)
text = text.replace("__RUNTIME_ROOT__", runtime)

if "__PAYLOAD_ROOT__" in text or "__RUNTIME_ROOT__" in text:
    raise RuntimeError("unresolved transfer placeholder")

active.write_text(text)
PY_CONFIG

export PYTHONPATH="$PAYLOAD/code"
set +e
"$PYTHON"     "$PAYLOAD/code/hh4b_multivariate_cut_structure_worker.py"     --config "$ACTIVE_CONFIG"     --input-root "$PAYLOAD/fold_tables"     --output-root "$OUTPUT_ROOT"     --outer-fold "$OUTER_FOLD"     --category-id "$CATEGORY_ID"     --structure-id "$STRUCTURE_ID"     --expected-repository-head "$EXPECTED_HEAD"     --execution-provenance "$PAYLOAD/execution_provenance.json"     >> "$WORKER_LOG" 2>&1
WORKER_RC="$?"
set -e

exit "$WORKER_RC"
