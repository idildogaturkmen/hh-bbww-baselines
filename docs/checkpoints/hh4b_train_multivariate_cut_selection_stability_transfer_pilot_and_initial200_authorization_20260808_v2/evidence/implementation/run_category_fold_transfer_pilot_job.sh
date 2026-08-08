#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "usage: $0 JOB_INDEX REPLICA OUTER_FOLD CATEGORY_ID EXPECTED_HEAD PAYLOAD_BASENAME EXPECTED_PAYLOAD_SHA EXPECTED_RUNTIME_SHA" >&2
    exit 64
fi

JOB_INDEX="$1"
REPLICA="$2"
OUTER_FOLD="$3"
CATEGORY_ID="$4"
EXPECTED_HEAD="$5"
PAYLOAD_BASENAME="$6"
EXPECTED_PAYLOAD_SHA="$7"
EXPECTED_RUNTIME_SHA="$8"

RUNTIME_BASENAME="hh4b_python_runtime_v8_r1.tar.gz"

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

RUNNER_LOG="$SCRATCH/runner.log"
RESULT_BUNDLE="$SCRATCH/result_bundle.tar.gz"
RECEIPT="$SCRATCH/job_receipt.json"
WORK_ROOT="$SCRATCH/work"
PAYLOAD="$SCRATCH/payload"
RUNTIME="$SCRATCH/runtime"
RESULT_ROOT="$WORK_ROOT/results"

RUNNER_RC=not_run
STRUCTURES_ATTEMPTED=0
STRUCTURES_COMPLETED=0
FINALIZED=FALSE
START_EPOCH="$(date +%s)"

finalize() {
    local shell_rc="$?"
    set +e

    if [[ "$FINALIZED" == "TRUE" ]]; then
        exit "$shell_rc"
    fi
    FINALIZED=TRUE

    [[ -f "$RUNNER_LOG" ]] || : > "$RUNNER_LOG"
    mkdir -p "$WORK_ROOT"

    RECEIPT_PYTHON=""
    if [[ -x "$RUNTIME/bin/python" ]]; then
        RECEIPT_PYTHON="$RUNTIME/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        RECEIPT_PYTHON="$(command -v python3)"
    fi

    END_EPOCH="$(date +%s)"
    ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

    if [[ -n "$RECEIPT_PYTHON" && -d "$RESULT_ROOT" ]]; then
        RESULT_ROOT_VALUE="$RESULT_ROOT" \
        CATEGORY_ID_VALUE="$CATEGORY_ID" \
        OUTER_FOLD_VALUE="$OUTER_FOLD" \
        "$RECEIPT_PYTHON" - <<'PY_AUDIT' >> "$RUNNER_LOG" 2>&1
from pathlib import Path
import json, os

root = Path(os.environ["RESULT_ROOT_VALUE"])
category = os.environ["CATEGORY_ID_VALUE"]
outer = int(os.environ["OUTER_FOLD_VALUE"])

results = sorted(root.glob("*/structure_result.json"))
print(f"RETURNED_STRUCTURE_RESULT_COUNT={len(results)}")

valid = 0
for p in results:
    x = json.loads(p.read_text())
    assert x["status"] == "pass_structure_job_complete"
    assert x["category_id"] == category
    assert int(x["outer_fold"]) == outer
    assert x["outer_fold_used_for_selection"] is False
    assert int(x["validation_payloads_opened"]) == 0
    assert int(x["test_payloads_opened"]) == 0
    valid += 1

print(f"VALID_STRUCTURE_RESULT_COUNT={valid}")
PY_AUDIT
    fi

    tar -C "$WORK_ROOT" -czf "$RESULT_BUNDLE" . 2>>"$RUNNER_LOG" || true

    if [[ -n "$RECEIPT_PYTHON" ]]; then
        RECEIPT_VALUE="$RECEIPT" \
        RESULT_BUNDLE_VALUE="$RESULT_BUNDLE" \
        RUNNER_LOG_VALUE="$RUNNER_LOG" \
        RESULT_ROOT_VALUE="$RESULT_ROOT" \
        JOB_INDEX_VALUE="$JOB_INDEX" \
        REPLICA_VALUE="$REPLICA" \
        OUTER_FOLD_VALUE="$OUTER_FOLD" \
        CATEGORY_ID_VALUE="$CATEGORY_ID" \
        EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" \
        RUNNER_RC_VALUE="$RUNNER_RC" \
        STRUCTURES_ATTEMPTED_VALUE="$STRUCTURES_ATTEMPTED" \
        STRUCTURES_COMPLETED_VALUE="$STRUCTURES_COMPLETED" \
        EXPECTED_PAYLOAD_SHA_VALUE="$EXPECTED_PAYLOAD_SHA" \
        EXPECTED_RUNTIME_SHA_VALUE="$EXPECTED_RUNTIME_SHA" \
        ELAPSED_SECONDS_VALUE="$ELAPSED_SECONDS" \
        "$RECEIPT_PYTHON" - <<'PY_RECEIPT'
from pathlib import Path
import hashlib, json, os

def sha(path: Path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

result_root = Path(os.environ["RESULT_ROOT_VALUE"])
result_paths = sorted(result_root.glob("*/structure_result.json")) if result_root.is_dir() else []

valid = 0
support_pass = 0
feasible = 0
for p in result_paths:
    try:
        x = json.loads(p.read_text())
        if (
            x.get("status") == "pass_structure_job_complete"
            and x.get("outer_fold_used_for_selection") is False
            and int(x.get("validation_payloads_opened", -1)) == 0
            and int(x.get("test_payloads_opened", -1)) == 0
        ):
            valid += 1
            support_pass += int(bool(x.get("pooled_support_pass")))
            feasible += int(bool(x.get("pooled_inner_oof_feasible")))
    except Exception:
        pass

runner_rc = os.environ["RUNNER_RC_VALUE"]
payload = {
    "schema_version": 1,
    "status": (
        "category_fold_transfer_pilot_job_complete"
        if runner_rc == "0" and valid == 27
        else "category_fold_transfer_pilot_job_failed"
    ),
    "job_index": int(os.environ["JOB_INDEX_VALUE"]),
    "bootstrap_replica": int(os.environ["REPLICA_VALUE"]),
    "outer_fold": int(os.environ["OUTER_FOLD_VALUE"]),
    "category_id": os.environ["CATEGORY_ID_VALUE"],
    "expected_repository_head": os.environ["EXPECTED_HEAD_VALUE"],
    "runner_rc": runner_rc,
    "structures_expected": 27,
    "structures_attempted": int(os.environ["STRUCTURES_ATTEMPTED_VALUE"]),
    "structures_completed_shell": int(os.environ["STRUCTURES_COMPLETED_VALUE"]),
    "valid_structure_results": valid,
    "pooled_support_pass_results": support_pass,
    "pooled_feasible_results": feasible,
    "expected_payload_archive_sha256": os.environ["EXPECTED_PAYLOAD_SHA_VALUE"],
    "expected_runtime_archive_sha256": os.environ["EXPECTED_RUNTIME_SHA_VALUE"],
    "result_bundle_sha256": sha(Path(os.environ["RESULT_BUNDLE_VALUE"])),
    "runner_log_sha256": sha(Path(os.environ["RUNNER_LOG_VALUE"])),
    "elapsed_seconds": int(os.environ["ELAPSED_SECONDS_VALUE"]),
    "pilot_results_may_enter_stability_aggregation": False,
    "initial_200_production_authorized": False,
    "validation_payloads_opened": 0,
    "test_payloads_opened": 0,
}
Path(os.environ["RECEIPT_VALUE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY_RECEIPT
    fi

    if [[ "$RUNNER_RC" == "not_run" ]]; then
        exit "$shell_rc"
    fi
    exit "$RUNNER_RC"
}
trap finalize EXIT

echo "CATEGORY_FOLD_TRANSFER_PILOT_START" >> "$RUNNER_LOG"
echo "JOB_INDEX=$JOB_INDEX" >> "$RUNNER_LOG"
echo "BOOTSTRAP_REPLICA=$REPLICA" >> "$RUNNER_LOG"
echo "OUTER_FOLD=$OUTER_FOLD" >> "$RUNNER_LOG"
echo "CATEGORY_ID=$CATEGORY_ID" >> "$RUNNER_LOG"

[[ -f "$PAYLOAD_BASENAME" ]]
[[ -f "$RUNTIME_BASENAME" ]]

ACTUAL_PAYLOAD_SHA="$(sha256sum "$PAYLOAD_BASENAME" | awk '{print $1}')"
ACTUAL_RUNTIME_SHA="$(sha256sum "$RUNTIME_BASENAME" | awk '{print $1}')"

echo "ACTUAL_PAYLOAD_SHA=$ACTUAL_PAYLOAD_SHA" >> "$RUNNER_LOG"
echo "ACTUAL_RUNTIME_SHA=$ACTUAL_RUNTIME_SHA" >> "$RUNNER_LOG"

[[ "$ACTUAL_PAYLOAD_SHA" == "$EXPECTED_PAYLOAD_SHA" ]]
[[ "$ACTUAL_RUNTIME_SHA" == "$EXPECTED_RUNTIME_SHA" ]]

mkdir -p "$RUNTIME" "$WORK_ROOT"
tar -C "$SCRATCH" -xzf "$PAYLOAD_BASENAME"
tar -C "$RUNTIME" -xzf "$RUNTIME_BASENAME"

(
    cd "$PAYLOAD"
    sha256sum -c SHA256SUMS
) >> "$RUNNER_LOG" 2>&1

PYTHON="$RUNTIME/bin/python"
[[ -x "$PYTHON" ]]

"$PYTHON" - <<'PY_IMPORTS' >> "$RUNNER_LOG" 2>&1
import numpy, pandas, pyarrow
print("TRANSFERRED_IMPORTS=PASS")
print("NUMPY_VERSION=" + numpy.__version__)
print("PANDAS_VERSION=" + pandas.__version__)
print("PYARROW_VERSION=" + pyarrow.__version__)
PY_IMPORTS

ACTIVE_CONFIG="$WORK_ROOT/runtime_config.json"

PAYLOAD_VALUE="$PAYLOAD" \
RUNTIME_VALUE="$RUNTIME" \
ACTIVE_CONFIG_VALUE="$ACTIVE_CONFIG" \
"$PYTHON" - <<'PY_CONFIG'
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

mapfile -t STRUCTURES < "$PAYLOAD/structure_ids.txt"
[[ "${#STRUCTURES[@]}" -eq 27 ]]

RUNNER_RC=0

for STRUCTURE_ID in "${STRUCTURES[@]}"; do
    STRUCTURES_ATTEMPTED="$((STRUCTURES_ATTEMPTED + 1))"
    echo "RUN_STRUCTURE_INDEX=$STRUCTURES_ATTEMPTED STRUCTURE_ID=$STRUCTURE_ID" >> "$RUNNER_LOG"

    set +e
    "$PYTHON" \
        "$PAYLOAD/code/hh4b_multivariate_cut_structure_worker.py" \
        --config "$ACTIVE_CONFIG" \
        --input-root "$PAYLOAD/fold_tables" \
        --output-root "$RESULT_ROOT" \
        --outer-fold "$OUTER_FOLD" \
        --category-id "$CATEGORY_ID" \
        --structure-id "$STRUCTURE_ID" \
        --expected-repository-head "$EXPECTED_HEAD" \
        --execution-provenance "$PAYLOAD/execution_provenance.json" \
        >> "$RUNNER_LOG" 2>&1
    RC="$?"
    set -e

    if [[ "$RC" -ne 0 ]]; then
        echo "STRUCTURE_WORKER_FAILURE structure=$STRUCTURE_ID rc=$RC" >> "$RUNNER_LOG"
        RUNNER_RC="$RC"
        break
    fi

    STRUCTURES_COMPLETED="$((STRUCTURES_COMPLETED + 1))"
done

if [[ "$STRUCTURES_COMPLETED" -ne 27 ]]; then
    RUNNER_RC=70
fi

echo "STRUCTURES_ATTEMPTED=$STRUCTURES_ATTEMPTED" >> "$RUNNER_LOG"
echo "STRUCTURES_COMPLETED=$STRUCTURES_COMPLETED" >> "$RUNNER_LOG"
echo "RUNNER_RC=$RUNNER_RC" >> "$RUNNER_LOG"

exit "$RUNNER_RC"
