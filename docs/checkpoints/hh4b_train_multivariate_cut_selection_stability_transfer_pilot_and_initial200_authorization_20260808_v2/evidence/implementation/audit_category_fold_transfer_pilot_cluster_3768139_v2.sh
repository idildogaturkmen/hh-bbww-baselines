#!/usr/bin/env bash
set -u -o pipefail

REPO="/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"
BRANCH="delphes-hh4b-production"

EXPECTED_HEAD="5b8447897cc98f7df00d460274b857ec882266b8"
EXPECTED_SUBJECT="docs: freeze stability full-budget pilot and granularity"

SCHEDD="lpcschedd4.fnal.gov"
CLUSTER_ID="3768139"
EXPECTED_JOB_COUNT=4

PILOT_ROOT="/uscms_data/d3/iturkmen/hh4b_delphes/baselines/multivariate_cut_selection_stability_category_fold_transfer_pilot_v1_20260808"
SUBMISSION_ROOT="$PILOT_ROOT/submission"
RETURN_ROOT="$PILOT_ROOT/returns"
LOG_ROOT="$PILOT_ROOT/logs"
AUDIT_ROOT="$PILOT_ROOT/audit"

MATRIX="$SUBMISSION_ROOT/pilot_matrix.tsv"
QUEUE_ITEMS="$SUBMISSION_ROOT/pilot_queue.items"
SUBFILE="$SUBMISSION_ROOT/category_fold_transfer_pilot.sub"
SUBMIT_OUT="$SUBMISSION_ROOT/condor_submit_output.txt"
SUBMISSION_RECEIPT="$SUBMISSION_ROOT/pilot_submission_receipt.json"
PACKAGE_AUDIT="$SUBMISSION_ROOT/package_build_audit.json"
RUNNER="$SUBMISSION_ROOT/run_category_fold_transfer_pilot_job.sh"

EXPECTED_RUNNER_SHA="7e1871e292f18ddaa2da5cd5bccbbe5b5659cca24e40d8b9815bbdd039d808ba"
EXPECTED_WORKER_SHA="739147c33d6597018f4dce8c483278c64ff8b41533de1f1b213749dfb39332ed"
EXPECTED_OPTIMIZER_SHA="7760ee7ae309bec6cd47eb1c73ed04ec12bb805f133527498fcb75aca8088a62"
EXPECTED_RUNTIME_SHA="4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112"

AMENDMENT="$REPO/docs/checkpoints/hh4b_train_multivariate_cut_category_endpoint_amendment_20260806_v1/multivariate_cut_category_endpoint_amendment.json"

PYTHON="/uscms_data/d3/iturkmen/hh4b_delphes/runtime/hh4b_broad_ml_py39_v1/bin/python"

QUEUE_SNAPSHOT="$AUDIT_ROOT/queue_snapshot.txt"
HISTORY_SNAPSHOT="$AUDIT_ROOT/history_snapshot.txt"
AUDIT_JSON="$AUDIT_ROOT/category_fold_transfer_pilot_audit_v2.json"
AUDIT_TSV="$AUDIT_ROOT/category_fold_transfer_pilot_audit_v2.tsv"
AUDIT_LOG="$AUDIT_ROOT/category_fold_transfer_pilot_audit_v2.txt"

die() {
    echo "============================================================"
    echo "CATEGORY-FOLD TRANSFER PILOT AUDIT FAILURE"
    echo "============================================================"
    echo "PASS=FALSE"
    echo "ERROR=$1"
    echo "DO_NOT_RESUBMIT_CLUSTER_3768139=TRUE"
    echo "INITIAL_200_PRODUCTION_AUTHORIZED=FALSE"
    echo "VALIDATION_PAYLOADS_OPENED=0"
    echo "TEST_PAYLOADS_OPENED=0"
    exit 1
}

cd "$REPO" || die "cannot enter repository"

mkdir -p "$AUDIT_ROOT" || die "cannot create audit root"

{
echo "============================================================"
echo "AUDIT BOUNDED CATEGORY-FOLD TRANSFER PILOT V2"
echo "============================================================"

echo
echo "===== 1. FROZEN REPOSITORY / SUBMISSION PREFLIGHT ====="

git fetch --prune origin "$BRANCH" >/dev/null 2>&1 || die "git fetch failed"

CURRENT_BRANCH="$(git branch --show-current)"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "origin/$BRANCH")"
HEAD_SUBJECT="$(git show -s --format='%s' HEAD)"
TRACKED_DIRTY_COUNT="$(
    git status --porcelain=v1 --untracked-files=no |
    awk 'NF {n++} END {print n+0}'
)"

echo "CURRENT_BRANCH=$CURRENT_BRANCH"
echo "LOCAL_HEAD=$LOCAL_HEAD"
echo "REMOTE_HEAD=$REMOTE_HEAD"
echo "HEAD_SUBJECT=$HEAD_SUBJECT"
echo "TRACKED_DIRTY_COUNT=$TRACKED_DIRTY_COUNT"

[[ "$CURRENT_BRANCH" == "$BRANCH" ]] || die "wrong branch"
[[ "$LOCAL_HEAD" == "$EXPECTED_HEAD" ]] || die "unexpected local HEAD"
[[ "$REMOTE_HEAD" == "$EXPECTED_HEAD" ]] || die "unexpected remote HEAD"
[[ "$HEAD_SUBJECT" == "$EXPECTED_SUBJECT" ]] || die "unexpected HEAD subject"
[[ "$TRACKED_DIRTY_COUNT" -eq 0 ]] || die "tracked worktree is dirty"

for required in \
    "$MATRIX" \
    "$QUEUE_ITEMS" \
    "$SUBFILE" \
    "$SUBMIT_OUT" \
    "$SUBMISSION_RECEIPT" \
    "$PACKAGE_AUDIT" \
    "$RUNNER" \
    "$AMENDMENT" \
    "$PYTHON"
do
    [[ -e "$required" ]] || die "missing required artifact: $required"
done

ACTUAL_RUNNER_SHA="$(sha256sum "$RUNNER" | awk '{print $1}')"
echo "EXPECTED_RUNNER_SHA=$EXPECTED_RUNNER_SHA"
echo "ACTUAL_RUNNER_SHA=$ACTUAL_RUNNER_SHA"
[[ "$ACTUAL_RUNNER_SHA" == "$EXPECTED_RUNNER_SHA" ]] || die "runner SHA mismatch"

SUBMISSION_RECEIPT_VALUE="$SUBMISSION_RECEIPT" \
PACKAGE_AUDIT_VALUE="$PACKAGE_AUDIT" \
MATRIX_VALUE="$MATRIX" \
SUBMIT_OUT_VALUE="$SUBMIT_OUT" \
EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" \
CLUSTER_ID_VALUE="$CLUSTER_ID" \
"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json, os, re
import pandas as pd

def require(c, m):
    if not c:
        raise RuntimeError(m)

receipt_path = Path(os.environ["SUBMISSION_RECEIPT_VALUE"])
package_path = Path(os.environ["PACKAGE_AUDIT_VALUE"])
matrix_path = Path(os.environ["MATRIX_VALUE"])
submit_out_path = Path(os.environ["SUBMIT_OUT_VALUE"])
expected_head = os.environ["EXPECTED_HEAD_VALUE"]
cluster = int(os.environ["CLUSTER_ID_VALUE"])

receipt = json.loads(receipt_path.read_text())
package = json.loads(package_path.read_text())
matrix = pd.read_csv(matrix_path, sep="\t")
submit_text = submit_out_path.read_text()

require(receipt["submission_occurred"] is True, "submission receipt says no submission")
require(receipt["cluster_parse_pass"] is True, "submission cluster parse did not pass")
require(int(receipt["cluster_id"]) == cluster, "cluster ID mismatch")
require(int(receipt["first_proc"]) == 0, "first proc mismatch")
require(int(receipt["last_proc"]) == 3, "last proc mismatch")
require(int(receipt["parsed_job_count"]) == 4, "parsed job count mismatch")
require(receipt["repository_head"] == expected_head, "receipt repository head mismatch")
require(receipt["initial_200_production_authorized"] is False, "production unexpectedly authorized")
require(receipt["validation_payloads_opened"] == 0, "validation access in receipt")
require(receipt["test_payloads_opened"] == 0, "test access in receipt")

require(package["status"] == "pass_category_fold_transfer_payload_build", "package status mismatch")
require(package["repository_head"] == expected_head, "package repository head mismatch")
require(int(package["pilot_jobs"]) == 4, "package pilot job count mismatch")
require(len(package["payloads"]) == 4, "payload count mismatch")
require(package["initial_200_production_authorized"] is False, "package production auth mismatch")
require(package["validation_payloads_opened"] == 0, "package validation access")
require(package["test_payloads_opened"] == 0, "package test access")

require(len(matrix) == 4, "matrix row count mismatch")
require(set(matrix["job_index"].astype(int)) == {0,1,2,3}, "matrix job indices mismatch")
require(set(matrix["replica"].astype(int)) == {2,3}, "matrix replica coverage mismatch")
require(set(matrix["category_id"].astype(str)) == {"exact3tag","ge4tag"}, "matrix category mismatch")
require("3768139.0 - 3768139.3" in submit_text, "raw submit output range missing")

print("SUBMISSION_RECEIPT_AUDIT=PASS")
print("PACKAGE_BUILD_AUDIT=PASS")
print("CLUSTER_ID=3768139")
print("EXPECTED_PROCS=0-3")
print("PILOT_JOB_COUNT=4")
print("PILOT_RESULTS_MAY_ENTER_STABILITY_AGGREGATION=FALSE")
print("INITIAL_200_PRODUCTION_AUTHORIZED=FALSE")
print("VALIDATION_PAYLOADS_OPENED=0")
print("TEST_PAYLOADS_OPENED=0")
PY
[[ "$?" -eq 0 ]] || die "submission/package contract audit failed"

echo "FROZEN_SUBMISSION_PREFLIGHT=PASS"

echo
echo "===== 2. NONBLOCKING CONDOR STATUS SNAPSHOT ====="

: > "$QUEUE_SNAPSHOT"
: > "$HISTORY_SNAPSHOT"

condor_q -name "$SCHEDD" "$CLUSTER_ID" \
    -af ClusterId ProcId JobStatus 2>/dev/null \
    | sort -n -k2,2 > "$QUEUE_SNAPSHOT" || true

condor_history -name "$SCHEDD" "$CLUSTER_ID" \
    -af ClusterId ProcId JobStatus ExitCode ExitBySignal RemoteWallClock \
    2>/dev/null \
    | sort -n -k2,2 > "$HISTORY_SNAPSHOT" || true

QUEUE_JOB_COUNT="$(awk '$1==3768139 {n++} END{print n+0}' "$QUEUE_SNAPSHOT")"
HISTORY_JOB_COUNT="$(awk '$1==3768139 {n++} END{print n+0}' "$HISTORY_SNAPSHOT")"
UNIQUE_HISTORY_PROCS="$(awk '$1==3768139 {print $2}' "$HISTORY_SNAPSHOT" | sort -n -u | wc -l | tr -d ' ')"

echo "QUEUE_JOB_COUNT=$QUEUE_JOB_COUNT"
echo "HISTORY_JOB_COUNT=$HISTORY_JOB_COUNT"
echo "UNIQUE_HISTORY_PROCS=$UNIQUE_HISTORY_PROCS"

if [[ -s "$QUEUE_SNAPSHOT" ]]; then
    echo "QUEUE_SNAPSHOT_BEGIN"
    cat "$QUEUE_SNAPSHOT"
    echo "QUEUE_SNAPSHOT_END"
fi

if [[ -s "$HISTORY_SNAPSHOT" ]]; then
    echo "HISTORY_SNAPSHOT_BEGIN"
    cat "$HISTORY_SNAPSHOT"
    echo "HISTORY_SNAPSHOT_END"
fi

if [[ "$QUEUE_JOB_COUNT" -gt 0 || "$UNIQUE_HISTORY_PROCS" -lt "$EXPECTED_JOB_COUNT" ]]; then
    echo
    echo "============================================================"
    echo "PILOT STATUS"
    echo "============================================================"
    echo "PASS=IN_PROGRESS"
    echo "PILOT_COMPLETION_READY=FALSE"
    echo "DO_NOT_RESUBMIT_CLUSTER_3768139=TRUE"
    echo "INITIAL_200_PRODUCTION_AUTHORIZED=FALSE"
    echo "VALIDATION_PAYLOADS_OPENED=0"
    echo "TEST_PAYLOADS_OPENED=0"
    echo "NEXT=RERUN_THIS_READ_ONLY_AUDIT_FOR_CLUSTER_3768139"
    exit 0
fi

echo "PILOT_COMPLETION_READY=TRUE"

echo
echo "===== 3. CONDOR HISTORY COMPLETION AUDIT V2 ====="
echo "HISTORY_PARSER_ACCEPTS_EXITBYSIGNAL_BOOLEAN=TRUE"
echo "REMOTE_WALLCLOCK_UNDEFINED_IS_ALLOWED=TRUE"

HISTORY_SNAPSHOT_VALUE="$HISTORY_SNAPSHOT" "$PYTHON" - <<'PY'
from pathlib import Path
import math
import os

path = Path(os.environ["HISTORY_SNAPSHOT_VALUE"])

def parse_bool_token(token: str, field: str) -> bool:
    value = token.strip().lower()
    if value in {"false", "0"}:
        return False
    if value in {"true", "1"}:
        return True
    raise RuntimeError(f"unexpected {field} token: {token!r}")

def parse_optional_float(token: str):
    value = token.strip().lower()
    if value in {"undefined", "unknown", "none", "nan", ""}:
        return None
    parsed = float(token)
    if not math.isfinite(parsed):
        return None
    return parsed

rows = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    parts = line.split()
    if len(parts) < 6:
        raise RuntimeError(f"unexpected history row: {line}")

    cluster = int(parts[0])
    proc = int(parts[1])
    status = int(parts[2])
    exit_code = int(parts[3])
    exit_by_signal = parse_bool_token(parts[4], "ExitBySignal")
    remote_wall = parse_optional_float(parts[5])

    if cluster == 3768139:
        rows.append(
            {
                "proc": proc,
                "status": status,
                "exit_code": exit_code,
                "exit_by_signal": exit_by_signal,
                "remote_wall": remote_wall,
            }
        )

if len(rows) != 4:
    raise RuntimeError(f"expected 4 history rows, found {len(rows)}")

if sorted(x["proc"] for x in rows) != [0, 1, 2, 3]:
    raise RuntimeError("history proc coverage mismatch")

for row in rows:
    proc = row["proc"]
    if row["status"] != 4:
        raise RuntimeError(
            f"proc {proc} JobStatus={row['status']}, expected 4"
        )
    if row["exit_code"] != 0:
        raise RuntimeError(
            f"proc {proc} ExitCode={row['exit_code']}"
        )
    if row["exit_by_signal"]:
        raise RuntimeError(
            f"proc {proc} ExitBySignal=true"
        )

walls = [
    row["remote_wall"]
    for row in rows
    if row["remote_wall"] is not None
]

print("CONDOR_HISTORY_AUDIT=PASS")
print("CONDOR_CLEAN_COMPLETIONS=4")
print("EXIT_BY_SIGNAL_FALSE_COMPLETIONS=4")
print(f"REMOTE_WALLCLOCK_DEFINED_COUNT={len(walls)}")
print(f"REMOTE_WALLCLOCK_UNDEFINED_COUNT={4 - len(walls)}")

if walls:
    print(f"REMOTE_WALLCLOCK_MIN_SECONDS={min(walls):.3f}")
    print(f"REMOTE_WALLCLOCK_MAX_SECONDS={max(walls):.3f}")
else:
    print("REMOTE_WALLCLOCK_DIAGNOSTIC=UNAVAILABLE_NOT_AN_ACCEPTANCE_GATE")
PY
[[ "$?" -eq 0 ]] || die "Condor history completion audit failed"

echo
echo "===== 4. RETURNED ARTIFACT + 108-STRUCTURE AUDIT ====="

RETURN_ROOT_VALUE="$RETURN_ROOT" \
MATRIX_VALUE="$MATRIX" \
PACKAGE_AUDIT_VALUE="$PACKAGE_AUDIT" \
AMENDMENT_VALUE="$AMENDMENT" \
AUDIT_JSON_VALUE="$AUDIT_JSON" \
AUDIT_TSV_VALUE="$AUDIT_TSV" \
EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" \
EXPECTED_WORKER_SHA_VALUE="$EXPECTED_WORKER_SHA" \
EXPECTED_OPTIMIZER_SHA_VALUE="$EXPECTED_OPTIMIZER_SHA" \
EXPECTED_RUNTIME_SHA_VALUE="$EXPECTED_RUNTIME_SHA" \
"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import pandas as pd

return_root = Path(os.environ["RETURN_ROOT_VALUE"])
matrix = pd.read_csv(Path(os.environ["MATRIX_VALUE"]), sep="\t")
package = json.loads(Path(os.environ["PACKAGE_AUDIT_VALUE"]).read_text())
amendment = json.loads(Path(os.environ["AMENDMENT_VALUE"]).read_text())

audit_json = Path(os.environ["AUDIT_JSON_VALUE"])
audit_tsv = Path(os.environ["AUDIT_TSV_VALUE"])

expected_head = os.environ["EXPECTED_HEAD_VALUE"]
expected_worker_sha = os.environ["EXPECTED_WORKER_SHA_VALUE"]
expected_optimizer_sha = os.environ["EXPECTED_OPTIMIZER_SHA_VALUE"]
expected_runtime_sha = os.environ["EXPECTED_RUNTIME_SHA_VALUE"]

def require(c, m):
    if not c:
        raise RuntimeError(m)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

configs = amendment["category_program"]["category_family_configurations"]
expected_structures = {}
for category in ("exact3tag", "ge4tag"):
    vals = sorted(
        str(x["structure_id"])
        for x in configs
        if str(x["category_id"]) == category
    )
    require(len(vals) == 27, f"{category}: expected 27 structures")
    require(len(set(vals)) == 27, f"{category}: duplicate structure IDs")
    expected_structures[category] = vals

package_by_job = {int(x["job_index"]): x for x in package["payloads"]}
require(set(package_by_job) == {0,1,2,3}, "package job mapping mismatch")

job_records = []
structure_records = []

with tempfile.TemporaryDirectory(prefix="hh4b_stability_transfer_audit_") as tmp:
    tmp = Path(tmp)

    for row in matrix.itertuples(index=False):
        job = int(row.job_index)
        replica = int(row.replica)
        outer = int(row.outer_fold)
        category = str(row.category_id)

        rdir = return_root / f"job_{job:02d}"
        bundle = rdir / "result_bundle.tar.gz"
        receipt_path = rdir / "job_receipt.json"
        runner_log = rdir / "runner.log"

        require(rdir.is_dir(), f"missing return dir {rdir}")
        require(bundle.is_file(), f"missing result bundle job {job}")
        require(receipt_path.is_file(), f"missing receipt job {job}")
        require(runner_log.is_file(), f"missing runner log job {job}")

        receipt = json.loads(receipt_path.read_text())
        pkg = package_by_job[job]

        require(
            receipt["status"] == "category_fold_transfer_pilot_job_complete",
            f"job {job}: receipt status {receipt.get('status')}",
        )
        require(int(receipt["job_index"]) == job, f"job {job}: receipt job index")
        require(int(receipt["bootstrap_replica"]) == replica, f"job {job}: replica")
        require(int(receipt["outer_fold"]) == outer, f"job {job}: outer fold")
        require(receipt["category_id"] == category, f"job {job}: category")
        require(receipt["expected_repository_head"] == expected_head, f"job {job}: head")
        require(str(receipt["runner_rc"]) == "0", f"job {job}: runner rc")
        require(int(receipt["structures_expected"]) == 27, f"job {job}: expected structures")
        require(int(receipt["structures_attempted"]) == 27, f"job {job}: attempted structures")
        require(int(receipt["structures_completed_shell"]) == 27, f"job {job}: completed structures")
        require(int(receipt["valid_structure_results"]) == 27, f"job {job}: valid results")
        require(
            receipt["expected_payload_archive_sha256"] == pkg["payload_archive_sha256"],
            f"job {job}: payload SHA in receipt",
        )
        require(
            receipt["expected_runtime_archive_sha256"] == expected_runtime_sha,
            f"job {job}: runtime SHA in receipt",
        )
        require(receipt["result_bundle_sha256"] == sha(bundle), f"job {job}: bundle SHA")
        require(receipt["runner_log_sha256"] == sha(runner_log), f"job {job}: runner log SHA")
        require(receipt["pilot_results_may_enter_stability_aggregation"] is False, f"job {job}: pilot leakage")
        require(receipt["initial_200_production_authorized"] is False, f"job {job}: production auth")
        require(int(receipt["validation_payloads_opened"]) == 0, f"job {job}: validation")
        require(int(receipt["test_payloads_opened"]) == 0, f"job {job}: test")

        # Verify runner log completion markers.
        log_text = runner_log.read_text(errors="replace")
        require("STRUCTURES_COMPLETED=27" in log_text, f"job {job}: completion marker")
        require("RUNNER_RC=0" in log_text, f"job {job}: runner marker")

        out = tmp / f"job_{job:02d}"
        out.mkdir()
        subprocess.run(
            ["tar", "-xzf", str(bundle), "-C", str(out)],
            check=True,
        )

        results_root = out / "results"
        require(results_root.is_dir(), f"job {job}: extracted results root missing")

        result_paths = sorted(results_root.glob("*/structure_result.json"))
        require(len(result_paths) == 27, f"job {job}: result count {len(result_paths)}")

        observed_structures = []
        support_pass = 0
        feasible = 0
        all_inner_support_pass_count = 0

        # Locate execution provenance from the original frozen build payload.
        payload_dir = (
            Path(pkg["payload_archive"]).parent.parent
            / "build"
            / "payloads"
            / f"job_{job:02d}"
            / "payload"
        )
        # The path above is not guaranteed from archive parent arithmetic.
        # Fall back to known pilot-root build layout.
        pilot_root = return_root.parent
        provenance_path = (
            pilot_root / "build" / "payloads" / f"job_{job:02d}"
            / "payload" / "execution_provenance.json"
        )
        require(provenance_path.is_file(), f"job {job}: source execution provenance missing")
        provenance_sha = sha(provenance_path)
        provenance = json.loads(provenance_path.read_text())
        require(provenance["repository_head"] == expected_head, f"job {job}: provenance head")
        require(int(provenance["bootstrap_replica"]) == replica, f"job {job}: provenance replica")
        require(provenance["transferred_category_id"] == category, f"job {job}: provenance category")
        require(int(provenance["structure_count"]) == 27, f"job {job}: provenance structure count")
        require(provenance["pilot_results_may_enter_stability_aggregation"] is False, f"job {job}: provenance pilot leakage")
        require(provenance["initial_200_production_authorized"] is False, f"job {job}: provenance production auth")
        require(int(provenance["validation_payloads_opened"]) == 0, f"job {job}: provenance validation")
        require(int(provenance["test_payloads_opened"]) == 0, f"job {job}: provenance test")

        for rp in result_paths:
            result = json.loads(rp.read_text())
            sdir = rp.parent
            inner = sdir / "inner_crossfit.tsv"
            sums = sdir / "SHA256SUMS"

            require(inner.is_file(), f"job {job}: inner_crossfit missing {sdir.name}")
            require(sums.is_file(), f"job {job}: SHA256SUMS missing {sdir.name}")

            # Internal per-structure artifact integrity.
            check = subprocess.run(
                ["sha256sum", "-c", "SHA256SUMS"],
                cwd=sdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            require(check.returncode == 0, f"job {job}: SHA256SUMS failure {sdir.name}: {check.stdout}")

            require(result["status"] == "pass_structure_job_complete", f"job {job}: structure status")
            require(result["repository_head"] == expected_head, f"job {job}: structure head")
            require(result["execution_worker_sha256"] == expected_worker_sha, f"job {job}: worker SHA")
            require(result["execution_optimizer_sha256"] == expected_optimizer_sha, f"job {job}: optimizer SHA")
            require(result["execution_provenance_mode"] == "transferred_manifest", f"job {job}: provenance mode")
            require(result["execution_provenance_sha256"] == provenance_sha, f"job {job}: provenance SHA")
            require(int(result["outer_fold"]) == outer, f"job {job}: structure outer")
            require(result["category_id"] == category, f"job {job}: structure category")
            require(result["outer_fold_used_for_selection"] is False, f"job {job}: outer used for selection")
            require(int(result["validation_payloads_opened"]) == 0, f"job {job}: structure validation")
            require(int(result["test_payloads_opened"]) == 0, f"job {job}: structure test")

            sid = str(result["structure_id"])
            observed_structures.append(sid)

            inner_df = pd.read_csv(inner, sep="\t")
            require(len(inner_df) == 4, f"job {job}: inner row count {sid}")
            require(inner_df["inner_heldout_fold"].nunique() == 4, f"job {job}: inner fold coverage {sid}")

            canonical = str(result.get("canonical_payload_sha256", ""))
            require(len(canonical) == 64 and all(c in "0123456789abcdef" for c in canonical), f"job {job}: canonical hash {sid}")

            support_pass += int(bool(result.get("pooled_support_pass")))
            feasible += int(bool(result.get("pooled_inner_oof_feasible")))
            all_inner_support_pass_count += int(bool(result.get("all_inner_support_pass")))

            structure_records.append({
                "job_index": job,
                "bootstrap_replica": replica,
                "outer_fold": outer,
                "category_id": category,
                "structure_id": sid,
                "pooled_support_pass": bool(result.get("pooled_support_pass")),
                "all_inner_support_pass": bool(result.get("all_inner_support_pass")),
                "pooled_inner_oof_feasible": bool(result.get("pooled_inner_oof_feasible")),
                "canonical_payload_sha256": canonical,
                "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
                "validation_payloads_opened": int(result["validation_payloads_opened"]),
                "test_payloads_opened": int(result["test_payloads_opened"]),
            })

        require(sorted(observed_structures) == expected_structures[category], f"job {job}: 27-structure set mismatch")
        require(len(set(observed_structures)) == 27, f"job {job}: duplicate structure IDs")

        job_records.append({
            "job_index": job,
            "bootstrap_replica": replica,
            "outer_fold": outer,
            "category_id": category,
            "structures_returned": 27,
            "pooled_support_pass_count": support_pass,
            "all_inner_support_pass_count": all_inner_support_pass_count,
            "pooled_feasible_count": feasible,
            "runner_elapsed_seconds": int(receipt["elapsed_seconds"]),
            "result_bundle_sha256": sha(bundle),
            "runner_log_sha256": sha(runner_log),
            "execution_provenance_sha256": provenance_sha,
        })

require(len(job_records) == 4, "job record count mismatch")
require(len(structure_records) == 108, "structure record count mismatch")

jobs_df = pd.DataFrame(job_records).sort_values("job_index")
structures_df = pd.DataFrame(structure_records).sort_values(
    ["job_index", "structure_id"]
)
structures_df.to_csv(audit_tsv, sep="\t", index=False)

summary = {
    "schema_version": 1,
    "status": "pass_bounded_category_fold_transfer_pilot_complete",
    "cluster_id": 3768139,
    "repository_head": expected_head,
    "pilot_jobs": 4,
    "structure_results_expected": 108,
    "structure_results_returned": 108,
    "execution_integrity_pass_jobs": 4,
    "runner_rc_zero_jobs": 4,
    "categories": sorted(set(jobs_df["category_id"])),
    "bootstrap_replicas": sorted(int(x) for x in set(jobs_df["bootstrap_replica"])),
    "outer_folds_exercised": sorted(int(x) for x in set(jobs_df["outer_fold"])),
    "pooled_support_pass_structure_count": int(structures_df["pooled_support_pass"].sum()),
    "pooled_support_fail_structure_count": int((~structures_df["pooled_support_pass"]).sum()),
    "all_inner_support_pass_structure_count": int(structures_df["all_inner_support_pass"].sum()),
    "all_inner_support_fail_structure_count": int((~structures_df["all_inner_support_pass"]).sum()),
    "pooled_feasible_structure_count": int(structures_df["pooled_inner_oof_feasible"].sum()),
    "pooled_infeasible_structure_count": int((~structures_df["pooled_inner_oof_feasible"]).sum()),
    "scientific_support_or_feasibility_used_as_infrastructure_gate": False,
    "pilot_results_may_enter_stability_aggregation": False,
    "initial_200_production_authorized": False,
    "condor_production_submission_performed": False,
    "outer_fold_used_for_selection_any": False,
    "validation_payloads_opened": 0,
    "test_payloads_opened": 0,
    "jobs": job_records,
    "next": "freeze_transfer_pilot_and_authorize_initial_200_selection_stability_replicas",
}
audit_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

print("RETURNED_ARTIFACT_AUDIT=PASS")
print("RETURNED_PILOT_JOBS=4")
print("RETURNED_STRUCTURE_RESULTS=108")
print("EXPECTED_STRUCTURE_RESULTS=108")
print("ALL_STRUCTURE_INTERNAL_CHECKSUMS=PASS")
print("ALL_EXECUTION_PROVENANCE=PASS")
print("ALL_OUTER_FOLD_USED_FOR_SELECTION=FALSE")
print(
    "POOLED_SUPPORT_PASS_STRUCTURE_COUNT="
    + str(summary["pooled_support_pass_structure_count"])
)
print(
    "POOLED_SUPPORT_FAIL_STRUCTURE_COUNT="
    + str(summary["pooled_support_fail_structure_count"])
)
print(
    "POOLED_FEASIBLE_STRUCTURE_COUNT="
    + str(summary["pooled_feasible_structure_count"])
)
print(
    "POOLED_INFEASIBLE_STRUCTURE_COUNT="
    + str(summary["pooled_infeasible_structure_count"])
)
print("SCIENTIFIC_SUPPORT_OR_FEASIBILITY_USED_AS_INFRASTRUCTURE_GATE=FALSE")
print("PILOT_RESULTS_MAY_ENTER_STABILITY_AGGREGATION=FALSE")
print("INITIAL_200_PRODUCTION_AUTHORIZED=FALSE")
print("VALIDATION_PAYLOADS_OPENED=0")
print("TEST_PAYLOADS_OPENED=0")
PY
[[ "$?" -eq 0 ]] || die "returned artifact/structure audit failed"

AUDIT_JSON_SHA="$(sha256sum "$AUDIT_JSON" | awk '{print $1}')"
AUDIT_TSV_SHA="$(sha256sum "$AUDIT_TSV" | awk '{print $1}')"
QUEUE_SHA="$(sha256sum "$QUEUE_SNAPSHOT" | awk '{print $1}')"
HISTORY_SHA="$(sha256sum "$HISTORY_SNAPSHOT" | awk '{print $1}')"

echo
echo "============================================================"
echo "FINAL"
echo "============================================================"
echo "PASS=TRUE"
echo "CLUSTER_ID=3768139"
echo "CONDOR_CLEAN_COMPLETIONS=4"
echo "TRANSFER_PILOT_EXECUTION_INTEGRITY=PASS"
echo "RETURNED_PILOT_JOBS=4"
echo "RETURNED_STRUCTURE_RESULTS=108"
echo "PILOT_RESULTS_MAY_ENTER_STABILITY_AGGREGATION=FALSE"
echo "INITIAL_200_PRODUCTION_AUTHORIZED=FALSE"
echo "CONDOR_PRODUCTION_SUBMISSION_PERFORMED=FALSE"
echo "VALIDATION_PAYLOADS_OPENED=0"
echo "TEST_PAYLOADS_OPENED=0"
echo "AUDIT_JSON=$AUDIT_JSON"
echo "AUDIT_JSON_SHA256=$AUDIT_JSON_SHA"
echo "AUDIT_TSV=$AUDIT_TSV"
echo "AUDIT_TSV_SHA256=$AUDIT_TSV_SHA"
echo "QUEUE_SNAPSHOT_SHA256=$QUEUE_SHA"
echo "HISTORY_SNAPSHOT_SHA256=$HISTORY_SHA"
echo "DO_NOT_RESUBMIT_CLUSTER_3768139=TRUE"
echo "NEXT=FREEZE_TRANSFER_PILOT_AND_AUTHORIZE_INITIAL_200_SELECTION_STABILITY_REPLICAS"
} 2>&1 | tee "$AUDIT_LOG"

echo "AUDIT_LOG=$AUDIT_LOG"
echo "AUDIT_LOG_SHA256=$(sha256sum "$AUDIT_LOG" | awk '{print $1}')"
