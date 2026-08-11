#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$#" -eq 4 ]]
ROW_INDEX="$1"
SOURCE_UID="$2"
CLUSTER_ID="$3"
PROC_ID="$4"

EXPECTED_HEAD=3e54a57a65aa572786a4816588c604898adfe40f
AUTHORIZATION_HEAD=d69be420d01dd2b731aac17ed4ff33cc0691722e
AUTHORIZATION_SHA=c0a828f2a1ed24ee89435da19c780bcf4808a98b25155c070bfef56f89e115c0
COMMON_SHA=7cf5d1f560d618f581bf84e0a96b88bd74fb46da4b05a7ca2b5df5559f0907d4
RUNTIME_SHA=4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112
RUNTIME_REMOTE=/store/user/iturkmen/hh4b_cut_baseline/validation_once_20260811_v1/runtime/hh4b_python_runtime_v8_r1.tar.gz
REMOTE_OUTPUT_ROOT=/store/user/iturkmen/hh4b_cut_baseline/validation_once_20260811_v1/results
EXPECTED_RUNTIME_JSON='{"awkward": "2.8.12", "numpy": "1.23.5", "pandas": "2.3.3", "pyarrow": "21.0.0", "uproot": "5.6.9"}'
EOS_HOST="root://cmseos.fnal.gov"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

MARKER_NAME="source_$(printf '%04d' "$ROW_INDEX")_VALIDATION_OPEN_DO_NOT_RERUN.json"
SUMMARY_NAME="source_$(printf '%04d' "$ROW_INDEX")_summary.json"
DISTRIBUTION_NAME="source_$(printf '%04d' "$ROW_INDEX")_distributions.tsv"
RECEIPT_NAME="source_$(printf '%04d' "$ROW_INDEX")_job_receipt.json"
REMOTE_MARKER="$REMOTE_OUTPUT_ROOT/markers/$MARKER_NAME"
REMOTE_SUMMARY="$REMOTE_OUTPUT_ROOT/summaries/$SUMMARY_NAME"
REMOTE_DISTRIBUTION="$REMOTE_OUTPUT_ROOT/distributions/$DISTRIBUTION_NAME"
REMOTE_RECEIPT="$REMOTE_OUTPUT_ROOT/receipts/$RECEIPT_NAME"
DURABLE_MARKER_URI="$EOS_HOST/$REMOTE_MARKER"
MARKER_CREATED=FALSE
SOURCE_WORKER_STARTED=FALSE

fail() {
    echo "VALIDATION_RUNNER_STATUS=FAIL;row_index=$ROW_INDEX;cluster_id=$CLUSTER_ID;proc_id=$PROC_ID;marker_created=$MARKER_CREATED;worker_started=$SOURCE_WORKER_STARTED;message=$*" >&2
    exit 1
}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "VALIDATION_RUNNER_EXIT_RC=$rc;row_index=$ROW_INDEX;marker_created=$MARKER_CREATED;worker_started=$SOURCE_WORKER_STARTED" >&2; fi' EXIT

for command in awk grep python3 sed sha256sum tar xrdcp xrdfs; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ -n "${X509_USER_PROXY:-}" && -f "$X509_USER_PROXY" ]] || fail "X509 proxy unavailable"
[[ "$ROW_INDEX" =~ ^[0-9]+$ ]] || fail "invalid row index"
[[ "$SOURCE_UID" =~ ^[A-Za-z0-9_./:+-]+$ ]] || fail "invalid source UID"
[[ "$(sha256sum validation_common_bundle.tar.gz | awk '{print $1}')" == "$COMMON_SHA" ]] || fail "common bundle SHA mismatch"

mkdir -p "$SCRATCH/common" "$SCRATCH/runtime" "$SCRATCH/downloads" "$SCRATCH/job_output" "$SCRATCH/job_work"
tar -xzf validation_common_bundle.tar.gz -C "$SCRATCH/common"
(
    cd "$SCRATCH/common"
    sha256sum -c COMMON_SHA256SUMS
) || fail "common bundle internal checksum failure"
[[ "$(sha256sum "$SCRATCH/common/validation_authorization.json" | awk '{print $1}')" == "$AUTHORIZATION_SHA" ]] || fail "authorization SHA mismatch"

RUNTIME_ARCHIVE="$SCRATCH/downloads/runtime_bundle.tar.gz"
runtime_rc=1
for attempt in 1 2 3 4 5; do
    if xrdcp -f --nopbar "$EOS_HOST/$RUNTIME_REMOTE" "$RUNTIME_ARCHIVE"; then
        runtime_rc=0
        break
    fi
    runtime_rc=$?
    echo "VALIDATION_RUNTIME_DOWNLOAD_RETRY;attempt=$attempt;rc=$runtime_rc" >&2
    sleep $((attempt * 10))
done
[[ "$runtime_rc" -eq 0 ]] || fail "runtime download failed"
[[ "$(sha256sum "$RUNTIME_ARCHIVE" | awk '{print $1}')" == "$RUNTIME_SHA" ]] || fail "runtime SHA mismatch"
tar -xzf "$RUNTIME_ARCHIVE" -C "$SCRATCH/runtime"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRATCH/runtime/site-packages"
EXPECTED_RUNTIME_JSON_VALUE="$EXPECTED_RUNTIME_JSON" python3 - <<'PY_RUNTIME' || fail "runtime probe failed"
import json
import os
import awkward
import numpy
import pandas
import pyarrow
import uproot
observed = {
    "awkward": awkward.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "uproot": uproot.__version__,
}
expected = json.loads(os.environ["EXPECTED_RUNTIME_JSON_VALUE"])
if observed != expected:
    raise RuntimeError(f"runtime drift: {observed}")
print("VALIDATION_RUNTIME=PASS")
PY_RUNTIME

for directory in markers summaries distributions receipts; do
    xrdfs "$EOS_HOST" mkdir -p "$REMOTE_OUTPUT_ROOT/$directory" || fail "cannot create remote $directory directory"
done
for specification in     "markers:$REMOTE_MARKER"     "summaries:$REMOTE_SUMMARY"     "distributions:$REMOTE_DISTRIBUTION"     "receipts:$REMOTE_RECEIPT"
do
    directory="${specification%%:*}"
    target="${specification#*:}"
    listing="$(xrdfs "$EOS_HOST" ls "$REMOTE_OUTPUT_ROOT/$directory")" || fail "cannot inspect remote $directory directory"
    if grep -F -x -- "$target" <<<"$listing" >/dev/null; then
        fail "prior durable source artifact exists: $target"
    fi
done

LOCAL_MARKER="$SCRATCH/$MARKER_NAME"
ROW_INDEX_VALUE="$ROW_INDEX" SOURCE_UID_VALUE="$SOURCE_UID" CLUSTER_ID_VALUE="$CLUSTER_ID" PROC_ID_VALUE="$PROC_ID" EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" AUTHORIZATION_HEAD_VALUE="$AUTHORIZATION_HEAD" AUTHORIZATION_SHA_VALUE="$AUTHORIZATION_SHA" DURABLE_MARKER_URI_VALUE="$DURABLE_MARKER_URI" LOCAL_MARKER_VALUE="$LOCAL_MARKER" python3 - <<'PY_MARKER'
from pathlib import Path
import json
import os
import time
payload = {
    "schema_version": 1,
    "status": "validation_source_open_attempt_durable_do_not_rerun",
    "repository_head": os.environ["EXPECTED_HEAD_VALUE"],
    "authorization_repository_head": os.environ["AUTHORIZATION_HEAD_VALUE"],
    "authorization_sha256": os.environ["AUTHORIZATION_SHA_VALUE"],
    "production_row_index": int(os.environ["ROW_INDEX_VALUE"]),
    "source_uid": os.environ["SOURCE_UID_VALUE"],
    "condor_cluster_id": int(os.environ["CLUSTER_ID_VALUE"]),
    "condor_proc_id": int(os.environ["PROC_ID_VALUE"]),
    "durable_marker_uri": os.environ["DURABLE_MARKER_URI_VALUE"],
    "source_payload_access_may_begin": True,
    "rerun_forbidden_even_if_downstream_bookkeeping_fails": True,
    "test_payloads_opened": 0,
    "created_unix_time": time.time(),
}
Path(os.environ["LOCAL_MARKER_VALUE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY_MARKER

# This is the one irreversible validation-access gate.  No -f is used: EOS
# must atomically reject an already-existing marker.
if ! xrdcp --nopbar --cksum adler32 "$LOCAL_MARKER" "$DURABLE_MARKER_URI"; then
    fail "exclusive durable marker publication failed or was ambiguous; source access forbidden"
fi
MARKER_CREATED=TRUE
LOCAL_ADLER="$(python3 - "$LOCAL_MARKER" <<'PY_ADLER'
from pathlib import Path
import sys
import zlib
print(f"{zlib.adler32(Path(sys.argv[1]).read_bytes()) & 0xffffffff:08x}")
PY_ADLER
)"
REMOTE_ADLER="$(xrdfs "$EOS_HOST" query checksum "$REMOTE_MARKER" | awk '{print tolower($NF)}' | sed 's/^0x//')" || fail "durable marker checksum query failed"
[[ "$REMOTE_ADLER" == "$LOCAL_ADLER" ]] || fail "durable marker checksum mismatch"

SOURCE_WORKER_STARTED=TRUE
python3 "$SCRATCH/common/worker.py"     --row-index "$ROW_INDEX"     --source-access-manifest "$SCRATCH/common/source_access_manifest.tsv"     --coefficient-registry "$SCRATCH/common/physical_coefficient_registry.tsv"     --extractor "$SCRATCH/common/extractor.py"     --reconstruction-module "$SCRATCH/common/reconstruction.py"     --authorization "$SCRATCH/common/validation_authorization.json"     --expected-head "$EXPECTED_HEAD"     --work-root "$SCRATCH/job_work"     --output-root "$SCRATCH/job_output"     --preexisting-attempt-marker "$LOCAL_MARKER"     --durable-attempt-marker-uri "$DURABLE_MARKER_URI"     || fail "validation source worker failed; marker forbids rerun"

LOCAL_SUMMARY="$SCRATCH/job_output/$SUMMARY_NAME"
LOCAL_DISTRIBUTION="$SCRATCH/job_output/$DISTRIBUTION_NAME"
LOCAL_MARKER_COPY="$SCRATCH/job_output/$MARKER_NAME"
[[ -f "$LOCAL_SUMMARY" && -f "$LOCAL_DISTRIBUTION" && -f "$LOCAL_MARKER_COPY" ]] || fail "source output closure failed"
[[ "$(sha256sum "$LOCAL_MARKER_COPY" | awk '{print $1}')" == "$(sha256sum "$LOCAL_MARKER" | awk '{print $1}')" ]] || fail "marker evidence changed"

publish_product() {
    local source="$1"
    local destination="$2"
    local label="$3"
    local temporary="$REMOTE_OUTPUT_ROOT/.upload_tmp/${label}.${CLUSTER_ID}.${PROC_ID}.$$"
    xrdfs "$EOS_HOST" mkdir -p "$REMOTE_OUTPUT_ROOT/.upload_tmp" || fail "cannot create upload temp directory"
    local upload_rc=1
    for attempt in 1 2 3 4 5; do
        if xrdcp -f --nopbar --cksum adler32 "$source" "$EOS_HOST/$temporary"; then
            upload_rc=0
            break
        fi
        upload_rc=$?
        echo "VALIDATION_OUTPUT_UPLOAD_RETRY;label=$label;attempt=$attempt;rc=$upload_rc" >&2
        sleep $((attempt * 10))
    done
    [[ "$upload_rc" -eq 0 ]] || fail "$label temporary upload failed"
    if ! xrdfs "$EOS_HOST" mv "$temporary" "$destination"; then
        fail "$label atomic publication failed; source marker forbids rerun"
    fi
}

publish_product "$LOCAL_SUMMARY" "$REMOTE_SUMMARY" "summary"
publish_product "$LOCAL_DISTRIBUTION" "$REMOTE_DISTRIBUTION" "distribution"

LOCAL_RECEIPT="$SCRATCH/$RECEIPT_NAME"
ROW_INDEX_VALUE="$ROW_INDEX" SOURCE_UID_VALUE="$SOURCE_UID" CLUSTER_ID_VALUE="$CLUSTER_ID" PROC_ID_VALUE="$PROC_ID" EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" AUTHORIZATION_SHA_VALUE="$AUTHORIZATION_SHA" MARKER_PATH_VALUE="$LOCAL_MARKER" SUMMARY_PATH_VALUE="$LOCAL_SUMMARY" DISTRIBUTION_PATH_VALUE="$LOCAL_DISTRIBUTION" DURABLE_MARKER_URI_VALUE="$DURABLE_MARKER_URI" LOCAL_RECEIPT_VALUE="$LOCAL_RECEIPT" python3 - <<'PY_RECEIPT'
from pathlib import Path
import hashlib
import json
import os
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "status": "pass_one_time_fixed_nominal_validation_source_job",
    "repository_head": os.environ["EXPECTED_HEAD_VALUE"],
    "authorization_sha256": os.environ["AUTHORIZATION_SHA_VALUE"],
    "production_row_index": int(os.environ["ROW_INDEX_VALUE"]),
    "source_uid": os.environ["SOURCE_UID_VALUE"],
    "condor_cluster_id": int(os.environ["CLUSTER_ID_VALUE"]),
    "condor_proc_id": int(os.environ["PROC_ID_VALUE"]),
    "durable_attempt_marker_uri": os.environ["DURABLE_MARKER_URI_VALUE"],
    "attempt_marker_sha256": sha(os.environ["MARKER_PATH_VALUE"]),
    "source_summary_sha256": sha(os.environ["SUMMARY_PATH_VALUE"]),
    "source_distributions_sha256": sha(os.environ["DISTRIBUTION_PATH_VALUE"]),
    "validation_payloads_opened": 1,
    "test_payloads_opened": 0,
    "rerun_authorized": False,
}
Path(os.environ["LOCAL_RECEIPT_VALUE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY_RECEIPT
publish_product "$LOCAL_RECEIPT" "$REMOTE_RECEIPT" "receipt"

echo "VALIDATION_RUNNER_STATUS=PASS;row_index=$ROW_INDEX;cluster_id=$CLUSTER_ID;proc_id=$PROC_ID;validation_payloads_opened=1;test_payloads_opened=0"
