#!/usr/bin/env python3
"""Prepare, but never submit, the exactly-once HH->4b validation campaign.

The generated runner creates and verifies an exclusive durable EOS marker before
the broad extractor can access a validation ROOT payload.  A restarted job sees
that marker and fails closed; it can never reopen the source automatically.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tarfile
from typing import Any

import pandas as pd


BRANCH = "delphes-hh4b-production"
EXPECTED_RUNTIME_VERSIONS = {
    "awkward": "2.8.12",
    "numpy": "1.23.5",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "uproot": "5.6.9",
}
COMMON_MEMBERS = {
    "worker.py": "validation_source_worker",
    "extractor.py": "broad_feature_extractor",
    "reconstruction.py": "candidate_reconstruction_module",
    "source_access_manifest.tsv": "source_access_manifest",
    "physical_coefficient_registry.tsv": "physical_coefficient_registry",
    "validation_authorization.json": None,
}


class CampaignPreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignPreparationError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def require_full_sha(value: str, label: str) -> None:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", value)), f"invalid {label}: {value}")


def committed_bytes(repo: Path, commit: str, path: Path) -> bytes:
    relative = path.resolve().relative_to(repo).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)


def verify_repository_and_authorization_checkpoint(
    repo: Path,
    authorization_checkpoint: Path,
    authorization: Path,
    authorization_commit: str,
) -> str:
    require(repo.resolve() == repo, "repository path is not canonical")
    require(git(repo, "branch", "--show-current") == BRANCH, "branch changed")
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    require(head == remote, "local and remote HEAD differ")
    require_full_sha(head, "execution repository head")
    require_full_sha(authorization_commit, "authorization checkpoint commit")
    require(
        git(repo, "rev-parse", authorization_commit) == authorization_commit,
        "authorization checkpoint commit is not exact",
    )
    require(
        git_is_ancestor(repo, authorization_commit, head),
        "authorization checkpoint commit is not an ancestor of execution HEAD",
    )
    require(
        authorization_checkpoint.is_dir() and not authorization_checkpoint.is_symlink(),
        "authorization checkpoint is invalid",
    )
    sums = authorization_checkpoint / "SHA256SUMS"
    for path in (authorization, sums):
        require(path.is_file() and not path.is_symlink(), f"missing checkpoint file: {path}")
        require(
            sha256_bytes(committed_bytes(repo, authorization_commit, path)) == sha256(path),
            f"authorization checkpoint differs from committed bytes: {path}",
        )
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=authorization_checkpoint,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(check.returncode == 0, f"authorization checkpoint checksum failure: {check.stdout}{check.stderr}")
    return head


def validate_remote_path(value: str, label: str) -> str:
    path = clean(value).rstrip("/")
    require(path.startswith("/store/user/iturkmen/"), f"unsafe {label}: {path}")
    require(".." not in Path(path).parts, f"parent traversal in {label}")
    require(bool(re.fullmatch(r"[A-Za-z0-9_./:+-]+", path)), f"unsafe characters in {label}")
    return path


def validate_source_token(value: str) -> str:
    token = clean(value)
    # Frozen source UIDs retain the physical bundle provenance as
    # ``bundle:/store/...``.  A slash is therefore part of the canonical UID,
    # while whitespace and shell metacharacters remain forbidden.
    require(bool(re.fullmatch(r"[A-Za-z0-9_./:+-]+", token)), f"unsafe source UID: {token}")
    return token


def validate_authorized_inputs(
    authorization: dict[str, Any],
    paths: dict[str, Path],
    execution_head: str,
    repo: Path,
    authorization_commit: str,
) -> pd.DataFrame:
    require(
        authorization.get("status") == "authorized_one_time_cut_baseline_validation",
        "validation is not explicitly authorized",
    )
    require(authorization.get("validation_access_authorized") is True, "validation access is not authorized")
    require(authorization.get("validation_payloads_opened_before_authorization") == 0, "validation was opened before authorization")
    require(authorization.get("validation_payloads_opened") == 0, "validation was already opened")
    require(authorization.get("test_payloads_opened") == 0, "test is not sealed")
    require(authorization.get("test_access_authorized") is False, "test access was authorized")
    require(authorization.get("authorized_validation_sources") == 116, "authorized source count changed")
    require(authorization.get("auxiliary_qcd_validation_sources_authorized") == 0, "auxiliary QCD was authorized")
    authorization_head = clean(authorization.get("repository_head"))
    require_full_sha(authorization_head, "authorization repository head")
    require(
        git_is_ancestor(repo, authorization_head, authorization_commit)
        and git_is_ancestor(repo, authorization_commit, execution_head),
        "authorization/commit/execution ancestry chain failed",
    )
    master_commit = clean(authorization.get("master_train_only_checkpoint_commit"))
    require_full_sha(master_commit, "master train-only checkpoint commit")
    require(git_is_ancestor(repo, master_commit, authorization_head), "master checkpoint is not upstream of authorization")

    expected_hashes = authorization.get("authorized_sha256", {})
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular authorized input: {path}")
        require(expected_hashes.get(key) == sha256(path), f"authorized SHA256 changed: {key}")

    access = pd.read_csv(paths["source_access_manifest"], sep="\t", keep_default_na=False)
    coefficients = pd.read_csv(paths["physical_coefficient_registry"], sep="\t", keep_default_na=False)
    require(len(access) == 121 and access["source_uid"].nunique() == 121, "validation source closure changed")
    physical = access["physical_evaluation_eligible"].map(truthy)
    auxiliary = access["auxiliary_qcd"].map(truthy)
    require(int(physical.sum()) == 116 and int(auxiliary.sum()) == 5, "validation source partition changed")
    require(physical.eq(~auxiliary).all(), "physical/auxiliary source partition changed")
    require(access["split"].astype(str).map(clean).eq("validation").all(), "non-validation source present")
    require(not access["validation_content_opened"].map(truthy).any(), "validation metadata reports opened content")
    require(not access["test_content_opened"].map(truthy).any(), "test metadata reports opened content")
    require(len(coefficients) == 121 and coefficients["source_uid"].is_unique, "coefficient closure changed")
    require(access["source_uid"].tolist() == coefficients["source_uid"].tolist(), "coefficient source order changed")
    require(
        int(pd.to_numeric(access.loc[physical, "generated_events"], errors="raise").sum())
        == int(authorization["authorized_generated_events"]),
        "authorized generated-event total changed",
    )
    for value in access.loc[physical, "source_uid"]:
        validate_source_token(value)
    return access.loc[physical].sort_values("production_row_index").copy()


def deterministic_tar_gz(output: Path, members: dict[str, bytes]) -> None:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(tar_buffer.getvalue())


def build_common_bundle(output: Path, paths: dict[str, Path], authorization: Path) -> dict[str, str]:
    members: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for archive_name, authorization_key in COMMON_MEMBERS.items():
        path = authorization if authorization_key is None else paths[authorization_key]
        payload = path.read_bytes()
        members[archive_name] = payload
        hashes[archive_name] = sha256_bytes(payload)
    members["COMMON_SHA256SUMS"] = "".join(
        f"{hashes[name]}  {name}\n" for name in sorted(hashes)
    ).encode("utf-8")
    deterministic_tar_gz(output, members)
    return hashes


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def render_runner(
    *,
    execution_head: str,
    authorization_head: str,
    authorization_sha: str,
    common_sha: str,
    runtime_sha: str,
    runtime_remote_path: str,
    remote_output_root: str,
) -> str:
    versions_json = json.dumps(EXPECTED_RUNTIME_VERSIONS, sort_keys=True)
    embedded = {
        "EXPECTED_HEAD": execution_head,
        "AUTHORIZATION_HEAD": authorization_head,
        "AUTHORIZATION_SHA": authorization_sha,
        "COMMON_SHA": common_sha,
        "RUNTIME_SHA": runtime_sha,
        "RUNTIME_REMOTE": runtime_remote_path,
        "REMOTE_OUTPUT_ROOT": remote_output_root,
        "EXPECTED_RUNTIME_JSON": versions_json,
    }
    assignments = "\n".join(f"{key}={shell_quote(value)}" for key, value in embedded.items())
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$#" -eq 4 ]]
ROW_INDEX="$1"
SOURCE_UID="$2"
CLUSTER_ID="$3"
PROC_ID="$4"

{assignments}
EOS_HOST="root://cmseos.fnal.gov"
SCRATCH="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
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

fail() {{
    echo "VALIDATION_RUNNER_STATUS=FAIL;row_index=$ROW_INDEX;cluster_id=$CLUSTER_ID;proc_id=$PROC_ID;marker_created=$MARKER_CREATED;worker_started=$SOURCE_WORKER_STARTED;message=$*" >&2
    exit 1
}}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "VALIDATION_RUNNER_EXIT_RC=$rc;row_index=$ROW_INDEX;marker_created=$MARKER_CREATED;worker_started=$SOURCE_WORKER_STARTED" >&2; fi' EXIT

for command in awk grep python3 sed sha256sum tar xrdcp xrdfs; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ -n "${{X509_USER_PROXY:-}}" && -f "$X509_USER_PROXY" ]] || fail "X509 proxy unavailable"
[[ "$ROW_INDEX" =~ ^[0-9]+$ ]] || fail "invalid row index"
[[ "$SOURCE_UID" =~ ^[A-Za-z0-9_./:+-]+$ ]] || fail "invalid source UID"
[[ "$(sha256sum validation_common_bundle.tar.gz | awk '{{print $1}}')" == "$COMMON_SHA" ]] || fail "common bundle SHA mismatch"

mkdir -p "$SCRATCH/common" "$SCRATCH/runtime" "$SCRATCH/downloads" "$SCRATCH/job_output" "$SCRATCH/job_work"
tar -xzf validation_common_bundle.tar.gz -C "$SCRATCH/common"
(
    cd "$SCRATCH/common"
    sha256sum -c COMMON_SHA256SUMS
) || fail "common bundle internal checksum failure"
[[ "$(sha256sum "$SCRATCH/common/validation_authorization.json" | awk '{{print $1}}')" == "$AUTHORIZATION_SHA" ]] || fail "authorization SHA mismatch"

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
[[ "$(sha256sum "$RUNTIME_ARCHIVE" | awk '{{print $1}}')" == "$RUNTIME_SHA" ]] || fail "runtime SHA mismatch"
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
observed = {{
    "awkward": awkward.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "uproot": uproot.__version__,
}}
expected = json.loads(os.environ["EXPECTED_RUNTIME_JSON_VALUE"])
if observed != expected:
    raise RuntimeError(f"runtime drift: {{observed}}")
print("VALIDATION_RUNTIME=PASS")
PY_RUNTIME

for directory in markers summaries distributions receipts; do
    xrdfs "$EOS_HOST" mkdir -p "$REMOTE_OUTPUT_ROOT/$directory" || fail "cannot create remote $directory directory"
done
for specification in \
    "markers:$REMOTE_MARKER" \
    "summaries:$REMOTE_SUMMARY" \
    "distributions:$REMOTE_DISTRIBUTION" \
    "receipts:$REMOTE_RECEIPT"
do
    directory="${{specification%%:*}}"
    target="${{specification#*:}}"
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
payload = {{
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
}}
Path(os.environ["LOCAL_MARKER_VALUE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\\n",
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
print(f"{{zlib.adler32(Path(sys.argv[1]).read_bytes()) & 0xffffffff:08x}}")
PY_ADLER
)"
REMOTE_ADLER="$(xrdfs "$EOS_HOST" query checksum "$REMOTE_MARKER" | awk '{{print tolower($NF)}}' | sed 's/^0x//')" || fail "durable marker checksum query failed"
[[ "$REMOTE_ADLER" == "$LOCAL_ADLER" ]] || fail "durable marker checksum mismatch"

SOURCE_WORKER_STARTED=TRUE
python3 "$SCRATCH/common/worker.py" \
    --row-index "$ROW_INDEX" \
    --source-access-manifest "$SCRATCH/common/source_access_manifest.tsv" \
    --coefficient-registry "$SCRATCH/common/physical_coefficient_registry.tsv" \
    --extractor "$SCRATCH/common/extractor.py" \
    --reconstruction-module "$SCRATCH/common/reconstruction.py" \
    --authorization "$SCRATCH/common/validation_authorization.json" \
    --expected-head "$EXPECTED_HEAD" \
    --work-root "$SCRATCH/job_work" \
    --output-root "$SCRATCH/job_output" \
    --preexisting-attempt-marker "$LOCAL_MARKER" \
    --durable-attempt-marker-uri "$DURABLE_MARKER_URI" \
    || fail "validation source worker failed; marker forbids rerun"

LOCAL_SUMMARY="$SCRATCH/job_output/$SUMMARY_NAME"
LOCAL_DISTRIBUTION="$SCRATCH/job_output/$DISTRIBUTION_NAME"
LOCAL_MARKER_COPY="$SCRATCH/job_output/$MARKER_NAME"
[[ -f "$LOCAL_SUMMARY" && -f "$LOCAL_DISTRIBUTION" && -f "$LOCAL_MARKER_COPY" ]] || fail "source output closure failed"
[[ "$(sha256sum "$LOCAL_MARKER_COPY" | awk '{{print $1}}')" == "$(sha256sum "$LOCAL_MARKER" | awk '{{print $1}}')" ]] || fail "marker evidence changed"

publish_product() {{
    local source="$1"
    local destination="$2"
    local label="$3"
    local temporary="$REMOTE_OUTPUT_ROOT/.upload_tmp/${{label}}.${{CLUSTER_ID}}.${{PROC_ID}}.$$"
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
}}

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
payload = {{
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
}}
Path(os.environ["LOCAL_RECEIPT_VALUE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\\n",
    encoding="utf-8",
)
PY_RECEIPT
publish_product "$LOCAL_RECEIPT" "$REMOTE_RECEIPT" "receipt"

echo "VALIDATION_RUNNER_STATUS=PASS;row_index=$ROW_INDEX;cluster_id=$CLUSTER_ID;proc_id=$PROC_ID;validation_payloads_opened=1;test_payloads_opened=0"
'''


def render_submit(
    *,
    package_root: Path,
    log_root: Path,
    x509_proxy: Path,
    apptainer_image: str,
) -> str:
    return f"""universe = vanilla
executable = {package_root / 'run_validation_source.sh'}
arguments = $(row_index) $(source_uid) $(ClusterId) $(ProcId)

output = {log_root}/validation.$(ClusterId).$(ProcId).row$(row_index).out
error = {log_root}/validation.$(ClusterId).$(ProcId).row$(row_index).err
log = {log_root}/validation.$(ClusterId).log

getenv = True
use_x509userproxy = True
x509userproxy = {x509_proxy}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {package_root / 'validation_common_bundle.tar.gz'}
transfer_output_files = ""

request_cpus = 1
request_memory = 8000MB
request_disk = 20GB

on_exit_remove = True
+JobBatchName = "hh4b_cut_baseline_one_time_validation_116"
+ApptainerImage = "{apptainer_image}"

queue row_index, source_uid from {package_root / 'validation_queue.items'}
"""


def write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authorization-checkpoint", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-checkpoint-commit", required=True)
    parser.add_argument("--source-access-manifest", type=Path, required=True)
    parser.add_argument("--physical-coefficient-registry", type=Path, required=True)
    parser.add_argument("--broad-feature-extractor", type=Path, required=True)
    parser.add_argument("--candidate-reconstruction-module", type=Path, required=True)
    parser.add_argument("--validation-source-worker", type=Path, required=True)
    parser.add_argument("--validation-campaign-auditor", type=Path, required=True)
    parser.add_argument("--validation-campaign-submitter", type=Path, required=True)
    parser.add_argument("--validation-return-auditor", type=Path, required=True)
    parser.add_argument("--validation-runtime-bundle", type=Path, required=True)
    parser.add_argument("--runtime-remote-path", required=True)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--local-log-root", type=Path, required=True)
    parser.add_argument("--x509-proxy", type=Path, required=True)
    parser.add_argument(
        "--apptainer-image",
        default="/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel9",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    authorization_checkpoint = args.authorization_checkpoint.resolve()
    authorization_path = args.authorization.resolve()
    execution_head = verify_repository_and_authorization_checkpoint(
        repo,
        authorization_checkpoint,
        authorization_path,
        args.authorization_checkpoint_commit,
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    paths = {
        "source_access_manifest": args.source_access_manifest.resolve(),
        "physical_coefficient_registry": args.physical_coefficient_registry.resolve(),
        "broad_feature_extractor": args.broad_feature_extractor.resolve(),
        "candidate_reconstruction_module": args.candidate_reconstruction_module.resolve(),
        "validation_source_worker": args.validation_source_worker.resolve(),
        "validation_campaign_preparer": Path(__file__).resolve(),
        "validation_campaign_auditor": args.validation_campaign_auditor.resolve(),
        "validation_campaign_submitter": args.validation_campaign_submitter.resolve(),
        "validation_return_auditor": args.validation_return_auditor.resolve(),
        "validation_runtime_bundle": args.validation_runtime_bundle.resolve(),
    }
    physical = validate_authorized_inputs(
        authorization,
        paths,
        execution_head,
        repo,
        args.authorization_checkpoint_commit,
    )
    runtime_remote = validate_remote_path(args.runtime_remote_path, "runtime remote path")
    remote_output = validate_remote_path(args.remote_output_root, "validation output root")
    require(runtime_remote.endswith(".tar.gz"), "runtime remote path is not a tar archive")
    require(args.local_log_root.is_absolute(), "local log root must be absolute")
    require(args.x509_proxy.is_absolute(), "X509 proxy path must be absolute")
    require(bool(re.fullmatch(r"[A-Za-z0-9_./:+-]+", args.apptainer_image)), "unsafe Apptainer image")

    output = args.output_dir.resolve()
    require(output.parent.is_dir(), "campaign output parent is missing")
    require(not output.exists(), f"campaign output already exists: {output}")
    build = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    require(not build.exists(), f"temporary campaign directory exists: {build}")
    build.mkdir()
    try:
        common_path = build / "validation_common_bundle.tar.gz"
        common_hashes = build_common_bundle(common_path, paths, authorization_path)
        common_sha = sha256(common_path)
        runtime_sha = sha256(paths["validation_runtime_bundle"])
        authorization_sha = sha256(authorization_path)

        queue_rows = []
        for row in physical.itertuples(index=False):
            queue_rows.append(
                f"{int(row.production_row_index)} {validate_source_token(row.source_uid)}\n"
            )
        require(len(queue_rows) == 116 and len(set(queue_rows)) == 116, "validation queue closure failed")
        write_text(build / "validation_queue.items", "".join(queue_rows))

        runner = render_runner(
            execution_head=execution_head,
            authorization_head=authorization["repository_head"],
            authorization_sha=authorization_sha,
            common_sha=common_sha,
            runtime_sha=runtime_sha,
            runtime_remote_path=runtime_remote,
            remote_output_root=remote_output,
        )
        runner_path = build / "run_validation_source.sh"
        write_text(runner_path, runner)
        runner_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        submit = render_submit(
            package_root=output,
            log_root=args.local_log_root.resolve(),
            x509_proxy=args.x509_proxy.resolve(),
            apptainer_image=args.apptainer_image,
        )
        write_text(build / "validation_116.submit", submit)

        summary = {
            "schema_version": 1,
            "status": "pass_prepared_exactly_once_cut_baseline_validation_campaign",
            "repository_head": execution_head,
            "authorization_repository_head": authorization["repository_head"],
            "authorization_checkpoint_commit": args.authorization_checkpoint_commit,
            "authorization_sha256": authorization_sha,
            "master_train_only_checkpoint_commit": authorization["master_train_only_checkpoint_commit"],
            "authorized_physical_validation_sources": 116,
            "authorized_generated_events": int(authorization["authorized_generated_events"]),
            "auxiliary_qcd_validation_sources_in_queue": 0,
            "condor_jobs_prepared": 116,
            "durable_marker_created_before_each_source_access": True,
            "automatic_source_rerun_authorized": False,
            "validation_common_bundle_sha256": common_sha,
            "validation_common_member_sha256": common_hashes,
            "validation_runtime_bundle_sha256": runtime_sha,
            "validation_runtime_remote_path": runtime_remote,
            "expected_runtime_versions": EXPECTED_RUNTIME_VERSIONS,
            "remote_output_root": remote_output,
            "local_log_root": str(args.local_log_root.resolve()),
            "x509_proxy": str(args.x509_proxy.resolve()),
            "apptainer_image": args.apptainer_image,
            "queue_items_sha256": sha256(build / "validation_queue.items"),
            "runner_sha256": sha256(runner_path),
            "submit_file_sha256": sha256(build / "validation_116.submit"),
            "production_submission_performed": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
            "test_access_authorized": False,
            "next": "freeze_and_push_campaign_then_run_remote_empty_preflight_before_one_condor_submit_call",
        }
        write_text(
            build / "campaign_summary.json",
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        artifact_rows = []
        for path in sorted(build.iterdir()):
            if path.is_file():
                artifact_rows.append(
                    {
                        "relative_path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        pd.DataFrame(artifact_rows).to_csv(
            build / "artifact_manifest.tsv", sep="\t", index=False, lineterminator="\n"
        )
        files = sorted(path for path in build.iterdir() if path.is_file())
        write_text(
            build / "SHA256SUMS",
            "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        )
        os.replace(build, output)
    finally:
        # A failed build is intentionally preserved for diagnosis.  A successful
        # build was atomically renamed to the requested output path.
        pass

    print("CUT_BASELINE_VALIDATION_CAMPAIGN_PREPARATION=PASS")
    print("CONDOR_JOBS_PREPARED=116")
    print("PRODUCTION_SUBMISSION_PERFORMED=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
