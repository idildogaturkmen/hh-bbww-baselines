from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/analysis/prepare_hh4b_cut_baseline_validation_campaign.py"
sys.path.insert(0, str(SCRIPT.parent))

from prepare_hh4b_cut_baseline_validation_campaign import (  # noqa: E402
    build_common_bundle,
    render_runner,
)
from audit_hh4b_cut_baseline_validation_campaign import (  # noqa: E402
    audit_local_package,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(list(args), cwd=cwd, text=True).strip()


def test_common_bundle_is_byte_reproducible_and_internally_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {}
        for key in (
            "source_access_manifest",
            "physical_coefficient_registry",
            "broad_feature_extractor",
            "candidate_reconstruction_module",
            "validation_source_worker",
        ):
            path = root / key
            path.write_text(f"{key}\n", encoding="utf-8")
            paths[key] = path
        authorization = root / "authorization.json"
        authorization.write_text("{}\n", encoding="utf-8")
        first = root / "first.tar.gz"
        second = root / "second.tar.gz"
        hashes = build_common_bundle(first, paths, authorization)
        build_common_bundle(second, paths, authorization)
        assert first.read_bytes() == second.read_bytes()
        with tarfile.open(first, "r:gz") as archive:
            names = set(archive.getnames())
            assert names == {
                "COMMON_SHA256SUMS",
                "extractor.py",
                "physical_coefficient_registry.tsv",
                "reconstruction.py",
                "source_access_manifest.tsv",
                "validation_authorization.json",
                "worker.py",
            }
            for name, expected in hashes.items():
                payload = archive.extractfile(name).read()
                assert hashlib.sha256(payload).hexdigest() == expected


def test_runner_places_irreversible_marker_before_source_worker() -> None:
    runner = render_runner(
        execution_head="1" * 40,
        authorization_head="0" * 40,
        authorization_sha="2" * 64,
        common_sha="3" * 64,
        runtime_sha="4" * 64,
        runtime_remote_path="/store/user/iturkmen/runtime.tar.gz",
        remote_output_root="/store/user/iturkmen/validation",
    )
    marker_upload = 'xrdcp --nopbar --cksum adler32 "$LOCAL_MARKER"'
    worker_start = 'python3 "$SCRATCH/common/worker.py"'
    assert marker_upload in runner
    assert worker_start in runner
    assert runner.index(marker_upload) < runner.index("MARKER_CREATED=TRUE")
    assert runner.index("MARKER_CREATED=TRUE") < runner.index(worker_start)
    assert "--preexisting-attempt-marker" in runner
    assert "--durable-attempt-marker-uri" in runner
    assert "rerun_forbidden_even_if_downstream_bookkeeping_fails" in runner
    assert 'xrdcp -f --nopbar --cksum adler32 "$LOCAL_MARKER"' not in runner
    assert 'export PATH="$SCRATCH/runtime/bin:$PATH"' in runner
    assert (
        'export PYTHONPATH="$SCRATCH/runtime/lib/python3.9/site-packages"'
        in runner
    )
    assert 'export PYTHONPATH="$SCRATCH/runtime/site-packages"' not in runner


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    verify_body = source.split(
        "def verify_repository_and_authorization_checkpoint", 1
    )[1].split("def validate_remote_path", 1)[0]
    assert "status\", \"--porcelain" not in verify_body
    assert "head == remote" in verify_body


def test_full_preparation_is_metadata_only_and_builds_116_jobs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repository = root / "repo"
        remote = root / "remote.git"
        package = root / "package"
        logs = root / "condor_submit" / "logs"
        proxy = root / "x509up"
        repository.mkdir()
        logs.mkdir(parents=True)
        proxy.write_text("synthetic proxy\n", encoding="utf-8")
        run("git", "init", "--bare", str(remote), cwd=root)
        run("git", "init", str(repository), cwd=root)
        run("git", "config", "user.email", "validation-test@example.invalid", cwd=repository)
        run("git", "config", "user.name", "Validation Test", cwd=repository)
        run("git", "checkout", "-b", "delphes-hh4b-production", cwd=repository)
        run("git", "remote", "add", "origin", str(remote), cwd=repository)

        inputs = repository / "inputs"
        inputs.mkdir()
        access_rows = []
        coefficient_rows = []
        for index in range(121):
            physical = index < 116
            uid = (
                "source_0000::background::4::"
                "bundle:/store/user/iturkmen/frozen/source_0000.tar.gz"
                if index == 0
                else f"source_{index:04d}"
            )
            access_rows.append(
                {
                    "production_row_index": index,
                    "source_uid": uid,
                    "split": "validation",
                    "physical_evaluation_eligible": physical,
                    "auxiliary_qcd": not physical,
                    "generated_events": 10,
                    "validation_content_opened": False,
                    "test_content_opened": False,
                }
            )
            coefficient_rows.append(
                {
                    "source_uid": uid,
                    "run2_yield_coefficient_per_generator_weight": 1.0 if physical else "",
                }
            )
        access = inputs / "source_access.tsv"
        coefficients = inputs / "coefficients.tsv"
        pd.DataFrame(access_rows).to_csv(access, sep="\t", index=False, lineterminator="\n")
        pd.DataFrame(coefficient_rows).to_csv(coefficients, sep="\t", index=False, lineterminator="\n")
        extractor = inputs / "extractor.py"
        reconstruction = inputs / "reconstruction.py"
        worker = inputs / "worker.py"
        runtime = inputs / "runtime.tar.gz"
        extractor.write_text("# extractor\n", encoding="utf-8")
        reconstruction.write_text("# reconstruction\n", encoding="utf-8")
        worker.write_text("# worker\n", encoding="utf-8")
        runtime.write_bytes(b"synthetic runtime")
        run("git", "add", "inputs", cwd=repository)
        run("git", "commit", "-m", "master train-only inputs", cwd=repository)
        master = run("git", "rev-parse", "HEAD", cwd=repository)

        checkpoint = repository / "docs/validation_authorization"
        checkpoint.mkdir(parents=True)
        authorization = checkpoint / "validation_authorization.json"
        authorized_paths = {
            "source_access_manifest": access,
            "physical_coefficient_registry": coefficients,
            "broad_feature_extractor": extractor,
            "candidate_reconstruction_module": reconstruction,
            "validation_source_worker": worker,
            "validation_campaign_preparer": SCRIPT,
            "validation_campaign_auditor": SCRIPT,
            "validation_campaign_submitter": SCRIPT,
            "validation_return_auditor": SCRIPT,
            "validation_runtime_bundle": runtime,
        }
        authorization_payload = {
            "schema_version": 1,
            "status": "authorized_one_time_cut_baseline_validation",
            "repository_head": master,
            "master_train_only_checkpoint_commit": master,
            "validation_access_authorized": True,
            "validation_payloads_opened_before_authorization": 0,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
            "test_access_authorized": False,
            "authorized_validation_sources": 116,
            "auxiliary_qcd_validation_sources_authorized": 0,
            "authorized_generated_events": 1160,
            "authorized_sha256": {
                key: sha(path) for key, path in authorized_paths.items()
            },
        }
        authorization.write_text(
            json.dumps(authorization_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (checkpoint / "SHA256SUMS").write_text(
            f"{sha(authorization)}  validation_authorization.json\n",
            encoding="utf-8",
        )
        run("git", "add", "docs/validation_authorization", cwd=repository)
        run("git", "commit", "-m", "authorize validation", cwd=repository)
        authorization_commit = run("git", "rev-parse", "HEAD", cwd=repository)
        run("git", "push", "-u", "origin", "delphes-hh4b-production", cwd=repository)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(repository.resolve()),
                "--authorization-checkpoint",
                str(checkpoint),
                "--authorization",
                str(authorization),
                "--authorization-checkpoint-commit",
                authorization_commit,
                "--source-access-manifest",
                str(access),
                "--physical-coefficient-registry",
                str(coefficients),
                "--broad-feature-extractor",
                str(extractor),
                "--candidate-reconstruction-module",
                str(reconstruction),
                "--validation-source-worker",
                str(worker),
                "--validation-campaign-auditor",
                str(SCRIPT),
                "--validation-campaign-submitter",
                str(SCRIPT),
                "--validation-return-auditor",
                str(SCRIPT),
                "--validation-runtime-bundle",
                str(runtime),
                "--runtime-remote-path",
                "/store/user/iturkmen/runtime.tar.gz",
                "--remote-output-root",
                "/store/user/iturkmen/validation_once",
                "--local-log-root",
                str(logs),
                "--x509-proxy",
                str(proxy),
                "--output-dir",
                str(package),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PRODUCTION_SUBMISSION_PERFORMED=FALSE" in result.stdout
        assert len((package / "validation_queue.items").read_text().splitlines()) == 116
        assert (
            package / "validation_queue.items"
        ).read_text().splitlines()[0].endswith(
            "source_0000::background::4::"
            "bundle:/store/user/iturkmen/frozen/source_0000.tar.gz"
        )
        summary = json.loads((package / "campaign_summary.json").read_text())
        assert summary["condor_jobs_prepared"] == 116
        assert summary["production_submission_performed"] is False
        assert summary["validation_payloads_opened"] == 0
        assert summary["test_payloads_opened"] == 0
        assert not (package / "submission_receipt.json").exists()
        audit = audit_local_package(
            package,
            authorization,
            access,
            runtime,
            SCRIPT,
        )
        assert audit["condor_jobs"] == 116
        assert audit["production_submission_performed"] is False
        assert audit["validation_payloads_opened"] == 0
        assert audit["test_payloads_opened"] == 0


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
