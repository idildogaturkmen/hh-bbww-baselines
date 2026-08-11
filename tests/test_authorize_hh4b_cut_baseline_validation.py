from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from authorize_hh4b_cut_baseline_validation import (  # noqa: E402
    NOMINAL_THRESHOLDS,
    AuthorizationError,
    validate_master_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZER = SCRIPT_DIR / "authorize_hh4b_cut_baseline_validation.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(list(args), cwd=cwd, text=True).strip()


def valid_summary():
    return {
        "status": "pass_master_train_only_hh4b_cut_baseline_freeze",
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "nominal_selection_changed": False,
        "nominal_thresholds": NOMINAL_THRESHOLDS,
        "cut_family_frozen": True,
        "variables_frozen": True,
        "thresholds_frozen": True,
        "reporting_choices_frozen": True,
    }


def test_valid_master_contract_passes() -> None:
    validate_master_summary(valid_summary())


def test_opened_validation_fails_closed() -> None:
    summary = valid_summary()
    summary["validation_payloads_opened"] = 1
    try:
        validate_master_summary(summary)
    except AuthorizationError:
        pass
    else:
        raise AssertionError("opened validation passed authorization gate")


def test_nominal_threshold_change_fails_closed() -> None:
    summary = valid_summary()
    summary["nominal_thresholds"] = {
        **NOMINAL_THRESHOLDS,
        "exact3tag": {
            **NOMINAL_THRESHOLDS["exact3tag"],
            "r_hh_125_125": 36.0,
        },
    }
    try:
        validate_master_summary(summary)
    except AuthorizationError:
        pass
    else:
        raise AssertionError("changed nominal threshold passed authorization gate")


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = AUTHORIZER.read_text(encoding="utf-8")
    verify_body = source.split("def verify_repo", 1)[1].split("def verify_checkpoint", 1)[0]
    assert "status\", \"--porcelain" not in verify_body
    assert "head == remote" in verify_body


def test_full_authorization_binds_all_execution_and_return_code() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repository = root / "repo"
        remote = root / "remote.git"
        output = root / "authorization_output"
        repository.mkdir()
        run("git", "init", "--bare", str(remote), cwd=root)
        run("git", "init", str(repository), cwd=root)
        run("git", "config", "user.email", "authorization-test@example.invalid", cwd=repository)
        run("git", "config", "user.name", "Authorization Test", cwd=repository)
        run("git", "checkout", "-b", "delphes-hh4b-production", cwd=repository)
        run("git", "remote", "add", "origin", str(remote), cwd=repository)

        master_checkpoint = repository / "docs/master"
        metadata_checkpoint = repository / "docs/metadata"
        master_checkpoint.mkdir(parents=True)
        metadata_checkpoint.mkdir(parents=True)
        master_summary = master_checkpoint / "master_summary.json"
        master_summary.write_text(
            json.dumps(valid_summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (master_checkpoint / "SHA256SUMS").write_text(
            f"{sha(master_summary)}  master_summary.json\n", encoding="utf-8"
        )

        access_rows = []
        coefficient_rows = []
        for index in range(121):
            physical = index < 116
            uid = f"source_{index:04d}"
            access_rows.append(
                {
                    "production_row_index": index,
                    "source_uid": uid,
                    "physical_evaluation_eligible": physical,
                    "auxiliary_qcd": not physical,
                    "generated_events": 10,
                    "validation_content_opened": False,
                    "validation_access_authorized": False,
                    "test_content_opened": False,
                }
            )
            coefficient_rows.append(
                {
                    "source_uid": uid,
                    "run2_yield_coefficient_per_generator_weight": 1.0 if physical else "",
                }
            )
        access = metadata_checkpoint / "source_access.tsv"
        coefficients = metadata_checkpoint / "coefficients.tsv"
        pd.DataFrame(access_rows).to_csv(
            access, sep="\t", index=False, lineterminator="\n"
        )
        pd.DataFrame(coefficient_rows).to_csv(
            coefficients, sep="\t", index=False, lineterminator="\n"
        )
        (metadata_checkpoint / "SHA256SUMS").write_text(
            f"{sha(access)}  source_access.tsv\n"
            f"{sha(coefficients)}  coefficients.tsv\n",
            encoding="utf-8",
        )
        runtime = repository / "runtime.tar.gz"
        runtime.write_bytes(b"synthetic pinned runtime")
        code_directory = repository / "scripts/analysis"
        code_directory.mkdir(parents=True)
        source_code_paths = {
            "--broad-feature-extractor": SCRIPT_DIR / "hh4b_broad_feature_extractor_v1.py",
            "--candidate-reconstruction-module": SCRIPT_DIR / "hh4b_train_common_table_single_source_worker_v1.py",
            "--validation-source-worker": SCRIPT_DIR / "run_hh4b_cut_baseline_validation_source.py",
            "--validation-aggregator": SCRIPT_DIR / "aggregate_hh4b_cut_baseline_validation.py",
            "--validation-plotter": SCRIPT_DIR / "plot_hh4b_cut_baseline_validation.py",
            "--validation-campaign-preparer": SCRIPT_DIR / "prepare_hh4b_cut_baseline_validation_campaign.py",
            "--validation-campaign-auditor": SCRIPT_DIR / "audit_hh4b_cut_baseline_validation_campaign.py",
            "--validation-campaign-presubmission-freezer": SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_campaign_presubmission.py",
            "--validation-campaign-submitter": SCRIPT_DIR / "submit_hh4b_cut_baseline_validation_once.py",
            "--validation-submission-freezer": SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_submission.py",
            "--validation-return-auditor": SCRIPT_DIR / "audit_hh4b_cut_baseline_validation_returns.py",
            "--validation-artifact-freezer": SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_artifact_checkpoint.py",
            "--final-report-builder": SCRIPT_DIR / "build_hh4b_cut_baseline_final_report.py",
        }
        paths = {}
        for option, source_path in source_code_paths.items():
            destination = code_directory / source_path.name
            destination.write_bytes(source_path.read_bytes())
            paths[option] = destination
        code_manifest = master_checkpoint / "code_manifest.tsv"
        pd.DataFrame(
            [
                {
                    "relative_path": path.relative_to(repository).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
                for path in paths.values()
            ]
        ).to_csv(code_manifest, sep="\t", index=False, lineterminator="\n")
        (master_checkpoint / "SHA256SUMS").write_text(
            f"{sha(code_manifest)}  code_manifest.tsv\n"
            f"{sha(master_summary)}  master_summary.json\n",
            encoding="utf-8",
        )
        run("git", "add", ".", cwd=repository)
        run("git", "commit", "-m", "freeze synthetic master and metadata", cwd=repository)
        master_commit = run("git", "rev-parse", "HEAD", cwd=repository)
        run("git", "push", "-u", "origin", "delphes-hh4b-production", cwd=repository)

        command = [
            sys.executable,
            str(AUTHORIZER),
            "--repo",
            str(repository.resolve()),
            "--master-checkpoint",
            str(master_checkpoint),
            "--master-summary",
            str(master_summary),
            "--master-checkpoint-commit",
            master_commit,
            "--validation-metadata-checkpoint",
            str(metadata_checkpoint),
            "--source-access-manifest",
            str(access),
            "--physical-coefficient-registry",
            str(coefficients),
        ]
        for option, path in paths.items():
            command.extend([option, str(path)])
        command.extend(
            [
                "--validation-runtime-bundle",
                str(runtime),
                "--output-dir",
                str(output),
            ]
        )
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        authorization = json.loads(
            (output / "validation_authorization.json").read_text()
        )
        assert authorization["repository_head"] == master_commit
        assert authorization["authorized_validation_sources"] == 116
        assert authorization["validation_payloads_opened"] == 0
        assert authorization["test_payloads_opened"] == 0
        assert authorization["test_access_authorized"] is False
        for option, path in paths.items():
            key = option.removeprefix("--").replace("-", "_")
            assert authorization["authorized_sha256"][key] == sha(path)
        assert authorization["authorized_sha256"]["validation_runtime_bundle"] == sha(runtime)
        check = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert check.returncode == 0


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
