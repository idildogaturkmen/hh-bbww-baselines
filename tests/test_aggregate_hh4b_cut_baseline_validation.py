from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_hh4b_cut_baseline_validation import (  # noqa: E402
    asimov_significance,
    combine_summaries,
    effective_events,
    METRICS,
    performance_row,
    relative_mc_uncertainty,
)
from run_hh4b_cut_baseline_validation_source import (  # noqa: E402
    DISTRIBUTION_SPECS,
    NOMINAL_THRESHOLDS,
)


AGGREGATOR = SCRIPT_DIR / "aggregate_hh4b_cut_baseline_validation.py"
WORKER = SCRIPT_DIR / "run_hh4b_cut_baseline_validation_source.py"
FREEZER = SCRIPT_DIR / "freeze_hh4b_cut_baseline_validation_artifact_checkpoint.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_summary(scale: int = 1):
    return {
        "signal": {
            "total_rows": 10 * scale,
            "selected_rows": 5 * scale,
            "negative_weight_rows": 0,
            "selected_negative_weight_rows": 0,
            "total_signed_yield": 100.0 * scale,
            "selected_signed_yield": 50.0 * scale,
            "total_sumw2": 1000.0 * scale,
            "selected_sumw2": 500.0 * scale,
        },
        "background": {
            "total_rows": 20 * scale,
            "selected_rows": 2 * scale,
            "negative_weight_rows": 1 * scale,
            "selected_negative_weight_rows": 0,
            "total_signed_yield": 1000.0 * scale,
            "selected_signed_yield": 100.0 * scale,
            "total_sumw2": 10000.0 * scale,
            "selected_sumw2": 1000.0 * scale,
        },
    }


def test_metric_helpers() -> None:
    assert effective_events(10.0, 4.0) == 25.0
    assert relative_mc_uncertainty(10.0, 4.0) == 0.2
    assert np.isclose(
        asimov_significance(417.46273293569186, 20240622849.775902),
        0.0029343085318647984,
        rtol=2.0e-15,
    )


def test_combination_and_validation_performance() -> None:
    combined = combine_summaries([sample_summary(), sample_summary(2)])
    assert combined["signal"]["total_rows"] == 30
    assert combined["background"]["selected_signed_yield"] == 300.0
    row = performance_row("combined", combined)
    assert row["signal_physical_efficiency"] == 0.5
    assert row["background_physical_efficiency"] == 0.1
    assert row["background_rejection"] == 10.0
    assert row["signal_over_background"] == 0.5
    assert row["validation_payloads_opened"] == 1
    assert row["test_payloads_opened"] == 0


def test_full_synthetic_116_source_aggregation_is_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        return_checkpoint = root / "return_checkpoint"
        frozen_return_root = return_checkpoint / "evidence/validation_returns"
        source_outputs = frozen_return_root / "source_outputs"
        train = root / "train"
        (train / "tables").mkdir(parents=True)
        source_outputs.mkdir(parents=True)
        execution_head = "1" * 40
        authorization_head = "0" * 40

        access_rows = []
        for index in range(121):
            physical = index < 116
            access_rows.append(
                {
                    "production_row_index": index,
                    "source_uid": f"source_{index:04d}",
                    "sample_class": "signal" if index < 19 else "background",
                    "process_or_mode": "synthetic",
                    "physical_evaluation_eligible": physical,
                    "auxiliary_qcd": not physical,
                }
            )
        access_path = root / "source_access.tsv"
        pd.DataFrame(access_rows).to_csv(
            access_path, sep="\t", index=False, lineterminator="\n"
        )
        authorization_path = root / "authorization.json"
        authorization = {
            "status": "authorized_one_time_cut_baseline_validation",
            "repository_head": authorization_head,
            "master_train_only_checkpoint_commit": authorization_head,
            "nominal_thresholds": NOMINAL_THRESHOLDS,
            "test_payloads_opened": 0,
            "authorized_sha256": {
                "source_access_manifest": sha(access_path),
                "validation_source_worker": sha(WORKER),
                "validation_aggregator": sha(AGGREGATOR),
                "validation_artifact_freezer": sha(FREEZER),
            },
        }
        authorization_path.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        authorization_sha = sha(authorization_path)

        base_distribution_rows = []
        for category in ("exact3tag", "ge4tag"):
            for stage in ("preselection", "postselection"):
                for variable, (minimum, maximum, bins, label) in DISTRIBUTION_SPECS.items():
                    edges = np.linspace(minimum, maximum, bins + 1)
                    for bin_index in range(bins):
                        base_distribution_rows.append(
                            {
                                "category_id": category,
                                "selection_stage": stage,
                                "selection_id": "fixed_nominal_deployment_cut",
                                "variable": variable,
                                "axis_label_latex": label,
                                "bin_index": bin_index,
                                "bin_low_inclusive": edges[bin_index],
                                "bin_high_exclusive": edges[bin_index + 1],
                                "rows": 1,
                                "signed_yield": 1.0,
                                "sumw2": 1.0,
                                "underflow_rows_distribution_total": 0,
                                "overflow_rows_distribution_total": 0,
                                "underflow_signed_yield_distribution_total": 0.0,
                                "overflow_signed_yield_distribution_total": 0.0,
                                "underflow_sumw2_distribution_total": 0.0,
                                "overflow_sumw2_distribution_total": 0.0,
                                "validation_payloads_opened": 1,
                                "test_payloads_opened": 0,
                            }
                        )
        assert len(base_distribution_rows) == 1360
        category_summary = {
            "total_rows": 10,
            "selected_rows": 5,
            "negative_weight_rows": 0,
            "selected_negative_weight_rows": 0,
            "total_signed_yield": 100.0,
            "selected_signed_yield": 50.0,
            "total_sumw2": 1000.0,
            "selected_sumw2": 500.0,
        }
        for index in range(116):
            uid = f"source_{index:04d}"
            sample_class = "signal" if index < 19 else "background"
            marker_path = (
                source_outputs
                / f"source_{index:04d}_VALIDATION_OPEN_DO_NOT_RERUN.json"
            )
            marker = {
                "status": "validation_source_open_attempt_durable_do_not_rerun",
                "repository_head": execution_head,
                "authorization_repository_head": authorization_head,
                "authorization_sha256": authorization_sha,
                "production_row_index": index,
                "source_uid": uid,
                "rerun_forbidden_even_if_downstream_bookkeeping_fails": True,
                "test_payloads_opened": 0,
            }
            marker_path.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            distribution_path = (
                source_outputs / f"source_{index:04d}_distributions.tsv"
            )
            distributions = pd.DataFrame(base_distribution_rows)
            distributions.insert(0, "sample_class", sample_class)
            distributions.insert(0, "source_uid", uid)
            distributions.to_csv(
                distribution_path, sep="\t", index=False, lineterminator="\n"
            )
            summary = {
                "status": "pass_one_time_fixed_nominal_validation_source_evaluation",
                "repository_head": execution_head,
                "authorization_repository_head": authorization_head,
                "authorization_sha256": authorization_sha,
                "production_row_index": index,
                "source_uid": uid,
                "group_id": f"group_{index:04d}",
                "sample_class": sample_class,
                "process_or_mode": "synthetic",
                "categories": {
                    "exact3tag": category_summary,
                    "ge4tag": category_summary,
                },
                "nominal_thresholds": NOMINAL_THRESHOLDS,
                "source_payload_opened_once": True,
                "source_payload_rerun_performed": False,
                "cut_scan_performed": False,
                "threshold_adjustment_performed": False,
                "family_adjustment_performed": False,
                "source_open_attempt_marker": marker_path.name,
                "source_open_attempt_marker_sha256": sha(marker_path),
                "distribution_sha256": sha(distribution_path),
                "validation_payloads_opened": 1,
                "test_payloads_opened": 0,
            }
            (source_outputs / f"source_{index:04d}_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return_audit_path = frozen_return_root / "validation_return_audit.json"
        return_audit_path.write_text(
            json.dumps(
                {
                    "status": "pass_complete_one_time_cut_baseline_validation_return_audit",
                    "repository_head": execution_head,
                    "clean_history_jobs": 116,
                    "physical_validation_sources_opened_once": 116,
                    "source_payload_reruns": 0,
                    "validation_evaluation_cycles": 1,
                    "validation_payloads_opened": 1,
                    "test_payloads_opened": 0,
                    "nominal_thresholds": NOMINAL_THRESHOLDS,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (return_checkpoint / "artifact_checkpoint_freeze.json").write_text(
            json.dumps(
                {
                    "status": "pass_validation_returns_artifact_checkpoint_frozen",
                    "role": "validation_returns",
                    "implementation_sha256": sha(FREEZER),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_files = sorted(path for path in return_checkpoint.rglob("*") if path.is_file())
        (return_checkpoint / "SHA256SUMS").write_text(
            "".join(
                f"{sha(path)}  {path.relative_to(return_checkpoint).as_posix()}\n"
                for path in checkpoint_files
            ),
            encoding="utf-8",
        )
        train_rows = []
        for selection_id in ("nested_outer_oof", "fixed_nominal_deployment_cut"):
            for scope in ("exact3tag", "ge4tag", "combined"):
                train_rows.append(
                    {
                        "selection_id": selection_id,
                        "evaluation_role": "synthetic_train",
                        "scope": scope,
                        "outer_fold": "pooled",
                        **{metric: 1.0 for metric in METRICS},
                    }
                )
        pd.DataFrame(train_rows).to_csv(
            train / "tables/train_performance_metrics.tsv",
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        (train / "train_performance_summary.json").write_text(
            '{"status":"synthetic"}\n', encoding="utf-8"
        )
        output = root / "aggregate"
        result = subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--source-access-manifest",
                str(access_path),
                "--authorization",
                str(authorization_path),
                "--validation-return-checkpoint",
                str(return_checkpoint),
                "--train-performance-root",
                str(train),
                "--validation-worker",
                str(WORKER),
                "--expected-head",
                execution_head,
                "--output-root",
                str(output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(
            (output / "validation_performance_summary.json").read_text()
        )
        assert summary["status"] == "pass_one_time_hh4b_cut_baseline_validation_aggregation"
        assert summary["validation_sources_opened"] == 116
        assert summary["validation_payloads_opened"] == 1
        assert summary["test_payloads_opened"] == 0
        distributions = pd.read_csv(
            output
            / "figure_data/validation_pre_post_nominal_variable_distributions.tsv",
            sep="\t",
        )
        assert len(distributions) == 2720


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
