#!/usr/bin/env python3
"""Build the publication-quality final HH->4b cut-baseline status report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


BRANCH = "delphes-hh4b-production"
NOMINAL_THRESHOLDS = {
    "exact3tag": {
        "r_hh_125_125": 36.40814019639858,
        "ht_candidate_jets": 176.5458068847656,
    },
    "ge4tag": {
        "r_hh_125_125": 33.92808917804956,
        "mhh": 164.73708096689654,
        "abs_h_delta_eta": 6.904302164473993,
    },
}


class FinalReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalReportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def verify_sha256s(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"invalid checksummed root: {root}")
    require((root / "SHA256SUMS").is_file(), f"missing SHA256SUMS: {root}")
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"checksum failure: {root}: {result.stdout}{result.stderr}")


def require_committed_unchanged(repo: Path, root: Path) -> None:
    relative = root.relative_to(repo).as_posix()
    actual = {
        path.relative_to(repo).as_posix() for path in root.rglob("*") if path.is_file()
    }
    tracked = set(git(repo, "ls-files", "--", relative).splitlines())
    require(actual <= tracked, f"artifact contains uncommitted/ignored files: {root}")
    require(
        subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=repo).returncode
        == 0,
        f"artifact has unstaged changes: {root}",
    )
    require(
        subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo
        ).returncode
        == 0,
        f"artifact has staged changes: {root}",
    )


def validate_final_summaries(summaries: dict[str, dict[str, Any]]) -> None:
    expected_statuses = {
        "initial200": "pass_initial200_predeclared_selection_stability_aggregation",
        "all1000": "pass_all1000_predeclared_selection_stability_aggregation",
        "train": "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator",
        "publication_train": "pass_hh4b_cut_baseline_train_only_publication_figures",
        "master": "pass_master_train_only_hh4b_cut_baseline_freeze",
        "authorization": "authorized_one_time_cut_baseline_validation",
        "campaign": "pass_exactly_once_validation_campaign_frozen_before_submission",
        "submission": "pass_exactly_once_validation_submission_frozen_before_monitoring",
        "validation_returns": "pass_complete_one_time_cut_baseline_validation_return_audit",
        "validation": "pass_one_time_hh4b_cut_baseline_validation_aggregation",
        "publication_validation": "pass_cut_baseline_validation_publication_figure",
    }
    for role, expected in expected_statuses.items():
        require(summaries[role].get("status") == expected, f"final input status changed: {role}")
        require(summaries[role].get("test_payloads_opened") == 0, f"test opened in {role}")
    for role in (
        "initial200", "all1000", "train", "publication_train", "master",
        "authorization", "campaign",
    ):
        require(summaries[role].get("validation_payloads_opened") == 0, f"validation opened before authorization in {role}")
    require(summaries["submission"].get("validation_payloads_opened_before_submission") == 0, "validation opened before submission")
    for role in ("validation_returns", "validation", "publication_validation"):
        require(summaries[role].get("validation_payloads_opened") == 1, f"validation cycle count changed in {role}")
    require(summaries["master"].get("nominal_thresholds") == NOMINAL_THRESHOLDS, "master nominal cut changed")
    require(summaries["all1000"].get("nominal_deployment_candidate_changed") is False, "stability changed nominal cut")
    require(summaries["train"].get("primary_generalization_estimate") == "pooled_nested_outer_oof", "primary train estimate changed")
    require(summaries["train"].get("historical_comparator_threshold_scan_performed") is False, "historical comparator was tuned")
    require(summaries["train"].get("historical_comparator_threshold") == {"r_hh_125_125_lt": 34.0}, "historical comparator changed")
    require(summaries["publication_train"].get("official_cms_status_claimed") is False, "train figures claim official CMS status")
    require(summaries["validation_returns"].get("physical_validation_sources_opened_once") == 116, "validation source closure changed")
    require(summaries["validation_returns"].get("source_payload_reruns") == 0, "validation source was rerun")
    require(summaries["validation"].get("nominal_selection_changed") is False, "validation changed nominal cut")
    require(summaries["validation"].get("validation_payload_reruns") == 0, "validation aggregation reports rerun")
    for role in ("train", "validation"):
        require(summaries[role].get("official_cms_result") is False, f"{role} claims official CMS status")


def checkpoint_record(repo: Path, root: Path) -> dict[str, Any]:
    verify_sha256s(root)
    require_committed_unchanged(repo, root)
    relative = root.relative_to(repo).as_posix()
    require(
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", f"{relative}/SHA256SUMS"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
        f"checkpoint is not committed: {root}",
    )
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "path": relative,
        "sha256s_sha256": sha256(root / "SHA256SUMS"),
        "files": len(files),
        "last_checkpoint_commit": git(repo, "log", "-1", "--format=%H", "--", relative),
    }


def inventory(paths: list[Path], repo: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(paths)):
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular artifact: {path}")
        rows.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def metric_text(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def metric_markdown_rows(train: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    rows = []
    fields = (
        "signal_physical_efficiency",
        "background_physical_efficiency",
        "background_rejection",
        "signal_selected_signed_yield",
        "background_selected_signed_yield",
        "signal_over_background",
        "asimov_significance_stat_only",
    )
    for selection, scopes in train["pooled_metrics"].items():
        for scope, metrics in scopes.items():
            rows.append(
                "| " + " | ".join([selection, scope, *(metric_text(metrics[field]) for field in fields)]) + " |"
            )
    for scope, metrics in validation["pooled_metrics"].items():
        rows.append(
            "| " + " | ".join(["one_time_validation", scope, *(metric_text(metrics[field]) for field in fields)]) + " |"
        )
    return rows


def stability_markdown(initial: dict[str, Any], all1000: dict[str, Any]) -> list[str]:
    rows = []
    fields = (
        "modal_replica_selected_frequency",
        "nominal_structure_recovery_frequency",
        "top_two_frequency_gap",
        "unique_modal_replica_category_fraction",
        "tie_resolution_replica_category_fraction",
        "infeasible_fold_winner_frequency",
    )
    for label, summary in (("initial200", initial), ("all1000", all1000)):
        for category, values in summary["categories"].items():
            rows.append(
                "| " + " | ".join([label, category, *(metric_text(values[field]) for field in fields)]) + " |"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"))
    parser.add_argument("--initial200-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_aggregation_20260809_v1"))
    parser.add_argument("--all1000-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1"))
    parser.add_argument("--train-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1"))
    parser.add_argument("--publication-train-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_train_cut_baseline_publication_figures_20260810_v1"))
    parser.add_argument("--master-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260810_v1"))
    parser.add_argument("--authorization-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_authorization_20260810_v1"))
    parser.add_argument("--campaign-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_campaign_presubmission_20260810_v1"))
    parser.add_argument("--validation-submission-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_submission_20260810_v1"))
    parser.add_argument("--validation-returns-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_complete_return_audit_20260810_v1"))
    parser.add_argument("--validation-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_performance_20260810_v1"))
    parser.add_argument("--publication-validation-checkpoint", type=Path, default=Path("docs/checkpoints/hh4b_cut_baseline_validation_publication_figure_20260810_v1"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    require(repo == args.repo, "repository path is not canonical")
    require(git(repo, "branch", "--show-current") == BRANCH, "branch changed")
    head = git(repo, "rev-parse", "HEAD")
    require(head == git(repo, "rev-parse", f"origin/{BRANCH}"), "local/remote HEAD mismatch")

    def resolve(value: Path) -> Path:
        path = value if value.is_absolute() else repo / value
        path = path.resolve()
        require(path.is_relative_to(repo), f"checkpoint outside repository: {path}")
        return path

    critical = {
        "initial200": resolve(args.initial200_checkpoint),
        "all1000": resolve(args.all1000_checkpoint),
        "train": resolve(args.train_checkpoint),
        "publication_train": resolve(args.publication_train_checkpoint),
        "master": resolve(args.master_checkpoint),
        "authorization": resolve(args.authorization_checkpoint),
        "campaign": resolve(args.campaign_checkpoint),
        "submission": resolve(args.validation_submission_checkpoint),
        "validation_returns": resolve(args.validation_returns_checkpoint),
        "validation": resolve(args.validation_checkpoint),
        "publication_validation": resolve(args.publication_validation_checkpoint),
    }
    summary_paths = {
        "initial200": critical["initial200"] / "evidence/aggregation/initial200_stability_summary.json",
        "all1000": critical["all1000"] / "evidence/aggregation/all1000_stability_summary.json",
        "train": critical["train"] / "evidence/train_performance/train_performance_summary.json",
        "publication_train": critical["publication_train"] / "evidence/publication_train_only/publication_figure_summary.json",
        "master": critical["master"] / "master_train_only_summary.json",
        "authorization": critical["authorization"] / "validation_authorization.json",
        "campaign": critical["campaign"] / "validation_campaign_presubmission_freeze.json",
        "submission": critical["submission"] / "validation_submission_freeze.json",
        "validation_returns": critical["validation_returns"] / "evidence/validation_returns/validation_return_audit.json",
        "validation": critical["validation"] / "evidence/validation_performance/validation_performance_summary.json",
        "publication_validation": critical["publication_validation"] / "evidence/publication_validation/manifests/figure17_validation_vs_train_oof_provenance.json",
    }
    summaries = {role: load_json(path) for role, path in summary_paths.items()}
    validate_final_summaries(summaries)
    authorized_hashes = summaries["authorization"].get("authorized_sha256", {})
    require(
        authorized_hashes.get("final_report_builder") == sha256(Path(__file__).resolve()),
        "final report builder is not the master-authorized implementation",
    )
    require(
        summaries["campaign"].get("implementation_sha256")
        == authorized_hashes.get("validation_campaign_presubmission_freezer"),
        "campaign freeze was not produced by the master-authorized implementation",
    )
    require(
        summaries["submission"].get("implementation_sha256")
        == authorized_hashes.get("validation_submission_freezer"),
        "submission freeze was not produced by the master-authorized implementation",
    )
    for role in ("validation_returns", "validation", "publication_validation"):
        artifact_freeze = load_json(critical[role] / "artifact_checkpoint_freeze.json")
        require(
            artifact_freeze.get("implementation_sha256")
            == authorized_hashes.get("validation_artifact_freezer"),
            f"{role} checkpoint was not produced by the master-authorized freezer",
        )

    checkpoint_root = repo / "docs/checkpoints"
    prefixes = (
        "hh4b_train_multivariate_cut_",
        "hh4b_train_cut_baseline_",
        "hh4b_train_historical_rhh125125_lt34_",
        "hh4b_cut_baseline_",
    )
    discovered = sorted(
        path for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith(prefixes)
    )
    require(set(critical.values()) <= set(discovered), "critical checkpoint is outside discovered cut-baseline checkpoints")
    checkpoints = [checkpoint_record(repo, path) for path in discovered]

    artifact_roots = {
        "initial200": repo / "artifacts/hh4b_cut_baseline/initial200_stability",
        "all1000": repo / "artifacts/hh4b_cut_baseline/all1000_stability",
        "train": repo / "artifacts/hh4b_cut_baseline/train_performance",
        "publication_train": repo / "artifacts/hh4b_cut_baseline/publication_train_only",
        "validation_returns": repo / "artifacts/hh4b_cut_baseline/validation_returns",
        "validation": repo / "artifacts/hh4b_cut_baseline/validation_performance",
        "publication_validation": repo / "artifacts/hh4b_cut_baseline/publication_validation",
    }
    for root in artifact_roots.values():
        verify_sha256s(root)
        require_committed_unchanged(repo, root)
    pdfs = sorted((artifact_roots["publication_train"] / "figures/pdf").glob("*.pdf"))
    pdfs += sorted((artifact_roots["publication_validation"] / "figures/pdf").glob("*.pdf"))
    pngs = sorted((artifact_roots["publication_train"] / "figures/png").glob("*.png"))
    pngs += sorted((artifact_roots["publication_validation"] / "figures/png").glob("*.png"))
    sidecars = sorted((artifact_roots["publication_train"] / "figure_data").glob("*.tsv"))
    sidecars += sorted((artifact_roots["publication_validation"] / "figure_data").glob("*.tsv"))
    require(len(pdfs) == len(pngs) == len(sidecars) == 17, "publication Figure 1-17 closure changed")
    tables = []
    for role in ("initial200", "all1000", "train", "validation"):
        table_root = artifact_roots[role] / "tables"
        tables.extend(path for path in table_root.rglob("*") if path.is_file())
    tables.extend(
        path for path in artifact_roots["validation_returns"].iterdir()
        if path.is_file() and path.suffix == ".tsv"
    )
    figures = inventory([*pdfs, *pngs, *sidecars], repo)
    table_inventory = inventory(tables, repo)

    paired_path = critical["train"] / "evidence/train_performance/tables/paired_bootstrap_difference_summary.tsv"
    paired_bootstrap = read_tsv(paired_path)
    validation_cluster = summaries["validation_returns"]["cluster_id"]
    validation_schedd = summaries["validation_returns"]["authoritative_schedd"]
    require(
        validation_cluster == summaries["submission"].get("cluster_id")
        and validation_schedd == summaries["submission"].get("authoritative_schedd"),
        "validation submission/return scheduler identity changed",
    )
    clusters = [
        {"cluster_id": 3754344, "role": "bounded_six_job_transfer_pilot", "jobs": 6, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 3755882, "role": "nominal_270", "jobs": 270, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 3768139, "role": "bounded_transfer_pilot", "jobs": 4, "authoritative_schedd": "lpcschedd4.fnal.gov", "never_resubmit": True},
        {"cluster_id": 30002685, "role": "initial200_stability", "jobs": 2000, "authoritative_schedd": "lpcschedd5.fnal.gov", "never_resubmit": True},
        {"cluster_id": 30020809, "role": "escalation800_stability", "jobs": 8000, "authoritative_schedd": "lpcschedd5.fnal.gov", "never_resubmit": True},
        {"cluster_id": validation_cluster, "role": "one_time_validation", "jobs": 116, "authoritative_schedd": validation_schedd, "never_resubmit": True},
    ]
    require(len({row["cluster_id"] for row in clusters}) == 6, "cluster registry is not unique")

    log_lines = git(repo, "log", "--reverse", "--format=%H%x09%aI%x09%s").splitlines()
    git_chain = [
        {"commit": line.split("\t", 2)[0], "authored_at": line.split("\t", 2)[1], "subject": line.split("\t", 2)[2]}
        for line in log_lines
    ]
    report = {
        "schema_version": 1,
        "status": "pass_final_publication_quality_hh4b_cut_baseline_report",
        "repository_parent_head": head,
        "branch": BRANCH,
        "git_commit_chain_through_report_parent": git_chain,
        "checkpoints": checkpoints,
        "clusters": clusters,
        "never_resubmit_registry": [row["cluster_id"] for row in clusters],
        "nominal_thresholds": NOMINAL_THRESHOLDS,
        "initial200_results": summaries["initial200"]["categories"],
        "all1000_results": summaries["all1000"]["categories"],
        "nested_outer_oof_and_fixed_comparator_performance": summaries["train"]["pooled_metrics"],
        "historical_paired_metric_bootstrap": {
            **summaries["train"]["paired_metric_bootstrap"],
            "difference_summary_rows": paired_bootstrap,
        },
        "validation_performance": summaries["validation"]["pooled_metrics"],
        "normalization": {
            "sqrt_s_tev": 13.0,
            "luminosity_fb_inverse": 138.0,
            "luminosity_pb_inverse_internal": 138000.0,
            "interpretation": "Run-2 expected-yield projection",
            "official_cms_result": False,
        },
        "figure_files": figures,
        "table_files": table_inventory,
        "validation_evaluation_cycles": 1,
        "physical_validation_sources_opened_once": 116,
        "validation_source_payload_reruns": 0,
        "auxiliary_qcd_validation_sources_opened": 0,
        "test_access_authorized": False,
        "test_payloads_opened": 0,
        "remaining_before_final_test_cross_model_comparison": [
            "freeze the final BDT, DNN, LBN, SPA-Net, and embedding model choices",
            "freeze the cross-model comparison protocol",
            "create a separate explicit final-test authorization",
        ],
    }

    checkpoint_lines = [
        f"| `{row['path']}` | `{row['sha256s_sha256']}` | `{row['last_checkpoint_commit']}` |"
        for row in checkpoints
    ]
    cluster_lines = [
        f"| {row['cluster_id']} | {row['role']} | {row['jobs']} | `{row['authoritative_schedd']}` | yes |"
        for row in clusters
    ]
    figure_lines = [f"- `{row['path']}` — `{row['sha256']}`" for row in figures]
    table_lines = [f"- `{row['path']}` — `{row['sha256']}`" for row in table_inventory]
    commit_lines = [
        f"- `{row['commit']}` — {row['subject']} ({row['authored_at']})" for row in git_chain
    ]
    markdown = "\n".join(
        [
            "# HH→4b cut-baseline final status",
            "",
            "Status: **PASS — publication-quality cut baseline frozen through one-time validation.**",
            "",
            "This is a Delphes-simulation Run-2 expected-yield projection at $\\sqrt{s}=13$ TeV and 138 fb$^{-1}$ equivalent. It is not an official CMS measurement.",
            "",
            "## Frozen nominal cut",
            "",
            "- exact3tag: $R_{HH}(125,125)<36.40814019639858$ and $H_T^{\\mathrm{cand.}}>176.5458068847656$ GeV.",
            "- ge4tag: $R_{HH}(125,125)<33.92808917804956$, $m_{HH}>164.73708096689654$ GeV, and $|\\Delta\\eta(H_1,H_2)|<6.904302164473993$.",
            "",
            "Selection stability is diagnostic and did not alter these thresholds.",
            "",
            "## Selection-stability diagnostics",
            "",
            "| campaign | category | modal frequency | nominal recovery | top-two gap | unique-mode fraction | tie fraction | infeasible fold-winner frequency |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *stability_markdown(summaries["initial200"], summaries["all1000"]),
            "",
            "## Train and validation performance",
            "",
            "The primary train-side generalization estimate is pooled five-fold nested outer-OOF performance. The fixed deployment cut on train is diagnostic; the historical $R_{HH}<34$ comparator was never tuned here.",
            "",
            "| selection | scope | $\\epsilon_S$ | $\\epsilon_{\\mathrm{bkg}}$ | $1/\\epsilon_{\\mathrm{bkg}}$ | signal yield | background yield | S/B | $Z_A$ stat-only |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            *metric_markdown_rows(summaries["train"], summaries["validation"]),
            "",
            "Full signed-yield, sumw2, $N_{\\mathrm{eff}}$, finite-MC uncertainty, per-fold, and paired-bootstrap values are in the machine-readable tables listed below and in the JSON report.",
            "",
            "## Condor provenance and never-resubmit registry",
            "",
            "| cluster | role | jobs | authoritative schedd | never resubmit |",
            "|---:|---|---:|---|---|",
            *cluster_lines,
            "",
            "## Checkpoints",
            "",
            "| checkpoint | SHA256(SHA256SUMS) | last checkpoint commit |",
            "|---|---|---|",
            *checkpoint_lines,
            "",
            "## Publication figures and sidecars",
            "",
            *figure_lines,
            "",
            "## Machine-readable tables",
            "",
            *table_lines,
            "",
            "## Git commit chain through the report parent",
            "",
            *commit_lines,
            "",
            "## Final seals and remaining gate",
            "",
            "- Validation evaluation cycles: **1**; 116 physical sources opened exactly once; source reruns: **0**.",
            "- Auxiliary-QCD validation sources opened: **0**.",
            "- `TEST_ACCESS_AUTHORIZED=FALSE`.",
            "- `TEST_PAYLOADS_OPENED=0`.",
            "- Before test: freeze every remaining model choice, freeze the cross-model protocol, and create a separate final-test authorization.",
            "",
        ]
    )

    root_md = repo / "HH4B_CUT_BASELINE_FINAL_STATUS.md"
    root_json = repo / "HH4B_CUT_BASELINE_FINAL_STATUS.json"
    reports = repo / "artifacts/hh4b_cut_baseline/reports"
    require(reports.parent.is_dir() and not reports.is_symlink(), "final report parent is invalid")
    report_md = reports / root_md.name
    report_json = reports / root_json.name
    sums = reports / "SHA256SUMS"
    for path in (root_md, root_json, report_md, report_json, sums):
        require(not path.exists(), f"final report output already exists: {path}")
    reports.mkdir(exist_ok=True)
    json_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    md_bytes = markdown.encode("utf-8")
    build = repo / f".hh4b_cut_baseline_final_report.build.{os.getpid()}"
    require(not build.exists(), "final report build directory already exists")
    build.mkdir()
    (build / root_md.name).write_bytes(md_bytes)
    (build / root_json.name).write_bytes(json_bytes)
    os.replace(build / root_md.name, root_md)
    os.replace(build / root_json.name, root_json)
    report_md.write_bytes(md_bytes)
    report_json.write_bytes(json_bytes)
    sums.write_text(
        f"{sha256(report_md)}  {report_md.name}\n"
        f"{sha256(report_json)}  {report_json.name}\n"
        f"{sha256(root_md)}  ../../../{root_md.name}\n"
        f"{sha256(root_json)}  ../../../{root_json.name}\n",
        encoding="utf-8",
    )
    build.rmdir()
    print("HH4B_CUT_BASELINE_FINAL_REPORT=PASS")
    print(f"CHECKPOINTS={len(checkpoints)}")
    print("FIGURES=17_PDF_17_PNG_17_SIDECARS")
    print("VALIDATION_EVALUATION_CYCLES=1")
    print("TEST_ACCESS_AUTHORIZED=FALSE")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
