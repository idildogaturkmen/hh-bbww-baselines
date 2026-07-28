#!/usr/bin/env python3
"""Inventory HH4b normalization provenance without opening candidate content."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPECTED_BACKGROUND_PROCESSES = {
    "bbh_hbb_4fs",
    "ggh_hbb",
    "qcd_bbbb_general",
    "qcd_bbbb_iht400to600",
    "qcd_hardqcd",
    "schannel_single_top",
    "tchannel_antitop",
    "tchannel_top",
    "ttbar_inclusive",
    "tth_hbb",
    "tttt",
    "ttw",
    "ttz_zbb",
    "tw_antitop",
    "tw_top",
    "vbf_hbb",
    "wh_hbb",
    "ww",
    "wwz_zbb",
    "wz_zbb",
    "wzz_zbb",
    "zbbbb",
    "zzz_zbb",
}

EXPECTED_SIGNAL_PROCESSES = {
    "ggf_hh4b",
    "vbf_hh4b",
}

QCD_PROCESSES = {
    "qcd_bbbb_general",
    "qcd_bbbb_iht400to600",
    "qcd_hardqcd",
}

REQUIRED_PROVENANCE_FIELDS = (
    "generator_process_definition",
    "matrix_element_generator",
    "matrix_element_generator_version",
    "parton_shower_generator",
    "parton_shower_version",
    "parton_shower_tune",
    "pdf_set",
    "generator_order",
    "generator_cross_section_pb",
    "generator_cross_section_uncertainty_pb",
    "reference_cross_section_pb",
    "reference_cross_section_order",
    "reference_cross_section_source",
    "k_factor",
    "forced_decay_definition",
    "branching_fraction_convention",
    "branching_fraction_factor",
    "filter_definition",
    "filter_efficiency",
    "sum_generator_weights",
    "sum_abs_generator_weights",
    "negative_weight_fraction",
    "importance_sampling_definition",
    "stitching_group",
    "stitching_phase_space",
    "overlap_removal_policy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def campaign_from_locator(locator: str) -> str:
    marker = "/bundles/"
    if marker in locator:
        return locator.split(marker, 1)[1].split("/", 1)[0]
    path = Path(locator.removeprefix("bundle:"))
    return path.parent.name or "unresolved_campaign"


def energy_tag_from_locators(locators: Iterable[str]) -> str:
    tags = set()
    for locator in locators:
        lowered = locator.lower()
        if "run2_13tev" in lowered:
            tags.add("run2_13tev")
        if "run3_13p6tev" in lowered or "run3_13.6tev" in lowered:
            tags.add("run3_13p6tev")
        if "hl_lhc_14tev" in lowered or "hllhc_14tev" in lowered:
            tags.add("hl_lhc_14tev")
    if not tags:
        return "unresolved"
    return ",".join(sorted(tags))


def normalization_strategy(sample_class: str, process: str) -> tuple[str, str]:
    if sample_class == "signal":
        return (
            "theory_normalized_mc",
            "Normalize generated HH signal to reviewed higher-order reference cross section.",
        )
    if process in QCD_PROCESSES:
        return (
            "data_driven_control_region_primary_mc_projection_secondary",
            "CMS-like primary treatment requires a lower-b-tag control-region transfer; "
            "direct QCD MC normalization is diagnostic only until stitching and closure pass.",
        )
    if process == "ttbar_inclusive":
        return (
            "theory_normalized_mc_with_top_pt_model_review",
            "Use reviewed inclusive ttbar reference cross section and document any differential correction.",
        )
    return (
        "theory_normalized_mc",
        "Normalize to a reviewed process-matched reference cross section after decay/filter conventions are resolved.",
    )


def validate_source_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 630:
        raise RuntimeError(f"source registry rows={len(rows)}, expected 630")

    required_columns = {
        "sample_class",
        "process_or_mode",
        "component",
        "final_split",
        "generated_events",
        "candidate_rows_metadata",
        "source_kind",
        "source_locator",
        "member_id",
        "group_id",
        "candidate_content_opened_in_this_step",
        "candidate_content_access_authorized_now",
        "physical_normalization_authorized",
    }
    missing = required_columns - set(rows[0])
    if missing:
        raise RuntimeError(f"source registry missing columns: {sorted(missing)}")

    class_counts = Counter(row["sample_class"] for row in rows)
    if class_counts != {"background": 520, "signal": 110}:
        raise RuntimeError(f"sample-class counts mismatch: {dict(class_counts)}")

    process_sets = defaultdict(set)
    for row in rows:
        process_sets[row["sample_class"]].add(row["process_or_mode"])

    if process_sets["background"] != EXPECTED_BACKGROUND_PROCESSES:
        raise RuntimeError(
            "background process set mismatch:\n"
            f"observed={sorted(process_sets['background'])}\n"
            f"expected={sorted(EXPECTED_BACKGROUND_PROCESSES)}"
        )
    if process_sets["signal"] != EXPECTED_SIGNAL_PROCESSES:
        raise RuntimeError(
            "signal process set mismatch:\n"
            f"observed={sorted(process_sets['signal'])}\n"
            f"expected={sorted(EXPECTED_SIGNAL_PROCESSES)}"
        )

    totals = defaultdict(int)
    for row in rows:
        generated = int(row["generated_events"])
        if generated <= 0:
            raise RuntimeError(f"nonpositive generated-events value: {row}")
        totals[row["sample_class"]] += generated

        for field in (
            "candidate_content_opened_in_this_step",
            "candidate_content_access_authorized_now",
            "physical_normalization_authorized",
        ):
            if parse_bool(row[field]):
                raise RuntimeError(
                    f"protected control unexpectedly true: {field} for {row['member_id']}"
                )

    if totals["background"] != 5_000_000:
        raise RuntimeError(
            f"background generated events={totals['background']}, expected 5000000"
        )
    if totals["signal"] != 200_000:
        raise RuntimeError(
            f"signal generated events={totals['signal']}, expected 200000"
        )

    group_ids = [row["group_id"] for row in rows]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("group IDs are not unique")


def build_process_inventory(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sample_class"], row["process_or_mode"])].append(row)

    output = []
    for (sample_class, process), members in sorted(grouped.items()):
        split_counts = Counter(row["final_split"] for row in members)
        components = sorted({row["component"] for row in members})
        source_kinds = sorted({row["source_kind"] for row in members})
        locators = [row["source_locator"] for row in members]
        campaigns = sorted({campaign_from_locator(locator) for locator in locators})
        strategy, rationale = normalization_strategy(sample_class, process)
        output.append(
            {
                "sample_class": sample_class,
                "process_or_mode": process,
                "members": len(members),
                "generated_events": sum(int(row["generated_events"]) for row in members),
                "candidate_rows_metadata": sum(
                    int(row["candidate_rows_metadata"]) for row in members
                ),
                "train_members": split_counts.get("train", 0),
                "validation_members": split_counts.get("validation", 0),
                "final_evaluation_members": split_counts.get("test", 0),
                "components": ",".join(components),
                "source_kinds": ",".join(source_kinds),
                "campaigns": ",".join(campaigns),
                "energy_tag_inferred_from_path": energy_tag_from_locators(locators),
                "normalization_strategy": strategy,
                "strategy_rationale": rationale,
                "physics_normalization_ready": False,
            }
        )
    return output


def build_campaign_inventory(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        campaign = campaign_from_locator(row["source_locator"])
        key = (row["sample_class"], row["process_or_mode"], campaign)
        grouped[key].append(row)

    output = []
    for (sample_class, process, campaign), members in sorted(grouped.items()):
        split_counts = Counter(row["final_split"] for row in members)
        locators = sorted({row["source_locator"] for row in members})
        output.append(
            {
                "sample_class": sample_class,
                "process_or_mode": process,
                "campaign": campaign,
                "members": len(members),
                "generated_events": sum(int(row["generated_events"]) for row in members),
                "candidate_rows_metadata": sum(
                    int(row["candidate_rows_metadata"]) for row in members
                ),
                "train_members": split_counts.get("train", 0),
                "validation_members": split_counts.get("validation", 0),
                "final_evaluation_members": split_counts.get("test", 0),
                "energy_tag_inferred_from_path": energy_tag_from_locators(locators),
                "source_locator_example": locators[0],
                "generator_artifacts_recovered": False,
                "sum_generator_weights_recovered": False,
                "cross_section_provenance_resolved": False,
                "status": "requires_generator_artifact_recovery",
            }
        )
    return output


def build_registry_template(
    process_inventory: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for row in process_inventory:
        sample_class = str(row["sample_class"])
        process = str(row["process_or_mode"])
        strategy, _ = normalization_strategy(sample_class, process)
        energy_tag = str(row["energy_tag_inferred_from_path"])
        collision_energy = "13000" if energy_tag == "run2_13tev" else ""

        item: dict[str, object] = {
            "sample_class": sample_class,
            "process_or_mode": process,
            "normalization_strategy": strategy,
            "collision_energy_GeV": collision_energy,
            "collision_energy_status": (
                "path_inferred_requires_generator_confirmation"
                if collision_energy
                else "unresolved"
            ),
            "generated_events_manifest": row["generated_events"],
            "generated_events_status": "known_from_frozen_source_registry",
            "normalization_denominator": "",
            "normalization_denominator_status": (
                "unresolved_do_not_assume_generated_event_count"
            ),
        }

        for field in REQUIRED_PROVENANCE_FIELDS:
            item[field] = ""
            item[f"{field}_status"] = "unresolved"

        if process in QCD_PROCESSES:
            item["stitching_group"] = "qcd_multijet_overlap_review"
            item["stitching_group_status"] = "preclassified_requires_proof"
            item["overlap_removal_policy_status"] = "blocking_unresolved"
        else:
            item["stitching_group_status"] = "unresolved_or_not_applicable"

        item["authoritative_physics_ready"] = False
        item["blocking_reason"] = (
            "Generator definition, cross-section provenance, decay/filter convention, "
            "normalization denominator, and overlap policy are not yet fully resolved."
        )
        output.append(item)
    return output


def build_artifact_requirements(
    campaign_inventory: list[dict[str, object]],
) -> list[dict[str, object]]:
    artifact_types = (
        (
            "generator_banner_or_lhe_init",
            "Required",
            "Generator-reported cross section, event-weight strategy, and process metadata.",
        ),
        (
            "proc_card",
            "Required",
            "Exact hard-process definition and decay syntax.",
        ),
        (
            "run_card",
            "Required",
            "Beam energy, cuts, matching/merging, event normalization, and generation settings.",
        ),
        (
            "param_card",
            "Required",
            "Masses, widths, couplings, and decay tables used at generation.",
        ),
        (
            "generator_stdout_or_summary",
            "Required",
            "Cross-section integration result, accepted events, and filter information.",
        ),
        (
            "sum_generator_weights_record",
            "Required",
            "Normalization denominator, including signed-weight handling.",
        ),
        (
            "parton_shower_card",
            "Required",
            "Shower version, tune, forced decays, and matching configuration.",
        ),
        (
            "delphes_card_and_version",
            "Required",
            "Detector-response provenance for the projection.",
        ),
        (
            "importance_sampling_record",
            "Conditional",
            "Required for biased-tail or phase-space-enhanced campaigns.",
        ),
        (
            "stitching_definition",
            "Conditional",
            "Required for overlapping inclusive, sliced, or filtered campaigns.",
        ),
    )

    output = []
    for campaign in campaign_inventory:
        for artifact_type, requirement, purpose in artifact_types:
            output.append(
                {
                    "sample_class": campaign["sample_class"],
                    "process_or_mode": campaign["process_or_mode"],
                    "campaign": campaign["campaign"],
                    "artifact_type": artifact_type,
                    "requirement": requirement,
                    "purpose": purpose,
                    "recovered": False,
                    "artifact_path": "",
                    "sha256": "",
                    "review_status": "pending",
                }
            )
    return output


def build_qcd_risk_rows() -> list[dict[str, object]]:
    pairs = (
        (
            "qcd_bbbb_general",
            "qcd_bbbb_iht400to600",
            "Potential inclusive-versus-IHT-sliced phase-space overlap.",
        ),
        (
            "qcd_bbbb_general",
            "qcd_hardqcd",
            "Potential heavy-flavor matrix-element versus inclusive hard-QCD overlap.",
        ),
        (
            "qcd_bbbb_iht400to600",
            "qcd_hardqcd",
            "Potential IHT-sliced versus hard-QCD overlap.",
        ),
    )
    return [
        {
            "process_a": a,
            "process_b": b,
            "risk": risk,
            "required_evidence": (
                "Exact process cards, run-card phase-space cuts, generator cross sections, "
                "and an explicit mutually exclusive stitching rule."
            ),
            "status": "blocking_unresolved",
            "direct_mc_yield_use_authorized": False,
        }
        for a, b, risk in pairs
    ]


def write_readme(path: Path) -> None:
    path.write_text(
        """# HH4b physical-normalization provenance inventory

This checkpoint inventories the exact source population and creates a fail-closed
normalization registry template. It intentionally assigns no cross sections and computes
no physical yields.

The inventory follows a CMS-like separation of responsibilities:

- HH signal and minor backgrounds: theory-normalized simulation after process-matched
  generator provenance and higher-order reference cross sections are reviewed.
- ttbar: theory-normalized simulation with a separately documented differential-modeling
  review.
- QCD multijet: lower-b-tag control-region prediction is the intended primary treatment;
  direct QCD simulation normalization is a secondary projection only after overlap and
  stitching closure.

The source registry contains 630 indivisible members, 5,000,000 generated background
events, and 200,000 generated signal events. Candidate parquet files, validation content,
and final-evaluation content were not opened.

## Blocking rule

No process is authorized for physical weighting until its generator definition, reference
cross section, branching-fraction convention, filter efficiency, normalization denominator,
and overlap policy are resolved with source paths and checksums.

## Next gate

Recover and checksum generator artifacts campaign by campaign without opening candidate
parquet content.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_registry = args.source_registry.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise RuntimeError(f"output already exists: {output_dir}")
    if not source_registry.is_file():
        raise RuntimeError(f"source registry is absent: {source_registry}")

    rows = read_tsv(source_registry)
    validate_source_rows(rows)

    output_dir.mkdir(parents=True)
    try:
        process_inventory = build_process_inventory(rows)
        campaign_inventory = build_campaign_inventory(rows)
        registry_template = build_registry_template(process_inventory)
        artifact_requirements = build_artifact_requirements(campaign_inventory)
        qcd_risks = build_qcd_risk_rows()

        write_tsv(
            output_dir / "process_inventory.tsv",
            process_inventory,
            list(process_inventory[0]),
        )
        write_tsv(
            output_dir / "campaign_inventory.tsv",
            campaign_inventory,
            list(campaign_inventory[0]),
        )
        write_tsv(
            output_dir / "provenance_registry_template.tsv",
            registry_template,
            list(registry_template[0]),
        )
        write_tsv(
            output_dir / "artifact_requirements.tsv",
            artifact_requirements,
            list(artifact_requirements[0]),
        )
        write_tsv(
            output_dir / "qcd_overlap_risks.tsv",
            qcd_risks,
            list(qcd_risks[0]),
        )

        strategy_rows = []
        for row in process_inventory:
            strategy, rationale = normalization_strategy(
                str(row["sample_class"]),
                str(row["process_or_mode"]),
            )
            strategy_rows.append(
                {
                    "sample_class": row["sample_class"],
                    "process_or_mode": row["process_or_mode"],
                    "strategy": strategy,
                    "rationale": rationale,
                    "direct_mc_physics_yield_authorized": False,
                }
            )
        write_tsv(
            output_dir / "normalization_strategy.tsv",
            strategy_rows,
            list(strategy_rows[0]),
        )

        write_readme(output_dir / "README.md")

        summary = {
            "schema_version": 1,
            "status": "hh4b_physical_normalization_provenance_inventory_pass",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "source_registry": str(args.source_registry),
            "source_registry_sha256": sha256_file(source_registry),
            "population": {
                "members": len(rows),
                "background_members": sum(
                    row["sample_class"] == "background" for row in rows
                ),
                "signal_members": sum(
                    row["sample_class"] == "signal" for row in rows
                ),
                "background_generated_events": sum(
                    int(row["generated_events"])
                    for row in rows
                    if row["sample_class"] == "background"
                ),
                "signal_generated_events": sum(
                    int(row["generated_events"])
                    for row in rows
                    if row["sample_class"] == "signal"
                ),
                "process_labels": len(process_inventory),
                "campaigns": len(campaign_inventory),
            },
            "readiness": {
                "processes_physics_ready": 0,
                "processes_total": len(process_inventory),
                "physics_normalization_ready": False,
                "cross_sections_assigned": 0,
                "sum_generator_weights_resolved": 0,
                "qcd_stitching_resolved": False,
            },
            "controls": {
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "thresholds_selected": 0,
                "models_trained": 0,
            },
            "next_gate": (
                "recover_generator_artifacts_and_sumweights_for_each_campaign"
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        checksum_path = output_dir / "SHA256SUMS"
        files = sorted(
            path
            for path in output_dir.iterdir()
            if path.is_file() and path != checksum_path
        )
        checksum_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in files
            )
            + "\n",
            encoding="utf-8",
        )

        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("SOURCE_REGISTRY_630_MEMBERS_PASS")
        print("PROCESS_INVENTORY_25_LABELS_PASS")
        print("NORMALIZATION_REGISTRY_FAIL_CLOSED_PASS")
        print("QCD_OVERLAP_RISK_RECORDED_PASS")
        print("NO_CROSS_SECTIONS_ASSIGNED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("HH4B_PHYSICAL_NORMALIZATION_PROVENANCE_INVENTORY_PASS")
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
