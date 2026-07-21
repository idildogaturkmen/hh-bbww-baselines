#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICY_FIELDS = [
    "family",
    "physics_yield_authorized",
    "training_authorized",
    "count_toward_5m",
    "overlap_group",
    "normalization_group",
    "normalization_rule",
    "physics_cross_section_pb",
    "branching_fraction",
    "filter_efficiency",
    "policy_status",
    "review_note",
]


KNOWN_POLICIES = {
    "tth_hbb": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "True",
        "overlap_group": "ttH",
        "normalization_group": "tth_hbb",
        "normalization_rule": (
            "decay_chain_generator_xsec_diagnostic;"
            "reference_xsec_times_Hbb_BR_pending"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "1.0",
        "filter_efficiency": "1.0",
        "policy_status": "counted_generation_normalization_pending",
        "review_note": (
            "MG5 decay-chain cross section is diagnostic. "
            "Do not multiply Hbb branching fraction again unless "
            "physics_cross_section_pb is an inclusive ttH cross section."
        ),
    },
    "ttz_zbb": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "True",
        "overlap_group": "ttZ",
        "normalization_group": "ttz_zbb",
        "normalization_rule": (
            "decay_chain_generator_xsec_diagnostic;"
            "reference_xsec_times_Zbb_BR_pending"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "1.0",
        "filter_efficiency": "1.0",
        "policy_status": "counted_generation_normalization_pending",
        "review_note": (
            "MG5 decay-chain cross section is diagnostic. "
            "Freeze the ttZ normalization convention before yields."
        ),
    },
    "tttt": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "True",
        "overlap_group": "tttt",
        "normalization_group": "tttt",
        "normalization_rule": (
            "inclusive_generator_xsec_diagnostic;"
            "higher_order_reference_xsec_pending"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "1.0",
        "filter_efficiency": "1.0",
        "policy_status": "counted_generation_normalization_pending",
        "review_note": (
            "Use an approved higher-order four-top cross section "
            "before authorizing physical yields."
        ),
    },
    "vbf_hbb": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "True",
        "overlap_group": "vbf_hbb",
        "normalization_group": "vbf_hbb",
        "normalization_rule": (
            "decay_chain_generator_xsec_diagnostic;"
            "reference_xsec_times_Hbb_BR_pending"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "1.0",
        "filter_efficiency": "1.0",
        "policy_status": "counted_generation_normalization_pending",
        "review_note": (
            "Inclusive VBF Hbb reference is training-authorized. "
            "Any heavy-flavor enrichment must remain a separate stratum."
        ),
    },
    "ttbar_inclusive": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "False",
        "overlap_group": "ttbar",
        "normalization_group": "ttbar_inclusive",
        "normalization_rule": (
            "higher_order_reference_xsec_pending"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "1.0",
        "filter_efficiency": "1.0",
        "policy_status": "valid_legacy_promotion_review_required",
        "review_note": (
            "Valid legacy production. Do not count toward 5M "
            "until lineage and promotion are explicitly approved."
        ),
    },
    "zbbbb": {
        "physics_yield_authorized": "False",
        "training_authorized": "True",
        "count_toward_5m": "False",
        "overlap_group": "zjets_hf",
        "normalization_group": "zbbbb_ml_enrichment",
        "normalization_rule": (
            "training_only_until_overlap_and_stitching_frozen"
        ),
        "physics_cross_section_pb": "",
        "branching_fraction": "",
        "filter_efficiency": "",
        "policy_status": "ml_enrichment_training_only",
        "review_note": (
            "Do not use for physical yields or 5M physical accounting "
            "until overlap with inclusive Z+jets is resolved."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    inventory_path = Path(
        args.inventory
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    if not inventory_path.is_file():
        raise SystemExit(
            f"ERROR: missing inventory: {inventory_path}"
        )

    if output_path.exists():
        raise SystemExit(
            f"ERROR: refusing to overwrite: {output_path}"
        )

    with inventory_path.open(
        newline=""
    ) as handle:
        inventory_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    families = sorted({
        row["family"]
        for row in inventory_rows
        if row.get("family")
    })

    if not families:
        raise SystemExit(
            "ERROR: inventory contains no families"
        )

    records = []

    for family in families:
        if family in KNOWN_POLICIES:
            policy = {
                **KNOWN_POLICIES[family],
            }
        else:
            policy = {
                "physics_yield_authorized": "False",
                "training_authorized": "False",
                "count_toward_5m": "False",
                "overlap_group": "REVIEW_REQUIRED",
                "normalization_group": family,
                "normalization_rule": "REVIEW_REQUIRED",
                "physics_cross_section_pb": "",
                "branching_fraction": "",
                "filter_efficiency": "",
                "policy_status": "review_required",
                "review_note": (
                    "No automatic promotion. Review lineage, "
                    "overlap, normalization, and training role."
                ),
            }

        records.append({
            "family": family,
            **policy,
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=POLICY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(records)

    ready_training = sum(
        row["training_authorized"] == "True"
        for row in records
    )

    counted = sum(
        row["count_toward_5m"] == "True"
        for row in records
    )

    review_required = sum(
        row["policy_status"] == "review_required"
        for row in records
    )

    print(f"families={len(records)}")
    print(f"training_authorized_families={ready_training}")
    print(f"default_counted_families={counted}")
    print(f"review_required_families={review_required}")
    print(f"contract={output_path}")
    print("UNIFIED_BACKGROUND_PHASE1_CONTRACT_CREATED")


if __name__ == "__main__":
    main()
