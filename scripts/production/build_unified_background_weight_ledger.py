#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TRUE_VALUES = {
    "true",
    "1",
    "yes",
}

FALSE_VALUES = {
    "false",
    "0",
    "no",
    "",
}


def parse_bool(
    value: object,
    field: str,
) -> bool:
    normalized = str(
        value
    ).strip().lower()

    if normalized in TRUE_VALUES:
        return True

    if normalized in FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid Boolean for {field}: {value!r}"
    )


def optional_float(
    value: object,
) -> float | None:
    text = str(
        value
    ).strip()

    if not text:
        return None

    return float(text)


def is_qa_record(
    campaign: str,
    target_tag: str,
) -> bool:
    text = (
        campaign
        + " "
        + target_tag
    ).lower()

    return any(
        token in text
        for token in (
            "smoke",
            "canary",
            "qa_only",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory",
        required=True,
    )

    parser.add_argument(
        "--contract",
        required=True,
    )

    parser.add_argument(
        "--output-tsv",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--lumi-fb",
        required=True,
        type=float,
    )

    args = parser.parse_args()

    inventory_path = Path(
        args.inventory
    ).resolve()

    contract_path = Path(
        args.contract
    ).resolve()

    output_tsv = Path(
        args.output_tsv
    ).resolve()

    output_json = Path(
        args.output_json
    ).resolve()

    if not inventory_path.is_file():
        raise SystemExit(
            f"ERROR: missing inventory: {inventory_path}"
        )

    if not contract_path.is_file():
        raise SystemExit(
            f"ERROR: missing contract: {contract_path}"
        )

    if args.lumi_fb <= 0:
        raise SystemExit(
            "ERROR: luminosity must be positive"
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

    with contract_path.open(
        newline=""
    ) as handle:
        contract_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    contract_by_family = {
        row["family"]: row
        for row in contract_rows
    }

    if len(contract_by_family) != len(
        contract_rows
    ):
        raise SystemExit(
            "ERROR: duplicate family in contract"
        )

    inventory_families = {
        row["family"]
        for row in inventory_rows
    }

    missing_policies = sorted(
        inventory_families
        - set(contract_by_family)
    )

    if missing_policies:
        raise SystemExit(
            "ERROR: contract does not cover: "
            + ", ".join(missing_policies)
        )

    prepared = []

    for source in inventory_rows:
        family = source["family"]
        policy = contract_by_family[
            family
        ]

        campaign = source.get(
            "campaign",
            "",
        )

        target_tag = source.get(
            "target_tag",
            "",
        )

        expected_events = int(
            source.get(
                "events",
                0,
            )
            or 0
        )

        lhe_events = int(
            source.get(
                "lhe_events",
                0,
            )
            or 0
        )

        hepmc_events = int(
            source.get(
                "hepmc_events",
                0,
            )
            or 0
        )

        root_events = int(
            source.get(
                "root_events",
                0,
            )
            or 0
        )

        candidate_rows = int(
            source.get(
                "candidate_rows",
                0,
            )
            or 0
        )

        exit_status = int(
            source.get(
                "exit_status",
                -1,
            )
            or -1
        )

        successful = (
            source.get("stage")
            == "complete_copied_and_verified"
            and exit_status == 0
            and expected_events > 0
            and lhe_events == expected_events
            and hepmc_events == expected_events
            and root_events == expected_events
        )

        qa_only = is_qa_record(
            campaign,
            target_tag,
        )

        policy_physics = parse_bool(
            policy[
                "physics_yield_authorized"
            ],
            "physics_yield_authorized",
        )

        policy_training = parse_bool(
            policy[
                "training_authorized"
            ],
            "training_authorized",
        )

        policy_count = parse_bool(
            policy[
                "count_toward_5m"
            ],
            "count_toward_5m",
        )

        original_count = parse_bool(
            source.get(
                "count_toward_5m",
                "False",
            ),
            "source count_toward_5m",
        )

        physics_authorized = (
            successful
            and not qa_only
            and policy_physics
        )

        training_authorized = (
            successful
            and not qa_only
            and policy_training
        )

        count_toward_5m = (
            successful
            and not qa_only
            and policy_count
            and original_count
        )

        generator_cross_section_pb = (
            optional_float(
                source.get(
                    "generator_xsec_pb",
                    "",
                )
            )
        )

        physics_cross_section_pb = (
            optional_float(
                policy.get(
                    "physics_cross_section_pb",
                    "",
                )
            )
        )

        branching_fraction = optional_float(
            policy.get(
                "branching_fraction",
                "",
            )
        )

        filter_efficiency = optional_float(
            policy.get(
                "filter_efficiency",
                "",
            )
        )

        prepared.append({
            **source,
            "successful": successful,
            "qa_only": qa_only,
            "physics_yield_authorized": (
                physics_authorized
            ),
            "training_authorized": (
                training_authorized
            ),
            "count_toward_5m": (
                count_toward_5m
            ),
            "overlap_group": policy[
                "overlap_group"
            ],
            "normalization_group": policy[
                "normalization_group"
            ],
            "normalization_rule": policy[
                "normalization_rule"
            ],
            "generator_cross_section_pb": (
                generator_cross_section_pb
            ),
            "physics_cross_section_pb": (
                physics_cross_section_pb
            ),
            "branching_fraction": (
                branching_fraction
            ),
            "filter_efficiency": (
                filter_efficiency
            ),
            "generated_events_this_shard": (
                root_events
            ),
            "candidate_rows_integer": (
                candidate_rows
            ),
            "policy_status": policy[
                "policy_status"
            ],
        })

    denominator_by_group_split = defaultdict(
        int
    )

    denominator_by_group_all = defaultdict(
        int
    )

    candidates_by_family_split = defaultdict(
        int
    )

    for row in prepared:
        if (
            row["successful"]
            and not row["qa_only"]
        ):
            group = row[
                "normalization_group"
            ]

            split = row.get(
                "dataset_split",
                "",
            )

            denominator_by_group_split[
                (group, split)
            ] += row[
                "generated_events_this_shard"
            ]

            denominator_by_group_all[
                group
            ] += row[
                "generated_events_this_shard"
            ]

        if row["training_authorized"]:
            candidates_by_family_split[
                (
                    row["family"],
                    row.get(
                        "dataset_split",
                        "",
                    ),
                )
            ] += row[
                "candidate_rows_integer"
            ]

    total_candidates_by_split = defaultdict(
        int
    )

    families_with_candidates_by_split = defaultdict(
        set
    )

    for (
        family,
        split,
    ), candidates in candidates_by_family_split.items():
        if candidates <= 0:
            continue

        total_candidates_by_split[
            split
        ] += candidates

        families_with_candidates_by_split[
            split
        ].add(family)

    luminosity_pb_inverse = (
        args.lumi_fb
        * 1000.0
    )

    output_rows = []

    for row in prepared:
        split = row.get(
            "dataset_split",
            "",
        )

        group = row[
            "normalization_group"
        ]

        generated_denominator = (
            denominator_by_group_split[
                (group, split)
            ]
        )

        generated_denominator_all_splits = (
            denominator_by_group_all[
                group
            ]
        )

        generator_xsec = row[
            "generator_cross_section_pb"
        ]

        w_generator = ""

        if (
            generator_xsec is not None
            and generated_denominator > 0
        ):
            w_generator = (
                generator_xsec
                / generated_denominator
            )

        w_phys = ""

        if row[
            "physics_yield_authorized"
        ]:
            required_values = {
                "physics_cross_section_pb": row[
                    "physics_cross_section_pb"
                ],
                "branching_fraction": row[
                    "branching_fraction"
                ],
                "filter_efficiency": row[
                    "filter_efficiency"
                ],
            }

            missing = [
                key
                for key, value
                in required_values.items()
                if value is None
            ]

            if missing:
                raise SystemExit(
                    "ERROR: authorized physical yield "
                    f"for {row['family']} lacks: "
                    + ", ".join(missing)
                )

            if generated_denominator <= 0:
                raise SystemExit(
                    "ERROR: nonpositive generated "
                    f"denominator for {row['family']}"
                )

            w_phys = (
                row["physics_cross_section_pb"]
                * row["branching_fraction"]
                * row["filter_efficiency"]
                * luminosity_pb_inverse
                / generated_denominator
            )

        family_candidate_total = (
            candidates_by_family_split[
                (
                    row["family"],
                    split,
                )
            ]
        )

        split_candidate_total = (
            total_candidates_by_split[
                split
            ]
        )

        number_of_families = len(
            families_with_candidates_by_split[
                split
            ]
        )

        w_train = ""

        if (
            row["training_authorized"]
            and family_candidate_total > 0
            and split_candidate_total > 0
            and number_of_families > 0
        ):
            w_train = (
                split_candidate_total
                / (
                    number_of_families
                    * family_candidate_total
                )
            )

        output_rows.append({
            "campaign": row.get(
                "campaign",
                "",
            ),
            "family": row["family"],
            "target_tag": row.get(
                "target_tag",
                "",
            ),
            "dataset_split": split,
            "dataset_role": row.get(
                "dataset_role",
                "",
            ),
            "accounting_class": row.get(
                "accounting_class",
                "",
            ),
            "successful": row[
                "successful"
            ],
            "qa_only": row[
                "qa_only"
            ],
            "physics_yield_authorized": row[
                "physics_yield_authorized"
            ],
            "training_authorized": row[
                "training_authorized"
            ],
            "count_toward_5m": row[
                "count_toward_5m"
            ],
            "overlap_group": row[
                "overlap_group"
            ],
            "normalization_group": group,
            "normalization_rule": row[
                "normalization_rule"
            ],
            "generator_cross_section_pb": (
                ""
                if generator_xsec is None
                else generator_xsec
            ),
            "physics_cross_section_pb": (
                ""
                if row[
                    "physics_cross_section_pb"
                ] is None
                else row[
                    "physics_cross_section_pb"
                ]
            ),
            "branching_fraction": (
                ""
                if row[
                    "branching_fraction"
                ] is None
                else row[
                    "branching_fraction"
                ]
            ),
            "filter_efficiency": (
                ""
                if row[
                    "filter_efficiency"
                ] is None
                else row[
                    "filter_efficiency"
                ]
            ),
            "generated_events_this_shard": row[
                "generated_events_this_shard"
            ],
            "generated_denominator": (
                generated_denominator
            ),
            "generated_denominator_all_splits": (
                generated_denominator_all_splits
            ),
            "candidate_rows": row[
                "candidate_rows_integer"
            ],
            "w_generator_pb_per_event": (
                w_generator
            ),
            "w_phys": w_phys,
            "w_train": w_train,
            "weight_luminosity_fb": (
                args.lumi_fb
            ),
            "policy_status": row[
                "policy_status"
            ],
            "remote_bundle": row.get(
                "remote_bundle",
                "",
            ),
            "receipt": row.get(
                "receipt",
                "",
            ),
        })

    fieldnames = list(
        output_rows[0]
    )

    output_tsv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_tsv.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    counted_events = sum(
        int(row["generated_events_this_shard"])
        for row in output_rows
        if row["count_toward_5m"]
    )

    training_events = sum(
        int(row["generated_events_this_shard"])
        for row in output_rows
        if row["training_authorized"]
    )

    training_candidates = sum(
        int(row["candidate_rows"])
        for row in output_rows
        if row["training_authorized"]
    )

    physics_rows = sum(
        bool(row["physics_yield_authorized"])
        for row in output_rows
    )

    review_families = sorted({
        row["family"]
        for row in output_rows
        if row["policy_status"]
        == "review_required"
    })

    summary = {
        "schema_version": 1,
        "lumi_fb": args.lumi_fb,
        "records": len(output_rows),
        "counted_successful_events": (
            counted_events
        ),
        "training_authorized_events": (
            training_events
        ),
        "training_authorized_candidates": (
            training_candidates
        ),
        "physics_authorized_records": (
            physics_rows
        ),
        "review_required_families": (
            review_families
        ),
        "weight_definitions": {
            "w_generator_pb_per_event": (
                "generator_cross_section_pb "
                "/ generated_denominator"
            ),
            "w_phys": (
                "physics_cross_section_pb "
                "* branching_fraction "
                "* filter_efficiency "
                "* lumi_pb_inverse "
                "/ generated_denominator"
            ),
            "w_train": (
                "split_candidate_total "
                "/ (authorized_family_count "
                "* family_split_candidate_total)"
            ),
            "generated_denominator": (
                "successful non-QA events in the same "
                "normalization_group and dataset_split"
            ),
        },
        "ledger_tsv": str(
            output_tsv
        ),
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"records={len(output_rows)}"
    )

    print(
        "counted_successful_events="
        f"{counted_events}"
    )

    print(
        "training_authorized_events="
        f"{training_events}"
    )

    print(
        "training_authorized_candidates="
        f"{training_candidates}"
    )

    print(
        "physics_authorized_records="
        f"{physics_rows}"
    )

    print(
        "review_required_families="
        f"{len(review_families)}"
    )

    print(f"ledger={output_tsv}")
    print(f"summary={output_json}")
    print(
        "UNIFIED_BACKGROUND_PHASE1_WEIGHT_LEDGER_VALID"
    )


if __name__ == "__main__":
    main()
