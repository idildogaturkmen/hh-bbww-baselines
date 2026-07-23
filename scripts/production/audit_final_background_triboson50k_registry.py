#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PILOT_CAMPAIGN = (
    "final_background_triboson_zbb_pilot10k_20260722_v1"
)

SCALEOUT_CAMPAIGN = (
    "final_background_triboson_zbb_scaleout40k_20260722_v1"
)

PILOT_AUDIT_TAG = (
    "final_background_triboson_pilot10k_final_audit_20260722_v1"
)

SCALEOUT_AUDIT_TAG = (
    "final_background_triboson_scaleout40k_final_audit_20260723_v1"
)

SCALEOUT_INSPECTION_TAG = (
    "final_background_triboson_scaleout40k_"
    "audit_input_inspection_20260723_v1"
)

ALLOCATION_TAG = (
    "final_background_triboson_zbb_pilot10k_"
    "submission_20260722_v1"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def close_float(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )


def pythia_seed(
    seed: int,
) -> int:
    value = (
        seed * 37 + 17
    ) % 900_000_000

    return value if value != 0 else 1


def make_scaleout_rows(
    *,
    family: str,
    shard_ids: list[int],
    seed_start: int,
    accepted_events: int,
    input_events: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for offset, shard_id in enumerate(
        shard_ids
    ):
        rows.append({
            "family": family,
            "target_tag": (
                f"{family}_run2_frozen_v2_"
                f"train_shard{shard_id}_"
                f"{accepted_events}"
            ),
            "shard_id": shard_id,
            "seed": seed_start + offset,
            "accepted_events":
                accepted_events,
            "input_events":
                input_events,
            "dataset_split":
                "train",
            "dataset_role":
                "canonical_train",
            "split_salt":
                "final-background-triboson-zbb-scaleout-v1",
            "source_campaign":
                SCALEOUT_CAMPAIGN,
            "source_kind":
                "scaleout",
        })

    return rows


PILOT_EXPECTED: list[dict[str, Any]] = [
    {
        "family": "wwz_zbb",
        "target_tag": (
            "wwz_zbb_run2_frozen_v2_"
            "pilot_validation_shard9601_5200"
        ),
        "shard_id": 9601,
        "seed": 996101,
        "accepted_events": 5200,
        "input_events": 46800,
        "dataset_split": "validation",
        "dataset_role": "pilot_validation",
        "split_salt":
            "final-background-triboson-zbb-pilot-v1",
        "source_campaign":
            PILOT_CAMPAIGN,
        "source_kind":
            "pilot",
    },
    {
        "family": "wzz_zbb",
        "target_tag": (
            "wzz_zbb_run2_frozen_v2_"
            "pilot_validation_shard9602_3300"
        ),
        "shard_id": 9602,
        "seed": 996102,
        "accepted_events": 3300,
        "input_events": 19800,
        "dataset_split": "validation",
        "dataset_role": "pilot_validation",
        "split_salt":
            "final-background-triboson-zbb-pilot-v1",
        "source_campaign":
            PILOT_CAMPAIGN,
        "source_kind":
            "pilot",
    },
    {
        "family": "zzz_zbb",
        "target_tag": (
            "zzz_zbb_run2_frozen_v2_"
            "pilot_validation_shard9603_1500"
        ),
        "shard_id": 9603,
        "seed": 996103,
        "accepted_events": 1500,
        "input_events": 7500,
        "dataset_split": "validation",
        "dataset_role": "pilot_validation",
        "split_salt":
            "final-background-triboson-zbb-pilot-v1",
        "source_campaign":
            PILOT_CAMPAIGN,
        "source_kind":
            "pilot",
    },
]

SCALEOUT_EXPECTED: list[dict[str, Any]] = []

SCALEOUT_EXPECTED.extend(
    make_scaleout_rows(
        family="wwz_zbb",
        shard_ids=[
            9901,
            9902,
            9903,
            9904,
        ],
        seed_start=997101,
        accepted_events=5200,
        input_events=46800,
    )
)

SCALEOUT_EXPECTED.extend(
    make_scaleout_rows(
        family="wzz_zbb",
        shard_ids=[
            9911,
            9912,
            9913,
            9914,
        ],
        seed_start=997201,
        accepted_events=3300,
        input_events=19800,
    )
)

SCALEOUT_EXPECTED.extend(
    make_scaleout_rows(
        family="zzz_zbb",
        shard_ids=[
            9921,
            9922,
            9923,
            9924,
        ],
        seed_start=997301,
        accepted_events=1500,
        input_events=7500,
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--store",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()
    outdir = arguments.outdir.resolve()

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / PILOT_AUDIT_TAG
        / "triboson_pilot10k_final_audit.json"
    )

    scaleout_audit_path = (
        repo
        / "outputs/agent_runs"
        / SCALEOUT_AUDIT_TAG
        / "triboson_scaleout40k_final_audit.json"
    )

    scaleout_inspection_path = (
        repo
        / "outputs/agent_runs"
        / SCALEOUT_INSPECTION_TAG
        / "triboson_scaleout40k_receipts.json"
    )

    allocation_path = (
        repo
        / "outputs/agent_runs"
        / ALLOCATION_TAG
        / "triboson_final50k_allocation_freeze.json"
    )

    for path in (
        pilot_audit_path,
        scaleout_audit_path,
        scaleout_inspection_path,
        allocation_path,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    scaleout_audit = json.loads(
        scaleout_audit_path.read_text()
    )

    scaleout_inspection = json.loads(
        scaleout_inspection_path.read_text()
    )

    allocation = json.loads(
        allocation_path.read_text()
    )

    required_pilot = {
        "status": "pass",
        "triboson_pilot10k_final_audit_valid":
            True,
        "triboson_pilot_events_validated":
            10000,
        "triboson_scaleout_events_remaining":
            40000,
        "validated_background_events_before_pilot":
            4950000,
        "validated_background_events_now":
            4960000,
        "final_background_target":
            5000000,
        "physics_yield_authorized":
            False,
    }

    for key, expected in required_pilot.items():
        observed = pilot_audit.get(key)

        require(
            observed == expected,
            (
                f"pilot audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_scaleout = {
        "status": "pass",
        "triboson_scaleout40k_final_audit_valid":
            True,
        "triboson_scaleout_events_validated":
            40000,
        "triboson_scaleout_events_by_family": {
            "wwz_zbb": 20800,
            "wzz_zbb": 13200,
            "zzz_zbb": 6000,
        },
        "triboson_scaleout_direct_zbb_truth_events":
            40000,
        "malformed_process_0_excluded":
            True,
        "malformed_process_0_counted_events":
            0,
        "validated_triboson_pilot_events":
            10000,
        "potential_canonical_triboson_events":
            50000,
        "triboson_final50k_registry_audit_authorized":
            True,
        "background_events_registry_validated_before":
            4960000,
        "background_events_registry_validated_now":
            4960000,
        "potential_background_events_after_unified_registry":
            5000000,
        "physics_yield_authorized":
            False,
    }

    for key, expected in required_scaleout.items():
        observed = scaleout_audit.get(key)

        require(
            observed == expected,
            (
                f"scale-out audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    require(
        allocation.get("status") == "frozen",
        "triboson allocation is not frozen",
    )

    require(
        allocation.get("final_total_events")
        == 50000,
        "frozen allocation total is not 50k",
    )

    require(
        allocation.get("pilot_total_events")
        == 10000,
        "frozen pilot allocation is not 10k",
    )

    require(
        allocation.get(
            "normalization_use_authorized"
        )
        is False,
        (
            "allocation was incorrectly "
            "authorized for normalization"
        ),
    )

    pilot_receipt_paths = sorted(
        (
            store
            / "condor_return"
            / PILOT_CAMPAIGN
            / "receipts"
        ).glob("*.json")
    )

    require(
        len(pilot_receipt_paths) == 3,
        (
            "expected three pilot receipts, "
            f"found {len(pilot_receipt_paths)}"
        ),
    )

    pilot_receipts: list[
        dict[str, Any]
    ] = []

    for path in pilot_receipt_paths:
        row = json.loads(
            path.read_text()
        )

        row["receipt_path"] = str(
            path
        )

        pilot_receipts.append(
            row
        )

    scaleout_receipts = (
        scaleout_inspection[
            "receipts"
        ]
    )

    require(
        len(scaleout_receipts) == 12,
        "expected twelve scale-out receipts",
    )

    require(
        scaleout_inspection.get(
            "malformed_process_0_excluded"
        )
        is True,
        (
            "malformed process 0 was not "
            "excluded by inspection"
        ),
    )

    require(
        scaleout_inspection.get(
            "malformed_process_0_counted_events"
        )
        == 0,
        (
            "malformed process 0 contributed "
            "events"
        ),
    )

    pilot_receipts_by_tag = {
        row["target_tag"]: row
        for row in pilot_receipts
    }

    scaleout_receipts_by_tag = {
        row["target_tag"]: row
        for row in scaleout_receipts
    }

    pilot_audit_by_tag = {
        row["target_tag"]: row
        for row in pilot_audit[
            "pilots"
        ]
    }

    scaleout_audit_by_tag = {
        row["target_tag"]: row
        for row in scaleout_audit[
            "scaleout"
        ]
    }

    expected_pilot_tags = {
        row["target_tag"]
        for row in PILOT_EXPECTED
    }

    expected_scaleout_tags = {
        row["target_tag"]
        for row in SCALEOUT_EXPECTED
    }

    require(
        set(pilot_receipts_by_tag)
        == expected_pilot_tags,
        "pilot receipt membership mismatch",
    )

    require(
        set(pilot_audit_by_tag)
        == expected_pilot_tags,
        "pilot audit membership mismatch",
    )

    require(
        set(scaleout_receipts_by_tag)
        == expected_scaleout_tags,
        "scale-out receipt membership mismatch",
    )

    require(
        set(scaleout_audit_by_tag)
        == expected_scaleout_tags,
        "scale-out audit membership mismatch",
    )

    registry_rows: list[
        dict[str, Any]
    ] = []

    all_expected = (
        PILOT_EXPECTED
        + SCALEOUT_EXPECTED
    )

    for expected in sorted(
        all_expected,
        key=lambda row: int(
            row["shard_id"]
        ),
    ):
        source_kind = str(
            expected["source_kind"]
        )

        target_tag = str(
            expected["target_tag"]
        )

        if source_kind == "pilot":
            receipt = pilot_receipts_by_tag[
                target_tag
            ]

            audit_row = pilot_audit_by_tag[
                target_tag
            ]

            source_audit_path = (
                pilot_audit_path
            )
        else:
            receipt = scaleout_receipts_by_tag[
                target_tag
            ]

            audit_row = scaleout_audit_by_tag[
                target_tag
            ]

            source_audit_path = (
                scaleout_audit_path
            )

        seed = int(
            expected["seed"]
        )

        expected_pythia = pythia_seed(
            seed
        )

        accepted_events = int(
            expected["accepted_events"]
        )

        input_events = int(
            expected["input_events"]
        )

        required_receipt = {
            "schema_version": 1,
            "campaign":
                expected[
                    "source_campaign"
                ],
            "family":
                expected["family"],
            "target_tag":
                target_tag,
            "shard_id":
                int(
                    expected[
                        "shard_id"
                    ]
                ),
            "n_events":
                accepted_events,
            "input_lhe_events_requested":
                input_events,
            "seed": seed,
            "pythia_seed":
                expected_pythia,
            "dataset_split":
                expected[
                    "dataset_split"
                ],
            "split_assignment_unit":
                "whole_shard",
            "split_salt":
                expected["split_salt"],
            "run_card_profile":
                "preserve_template",
            "dataset_role":
                expected[
                    "dataset_role"
                ],
            "stage":
                "complete_copied_and_verified",
            "lhe_events":
                input_events,
            "accepted_events":
                accepted_events,
            "hepmc_events":
                accepted_events,
            "root_events":
                accepted_events,
            "filter_requirement":
                "at_least_one_direct_z_to_bb",
            "generator_xsec_semantics":
                "inclusive_before_zbb_filter",
            "exit_status": 0,
        }

        for key, value in (
            required_receipt.items()
        ):
            observed = receipt.get(key)

            require(
                observed == value,
                (
                    f"{target_tag}: receipt "
                    f"mismatch for {key}: "
                    f"{observed!r} != {value!r}"
                ),
            )

        processed = int(
            receipt[
                "processed_input_events"
            ]
        )

        rejected = int(
            receipt[
                "rejected_input_events"
            ]
        )

        require(
            processed
            == rejected
            + accepted_events,
            (
                f"{target_tag}: filter "
                "accounting does not close"
            ),
        )

        require(
            accepted_events
            <= processed
            <= input_events,
            (
                f"{target_tag}: invalid "
                "processed event count"
            ),
        )

        require(
            int(
                audit_row[
                    "accepted_events"
                ]
            )
            == accepted_events,
            (
                f"{target_tag}: audit "
                "accepted-event mismatch"
            ),
        )

        require(
            int(
                audit_row[
                    "direct_zbb_truth_events"
                ]
            )
            == accepted_events,
            (
                f"{target_tag}: direct-Zbb "
                "truth mismatch"
            ),
        )

        require(
            int(
                audit_row[
                    "input_lhe_events"
                ]
            )
            == input_events,
            (
                f"{target_tag}: audit input "
                "LHE mismatch"
            ),
        )

        require(
            int(
                audit_row[
                    "processed_input_events"
                ]
            )
            == processed,
            (
                f"{target_tag}: audit "
                "processed count mismatch"
            ),
        )

        require(
            int(
                audit_row[
                    "rejected_input_events"
                ]
            )
            == rejected,
            (
                f"{target_tag}: audit "
                "rejected count mismatch"
            ),
        )

        require(
            int(
                audit_row[
                    "candidate_rows"
                ]
            )
            == int(
                receipt[
                    "candidate_rows"
                ]
            ),
            (
                f"{target_tag}: candidate "
                "row mismatch"
            ),
        )

        require(
            audit_row[
                "bundle_sha256"
            ]
            == receipt[
                "bundle_sha256"
            ],
            (
                f"{target_tag}: bundle "
                "SHA mismatch"
            ),
        )

        require(
            str(
                audit_row[
                    "bundle_adler32"
                ]
            ).lower()
            == str(
                receipt[
                    "bundle_adler32"
                ]
            ).lower(),
            (
                f"{target_tag}: bundle "
                "Adler mismatch"
            ),
        )

        require(
            close_float(
                float(
                    audit_row[
                        "generator_xsec_pb"
                    ]
                ),
                float(
                    receipt[
                        "generator_xsec_pb"
                    ]
                ),
            ),
            (
                f"{target_tag}: generator "
                "cross-section mismatch"
            ),
        )

        registry_rows.append({
            "family":
                expected["family"],
            "source_kind":
                source_kind,
            "source_campaign":
                expected[
                    "source_campaign"
                ],
            "source_final_audit":
                str(
                    source_audit_path
                ),
            "source_receipt":
                str(
                    receipt[
                        "receipt_path"
                    ]
                ),
            "target_tag":
                target_tag,
            "shard_id":
                int(
                    expected[
                        "shard_id"
                    ]
                ),
            "seed": seed,
            "pythia_seed":
                expected_pythia,
            "dataset_split":
                expected[
                    "dataset_split"
                ],
            "dataset_role":
                expected[
                    "dataset_role"
                ],
            "input_lhe_events":
                input_events,
            "processed_input_events":
                processed,
            "rejected_input_events":
                rejected,
            "accepted_events":
                accepted_events,
            "direct_zbb_truth_events":
                accepted_events,
            "candidate_rows":
                int(
                    receipt[
                        "candidate_rows"
                    ]
                ),
            "generator_xsec_pb":
                float(
                    receipt[
                        "generator_xsec_pb"
                    ]
                ),
            "generator_xsec_semantics":
                "inclusive_before_zbb_filter",
            "remote_bundle":
                receipt[
                    "remote_bundle"
                ],
            "bundle_sha256":
                receipt[
                    "bundle_sha256"
                ],
            "bundle_adler32":
                str(
                    receipt[
                        "bundle_adler32"
                    ]
                ).lower(),
            "bundle_size_bytes":
                int(
                    receipt[
                        "bundle_size_bytes"
                    ]
                ),
            "payload_sha256":
                receipt[
                    "payload_sha256"
                ],
            "root_sha256":
                receipt[
                    "root_sha256"
                ],
            "event_summary_sha256":
                receipt[
                    "event_summary_sha256"
                ],
            "candidate_sha256":
                receipt[
                    "candidate_sha256"
                ],
            "filter_accounting_sha256":
                receipt[
                    "filter_accounting_sha256"
                ],
            "count_toward_background_target":
                True,
            "final_audit_valid":
                True,
            "physics_yield_authorized":
                False,
        })

    require(
        len(registry_rows) == 15,
        (
            "canonical triboson registry "
            "does not contain 15 shards"
        ),
    )

    for field in (
        "target_tag",
        "shard_id",
        "seed",
        "pythia_seed",
        "remote_bundle",
        "bundle_sha256",
    ):
        require(
            len({
                str(row[field])
                for row in registry_rows
            }) == 15,
            (
                "canonical triboson registry "
                f"contains duplicate {field}"
            ),
        )

    total_events = sum(
        int(row["accepted_events"])
        for row in registry_rows
    )

    total_truth = sum(
        int(
            row[
                "direct_zbb_truth_events"
            ]
        )
        for row in registry_rows
    )

    train_events = sum(
        int(row["accepted_events"])
        for row in registry_rows
        if row["dataset_split"] == "train"
    )

    validation_events = sum(
        int(row["accepted_events"])
        for row in registry_rows
        if (
            row["dataset_split"]
            == "validation"
        )
    )

    candidate_rows = sum(
        int(row["candidate_rows"])
        for row in registry_rows
    )

    family_totals = {
        family: sum(
            int(
                row[
                    "accepted_events"
                ]
            )
            for row in registry_rows
            if row["family"] == family
        )
        for family in (
            "wwz_zbb",
            "wzz_zbb",
            "zzz_zbb",
        )
    }

    expected_family_totals = {
        "wwz_zbb": 26000,
        "wzz_zbb": 16500,
        "zzz_zbb": 7500,
    }

    require(
        total_events == 50000,
        "canonical triboson total is not 50k",
    )

    require(
        total_truth == 50000,
        (
            "canonical triboson direct-Zbb "
            "truth total is not 50k"
        ),
    )

    require(
        train_events == 40000,
        (
            "canonical triboson training "
            "total is not 40k"
        ),
    )

    require(
        validation_events == 10000,
        (
            "canonical triboson validation "
            "total is not 10k"
        ),
    )

    require(
        family_totals
        == expected_family_totals,
        (
            "canonical family totals mismatch: "
            f"{family_totals}"
        ),
    )

    outdir.mkdir(parents=True)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "registry": registry_rows,
        "canonical_triboson50k_membership_frozen":
            True,
        "triboson_final50k_registry_valid":
            True,
        "canonical_triboson_shards":
            15,
        "canonical_triboson_events":
            50000,
        "canonical_triboson_train_events":
            40000,
        "canonical_triboson_validation_events":
            10000,
        "canonical_triboson_direct_zbb_truth_events":
            50000,
        "canonical_triboson_candidate_rows":
            candidate_rows,
        "canonical_triboson_events_by_family":
            family_totals,
        "malformed_process_0_excluded":
            True,
        "malformed_process_0_counted_events":
            0,
        "validated_background_events_before_triboson":
            4950000,
        "potential_background_events_after_unified_registry":
            5000000,
        "unified_background_5m_registry_audit_authorized":
            True,
        "pilot_final_audit":
            str(pilot_audit_path),
        "pilot_final_audit_sha256":
            sha256_file(
                pilot_audit_path
            ),
        "scaleout_final_audit":
            str(scaleout_audit_path),
        "scaleout_final_audit_sha256":
            sha256_file(
                scaleout_audit_path
            ),
        "raw_generator_xsec_normalization_authorized":
            False,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "triboson_final50k_registry.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    table_path = (
        outdir
        / "triboson_final50k_registry.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                registry_rows[0]
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            registry_rows
        )

    print(
        "TRIBOSON_FINAL50K_REGISTRY_VALID"
    )
    print(
        "TRIBOSON_FINAL50K_MEMBERSHIP_FROZEN"
    )
    print(
        "TRIBOSON_FINAL50K_EVENTS=50000"
    )
    print(
        "TRIBOSON_FINAL50K_TRAIN_EVENTS=40000"
    )
    print(
        "TRIBOSON_FINAL50K_VALIDATION_EVENTS=10000"
    )
    print(
        "TRIBOSON_FINAL50K_DIRECT_ZBB_TRUTH_EVENTS=50000"
    )
    print(
        "UNIFIED_BACKGROUND_5M_REGISTRY_AUDIT_AUTHORIZED"
    )
    print(
        "BACKGROUND_5M_NOT_YET_UNIFIED_OR_WEIGHT_AUTHORIZED"
    )
    print(
        "NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
