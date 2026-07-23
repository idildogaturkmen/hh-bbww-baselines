#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FAMILY = "vbf_hh4b_sm"

PILOT_CAMPAIGN = (
    "vbf_hh4b_sm_pilot10k_20260722_v1"
)

SCALEOUT_CAMPAIGN = (
    "vbf_hh4b_sm_scaleout90k_20260722_v1"
)

PAYLOAD_SHA256 = (
    "adfdc4ac14341e195c31fe6dcd9a2ef1a4dd1e7c21f7bd87ab1e86dda305726a"
)

PILOT_EXPECTED = {
    "target_tag":
        "vbf_hh4b_sm_run2_frozen_v2_pilot_shard9301_10k",
    "shard_id": 9301,
    "seed": 983001,
    "pythia_seed": 36371054,
    "dataset_split": "validation",
    "dataset_role": "pilot_validation",
    "events": 10000,
}

SCALEOUT_EXPECTED: list[dict[str, Any]] = [
    {
        "target_tag": (
            "vbf_hh4b_sm_run2_frozen_v2_"
            f"train_shard{shard_id}_10k"
        ),
        "shard_id": shard_id,
        "seed": 984000 + offset,
        "dataset_split": "train",
        "dataset_role": "canonical_train",
        "events": 10000,
    }
    for offset, shard_id in enumerate(
        range(9401, 9409),
        start=1,
    )
]

SCALEOUT_EXPECTED.append({
    "target_tag": (
        "vbf_hh4b_sm_run2_frozen_v2_"
        "validation_shard9409_10k"
    ),
    "shard_id": 9409,
    "seed": 984009,
    "dataset_split": "validation",
    "dataset_role": "canonical_validation",
    "events": 10000,
})


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
        / "vbf_hh4b_sm_pilot10k_final_audit_20260722_v1"
        / "vbf_hh4b_pilot10k_final_audit.json"
    )

    scaleout_audit_path = (
        repo
        / "outputs/agent_runs"
        / "vbf_hh4b_sm_scaleout90k_final_audit_20260723_v2"
        / "vbf_hh4b_scaleout90k_final_audit.json"
    )

    scaleout_receipts_path = (
        repo
        / "outputs/agent_runs"
        / "vbf_hh4b_sm_scaleout90k_audit_input_inspection_20260723_v1"
        / "vbf_scaleout90k_receipts.json"
    )

    for path in (
        pilot_audit_path,
        scaleout_audit_path,
        scaleout_receipts_path,
    ):
        require(
            path.is_file(),
            f"missing required audit input {path}",
        )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    scaleout_audit = json.loads(
        scaleout_audit_path.read_text()
    )

    scaleout_receipts = json.loads(
        scaleout_receipts_path.read_text()
    )

    required_pilot_audit = {
        "status": "pass",
        "vbf_hh4b_pilot10k_final_audit_valid":
            True,
        "canonical_signal_target_events":
            100000,
        "validated_pilot_events":
            10000,
        "physics_yield_authorized":
            False,
    }

    for key, expected in (
        required_pilot_audit.items()
    ):
        observed = pilot_audit.get(key)

        require(
            observed == expected,
            (
                f"pilot audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_scaleout_audit = {
        "status": "pass",
        "vbf_hh4b_scaleout90k_final_audit_valid":
            True,
        "validated_scaleout_events":
            90000,
        "validated_scaleout_train_events":
            80000,
        "validated_scaleout_validation_events":
            10000,
        "validated_scaleout_hh4b_truth_events":
            90000,
        "validated_pilot_events":
            10000,
        "canonical_signal_events_after_combination":
            100000,
        "canonical_signal_train_events_after_combination":
            80000,
        "canonical_signal_validation_events_after_combination":
            20000,
        "vbf_hh4b_canonical100k_registry_audit_authorized":
            True,
        "physics_yield_authorized":
            False,
    }

    for key, expected in (
        required_scaleout_audit.items()
    ):
        observed = scaleout_audit.get(key)

        require(
            observed == expected,
            (
                f"scale-out audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    pilot_rows = list(
        (
            store
            / "condor_return"
            / PILOT_CAMPAIGN
            / "receipts"
        ).glob("*.json")
    )

    require(
        len(pilot_rows) == 1,
        (
            "expected exactly one VBF pilot receipt, "
            f"found {len(pilot_rows)}"
        ),
    )

    pilot_receipt_path = pilot_rows[0]

    pilot_receipt = json.loads(
        pilot_receipt_path.read_text()
    )

    required_pilot_receipt = {
        "schema_version": 1,
        "campaign": PILOT_CAMPAIGN,
        "family": FAMILY,
        "target_tag":
            PILOT_EXPECTED["target_tag"],
        "shard_id":
            PILOT_EXPECTED["shard_id"],
        "n_events": 10000,
        "seed": PILOT_EXPECTED["seed"],
        "pythia_seed":
            PILOT_EXPECTED["pythia_seed"],
        "dataset_split": "validation",
        "dataset_role": "pilot_validation",
        "stage":
            "complete_copied_and_verified",
        "payload_sha256":
            PAYLOAD_SHA256,
        "expected_payload_sha256":
            PAYLOAD_SHA256,
        "lhe_events": 10000,
        "hepmc_events": 10000,
        "root_events": 10000,
        "exit_status": 0,
    }

    for key, expected in (
        required_pilot_receipt.items()
    ):
        observed = pilot_receipt.get(key)

        require(
            observed == expected,
            (
                f"pilot receipt mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    pilot_summary_row = pilot_audit["pilot"]

    require(
        int(
            pilot_summary_row["events"]
        ) == 10000,
        "pilot audit event total is not 10k",
    )

    require(
        int(
            pilot_summary_row[
                "forced_hh4b_truth_events"
            ]
        ) == 10000,
        "pilot HH4b truth total is not 10k",
    )

    require(
        pilot_summary_row[
            "bundle_sha256"
        ]
        == pilot_receipt[
            "bundle_sha256"
        ],
        "pilot audit/receipt bundle SHA mismatch",
    )

    require(
        pilot_summary_row[
            "bundle_adler32"
        ]
        == str(
            pilot_receipt[
                "bundle_adler32"
            ]
        ).lower(),
        "pilot audit/receipt Adler mismatch",
    )

    require(
        close_float(
            float(
                pilot_summary_row[
                    "generator_xsec_pb"
                ]
            ),
            float(
                pilot_receipt[
                    "generator_xsec_pb"
                ]
            ),
        ),
        "pilot audit/receipt xsec mismatch",
    )

    require(
        len(scaleout_receipts) == 9,
        "expected nine scale-out receipts",
    )

    scaleout_receipts_by_tag = {
        row["target_tag"]: row
        for row in scaleout_receipts
    }

    scaleout_audit_rows = {
        row["target_tag"]: row
        for row in scaleout_audit[
            "scaleout"
        ]
    }

    expected_scaleout_by_tag = {
        row["target_tag"]: row
        for row in SCALEOUT_EXPECTED
    }

    require(
        set(scaleout_receipts_by_tag)
        == set(expected_scaleout_by_tag),
        "scale-out receipt membership mismatch",
    )

    require(
        set(scaleout_audit_rows)
        == set(expected_scaleout_by_tag),
        "scale-out audit membership mismatch",
    )

    registry_rows: list[
        dict[str, Any]
    ] = []

    registry_rows.append({
        "family": FAMILY,
        "source_campaign":
            PILOT_CAMPAIGN,
        "source_final_audit":
            str(pilot_audit_path),
        "source_receipt":
            str(pilot_receipt_path),
        "target_tag":
            PILOT_EXPECTED["target_tag"],
        "shard_id":
            PILOT_EXPECTED["shard_id"],
        "seed":
            PILOT_EXPECTED["seed"],
        "pythia_seed":
            PILOT_EXPECTED["pythia_seed"],
        "dataset_split":
            "validation",
        "dataset_role":
            "pilot_validation",
        "events": 10000,
        "candidate_rows":
            int(
                pilot_receipt[
                    "candidate_rows"
                ]
            ),
        "generator_xsec_pb":
            float(
                pilot_receipt[
                    "generator_xsec_pb"
                ]
            ),
        "remote_bundle":
            pilot_receipt[
                "remote_bundle"
            ],
        "bundle_sha256":
            pilot_receipt[
                "bundle_sha256"
            ],
        "bundle_adler32":
            str(
                pilot_receipt[
                    "bundle_adler32"
                ]
            ).lower(),
        "payload_sha256":
            PAYLOAD_SHA256,
        "forced_hh4b_truth_events":
            10000,
        "count_toward_signal_target":
            True,
        "final_audit_valid":
            True,
        "physics_yield_authorized":
            False,
    })

    for expected in sorted(
        SCALEOUT_EXPECTED,
        key=lambda row: int(
            row["shard_id"]
        ),
    ):
        target_tag = expected[
            "target_tag"
        ]

        receipt = (
            scaleout_receipts_by_tag[
                target_tag
            ]
        )

        audit_row = (
            scaleout_audit_rows[
                target_tag
            ]
        )

        seed = int(
            expected["seed"]
        )

        pythia_seed = (
            seed * 37 + 17
        ) % 900_000_000

        if pythia_seed == 0:
            pythia_seed = 1

        required_receipt = {
            "campaign":
                SCALEOUT_CAMPAIGN,
            "family": FAMILY,
            "target_tag":
                target_tag,
            "shard_id":
                expected["shard_id"],
            "n_events": 10000,
            "seed": seed,
            "pythia_seed":
                pythia_seed,
            "dataset_split":
                expected[
                    "dataset_split"
                ],
            "dataset_role":
                expected[
                    "dataset_role"
                ],
            "stage":
                "complete_copied_and_verified",
            "payload_sha256":
                PAYLOAD_SHA256,
            "lhe_events": 10000,
            "hepmc_events": 10000,
            "root_events": 10000,
            "exit_status": 0,
        }

        for key, expected_value in (
            required_receipt.items()
        ):
            observed = receipt.get(key)

            require(
                observed == expected_value,
                (
                    f"{target_tag}: receipt "
                    f"mismatch for {key}: "
                    f"{observed!r} != "
                    f"{expected_value!r}"
                ),
            )

        require(
            int(
                audit_row["events"]
            ) == 10000,
            f"{target_tag}: audit event mismatch",
        )

        require(
            int(
                audit_row[
                    "forced_hh4b_truth_events"
                ]
            ) == 10000,
            (
                f"{target_tag}: HH4b truth "
                "audit mismatch"
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
                f"{target_tag}: audit/receipt "
                "bundle SHA mismatch"
            ),
        )

        require(
            audit_row[
                "bundle_adler32"
            ]
            == str(
                receipt[
                    "bundle_adler32"
                ]
            ).lower(),
            (
                f"{target_tag}: audit/receipt "
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
                f"{target_tag}: audit/receipt "
                "xsec mismatch"
            ),
        )

        registry_rows.append({
            "family": FAMILY,
            "source_campaign":
                SCALEOUT_CAMPAIGN,
            "source_final_audit":
                str(scaleout_audit_path),
            "source_receipt":
                str(
                    receipt[
                        "receipt_path"
                    ]
                ),
            "target_tag":
                target_tag,
            "shard_id":
                expected["shard_id"],
            "seed": seed,
            "pythia_seed":
                pythia_seed,
            "dataset_split":
                expected[
                    "dataset_split"
                ],
            "dataset_role":
                expected[
                    "dataset_role"
                ],
            "events": 10000,
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
            "payload_sha256":
                PAYLOAD_SHA256,
            "forced_hh4b_truth_events":
                10000,
            "count_toward_signal_target":
                True,
            "final_audit_valid":
                True,
            "physics_yield_authorized":
                False,
        })

    require(
        len(registry_rows) == 10,
        "canonical VBF registry does not have 10 shards",
    )

    for field in (
        "target_tag",
        "shard_id",
        "seed",
        "pythia_seed",
        "remote_bundle",
        "bundle_sha256",
    ):
        values = {
            row[field]
            for row in registry_rows
        }

        require(
            len(values) == 10,
            (
                "canonical registry contains "
                f"duplicate {field}"
            ),
        )

    total_events = sum(
        int(row["events"])
        for row in registry_rows
    )

    train_events = sum(
        int(row["events"])
        for row in registry_rows
        if row["dataset_split"] == "train"
    )

    validation_events = sum(
        int(row["events"])
        for row in registry_rows
        if row["dataset_split"] == "validation"
    )

    truth_events = sum(
        int(
            row[
                "forced_hh4b_truth_events"
            ]
        )
        for row in registry_rows
    )

    candidate_rows = sum(
        int(row["candidate_rows"])
        for row in registry_rows
    )

    require(
        total_events == 100000,
        "canonical VBF total is not 100k",
    )

    require(
        train_events == 80000,
        "canonical VBF train total is not 80k",
    )

    require(
        validation_events == 20000,
        (
            "canonical VBF validation total "
            "is not 20k"
        ),
    )

    require(
        truth_events == 100000,
        (
            "canonical VBF HH4b truth total "
            "is not 100k"
        ),
    )

    xsecs = [
        float(
            row[
                "generator_xsec_pb"
            ]
        )
        for row in registry_rows
    ]

    outdir.mkdir(parents=True)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "family": FAMILY,
        "registry": registry_rows,
        "canonical_vbf_hh4b_membership_frozen":
            True,
        "vbf_hh4b_canonical100k_registry_valid":
            True,
        "canonical_signal_shards":
            10,
        "canonical_signal_events":
            100000,
        "canonical_signal_train_events":
            80000,
        "canonical_signal_validation_events":
            20000,
        "canonical_signal_hh4b_truth_events":
            100000,
        "canonical_signal_candidate_rows":
            candidate_rows,
        "generator_xsec_min_pb":
            min(xsecs),
        "generator_xsec_max_pb":
            max(xsecs),
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
        / "vbf_hh4b_canonical100k_registry.json"
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
        / "vbf_hh4b_canonical100k_registry.tsv"
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
        "VBF_HH4B_CANONICAL100K_REGISTRY_VALID"
    )
    print(
        "VBF_HH4B_CANONICAL_MEMBERSHIP_FROZEN"
    )
    print(
        "VBF_HH4B_CANONICAL_TRAIN_EVENTS=80000"
    )
    print(
        "VBF_HH4B_CANONICAL_VALIDATION_EVENTS=20000"
    )
    print(
        "VBF_HH4B_CANONICAL_HH4B_TRUTH_EVENTS=100000"
    )
    print(
        "NO_VBF_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
