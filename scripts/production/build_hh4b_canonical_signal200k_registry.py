#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_GGF_MEMBERS = 100
EXPECTED_VBF_MEMBERS = 10
EXPECTED_TOTAL_MEMBERS = 110

EXPECTED_GENERATED_BY_MODE = {
    "ggf_hh4b": 100_000,
    "vbf_hh4b": 100_000,
}

EXPECTED_CANDIDATES_BY_MODE = {
    "ggf_hh4b": 6_112,
    "vbf_hh4b": 6_283,
}

EXPECTED_GENERATED_BY_SPLIT = {
    "train": 160_000,
    "validation": 37_000,
    "test": 3_000,
}

EXPECTED_CANDIDATES_BY_SPLIT = {
    "train": 9_975,
    "validation": 2_246,
    "test": 174,
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_tsv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(
            errors="replace",
        )
    )

    require(
        isinstance(document, dict),
        f"{path}: JSON is not an object",
    )

    return document


def parse_bool(
    value: object,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise RuntimeError(
        f"cannot parse Boolean value {value!r}"
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_split_flags(
    *,
    mode: str,
    member_id: str,
    split: str,
    training_authorized: bool,
    validation_authorized: bool,
    test_sealed: bool,
) -> None:
    if split == "train":
        require(
            training_authorized,
            f"{mode}/{member_id}: train is not authorized",
        )

        require(
            not validation_authorized,
            f"{mode}/{member_id}: train marked validation",
        )

        require(
            not test_sealed,
            f"{mode}/{member_id}: train marked sealed test",
        )

    elif split == "validation":
        require(
            validation_authorized,
            (
                f"{mode}/{member_id}: validation "
                "is not authorized"
            ),
        )

        require(
            not training_authorized,
            (
                f"{mode}/{member_id}: validation "
                "marked train"
            ),
        )

        require(
            not test_sealed,
            (
                f"{mode}/{member_id}: validation "
                "marked sealed test"
            ),
        )

    elif split == "test":
        require(
            mode == "ggf_hh4b",
            (
                f"{mode}/{member_id}: only ggF may "
                "provide the current test split"
            ),
        )

        require(
            test_sealed,
            f"{mode}/{member_id}: test is not sealed",
        )

        require(
            not training_authorized,
            f"{mode}/{member_id}: test marked train",
        )

        require(
            not validation_authorized,
            f"{mode}/{member_id}: test marked validation",
        )

    else:
        raise RuntimeError(
            f"{mode}/{member_id}: unknown split {split!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ggf-registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ggf-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vbf-registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vbf-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    for path in (
        args.ggf_registry,
        args.ggf_summary,
        args.vbf_registry,
        args.vbf_summary,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        not args.outdir.exists(),
        f"refusing to overwrite {args.outdir}",
    )

    ggf_rows = load_tsv(
        args.ggf_registry
    )

    vbf_rows = load_tsv(
        args.vbf_registry
    )

    ggf_summary = load_json(
        args.ggf_summary
    )

    vbf_summary = load_json(
        args.vbf_summary
    )

    require(
        ggf_summary["status"] == "pass",
        "ggF summary status is not pass",
    )

    require(
        vbf_summary["status"] == "pass",
        "VBF summary status is not pass",
    )

    require(
        len(ggf_rows) == EXPECTED_GGF_MEMBERS,
        (
            f"ggF registry has {len(ggf_rows)} "
            "members, not 100"
        ),
    )

    require(
        len(vbf_rows) == EXPECTED_VBF_MEMBERS,
        (
            f"VBF registry has {len(vbf_rows)} "
            "members, not 10"
        ),
    )

    require(
        int(ggf_summary["generated_events"])
        == 100_000,
        "ggF generated-event total is not 100000",
    )

    require(
        int(vbf_summary["generated_events"])
        == 100_000,
        "VBF generated-event total is not 100000",
    )

    require(
        int(ggf_summary["candidate_rows"])
        == 6_112,
        "ggF candidate-row total is not 6112",
    )

    require(
        int(vbf_summary["candidate_rows"])
        == 6_283,
        "VBF candidate-row total is not 6283",
    )

    require(
        ggf_summary["physics_yield_authorized"]
        is False,
        "ggF physics yield is unexpectedly authorized",
    )

    require(
        vbf_summary["physics_yield_authorized"]
        is False,
        "VBF physics yield is unexpectedly authorized",
    )

    common72_sha256 = str(
        ggf_summary[
            "common_ml_schema_sha256"
        ]
    )

    require(
        common72_sha256
        == str(
            vbf_summary[
                "candidate_schema_sha256"
            ]
        ),
        (
            "ggF and VBF common-72 schema "
            "hashes differ"
        ),
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{64}",
            common72_sha256,
        )
        is not None,
        "invalid common-72 schema SHA-256",
    )

    ggf_schema_sha256 = str(
        ggf_summary[
            "signal_schema_sha256"
        ]
    )

    require(
        int(
            ggf_summary[
                "signal_schema_columns"
            ]
        )
        == 75,
        "ggF signal schema is not 75 columns",
    )

    require(
        int(
            vbf_summary[
                "candidate_schema_columns"
            ]
        )
        == 72,
        "VBF candidate schema is not 72 columns",
    )

    require(
        vbf_summary[
            "test_members_present"
        ]
        is False,
        "VBF unexpectedly contains test members",
    )

    ggf_indices = {
        int(row["member_index"])
        for row in ggf_rows
    }

    vbf_indices = {
        int(row["member_index"])
        for row in vbf_rows
    }

    require(
        ggf_indices == set(range(100)),
        "ggF member indices are not 0 through 99",
    )

    require(
        vbf_indices == set(range(10)),
        "VBF member indices are not 0 through 9",
    )

    source_registry_sha256 = {
        "ggf_hh4b":
            sha256_file(
                args.ggf_registry
            ),
        "vbf_hh4b":
            sha256_file(
                args.vbf_registry
            ),
    }

    source_summary_sha256 = {
        "ggf_hh4b":
            sha256_file(
                args.ggf_summary
            ),
        "vbf_hh4b":
            sha256_file(
                args.vbf_summary
            ),
    }

    output_rows: list[
        dict[str, object]
    ] = []

    generated_by_mode: Counter[str] = Counter()
    candidates_by_mode: Counter[str] = Counter()

    generated_by_split: Counter[str] = Counter()
    candidates_by_split: Counter[str] = Counter()

    candidate_locations: set[str] = set()
    source_bundles: set[str] = set()

    global_member_index = 0

    for row in sorted(
        ggf_rows,
        key=lambda item: int(
            item["member_index"]
        ),
    ):
        mode = "ggf_hh4b"

        mode_member_index = int(
            row["member_index"]
        )

        member_id = (
            f"ggf_shard_"
            f"{int(row['local_shard']):03d}"
        )

        require(
            row["signal_mode"] == mode,
            f"{member_id}: incorrect signal mode",
        )

        require(
            row["status"] == "pass",
            f"{member_id}: status is not pass",
        )

        generated_events = int(
            row["generated_events"]
        )

        candidate_rows = int(
            row["candidate_rows"]
        )

        split = row[
            "immutable_split"
        ]

        require(
            generated_events == 1_000,
            (
                f"{member_id}: generated count "
                "is not 1000"
            ),
        )

        require(
            int(
                row[
                    "candidate_schema_columns"
                ]
            )
            == 75,
            (
                f"{member_id}: candidate schema "
                "is not 75 columns"
            ),
        )

        require(
            row[
                "candidate_schema_sha256"
            ]
            == ggf_schema_sha256,
            (
                f"{member_id}: ggF schema "
                "SHA-256 mismatch"
            ),
        )

        require(
            row[
                "common72_schema_sha256"
            ]
            == common72_sha256,
            (
                f"{member_id}: common-72 "
                "schema mismatch"
            ),
        )

        candidate_path = Path(
            row["candidate_parquet"]
        )

        require(
            candidate_path.is_file(),
            (
                f"{member_id}: missing local "
                f"candidate {candidate_path}"
            ),
        )

        candidate_sha256 = sha256_file(
            candidate_path
        )

        require(
            candidate_sha256
            == row[
                "candidate_parquet_sha256"
            ],
            (
                f"{member_id}: candidate "
                "SHA-256 mismatch"
            ),
        )

        training_authorized = parse_bool(
            row["training_authorized"]
        )

        validation_authorized = parse_bool(
            row["validation_authorized"]
        )

        test_sealed = parse_bool(
            row["test_sealed"]
        )

        physics_yield_authorized = parse_bool(
            row[
                "physics_yield_authorized"
            ]
        )

        require(
            not physics_yield_authorized,
            (
                f"{member_id}: physics yield "
                "unexpectedly authorized"
            ),
        )

        validate_split_flags(
            mode=mode,
            member_id=member_id,
            split=split,
            training_authorized=
                training_authorized,
            validation_authorized=
                validation_authorized,
            test_sealed=test_sealed,
        )

        candidate_location = str(
            candidate_path
        )

        source_bundle = row[
            "remote_bundle"
        ]

        require(
            candidate_location
            not in candidate_locations,
            (
                f"{member_id}: duplicate "
                "candidate location"
            ),
        )

        require(
            source_bundle
            not in source_bundles,
            (
                f"{member_id}: duplicate "
                "source bundle"
            ),
        )

        candidate_locations.add(
            candidate_location
        )

        source_bundles.add(
            source_bundle
        )

        generated_by_mode[
            mode
        ] += generated_events

        candidates_by_mode[
            mode
        ] += candidate_rows

        generated_by_split[
            split
        ] += generated_events

        candidates_by_split[
            split
        ] += candidate_rows

        output_rows.append({
            "global_member_index":
                global_member_index,
            "signal_mode":
                mode,
            "mode_member_index":
                mode_member_index,
            "member_id":
                member_id,
            "source_campaign":
                row["campaign"],
            "generated_events":
                generated_events,
            "immutable_split":
                split,
            "candidate_rows":
                candidate_rows,
            "candidate_access":
                "local_path",
            "candidate_parquet":
                candidate_location,
            "candidate_parquet_sha256":
                candidate_sha256,
            "candidate_schema_columns":
                75,
            "candidate_schema_sha256":
                ggf_schema_sha256,
            "common72_schema_sha256":
                common72_sha256,
            "source_bundle":
                source_bundle,
            "source_registry":
                str(
                    args.ggf_registry.resolve()
                ),
            "source_registry_sha256":
                source_registry_sha256[
                    mode
                ],
            "source_summary":
                str(
                    args.ggf_summary.resolve()
                ),
            "source_summary_sha256":
                source_summary_sha256[
                    mode
                ],
            "training_authorized":
                training_authorized,
            "validation_authorized":
                validation_authorized,
            "test_sealed":
                test_sealed,
            "physics_yield_authorized":
                False,
            "status":
                "pass",
        })

        global_member_index += 1

    for row in sorted(
        vbf_rows,
        key=lambda item: int(
            item["member_index"]
        ),
    ):
        mode = "vbf_hh4b"

        mode_member_index = int(
            row["member_index"]
        )

        member_id = row[
            "target_tag"
        ]

        require(
            row["signal_mode"] == mode,
            f"{member_id}: incorrect signal mode",
        )

        require(
            row["status"] == "pass",
            f"{member_id}: status is not pass",
        )

        generated_events = int(
            row["generated_events"]
        )

        candidate_rows = int(
            row["candidate_rows"]
        )

        split = row[
            "immutable_split"
        ]

        require(
            generated_events == 10_000,
            (
                f"{member_id}: generated count "
                "is not 10000"
            ),
        )

        require(
            split in {
                "train",
                "validation",
            },
            (
                f"{member_id}: invalid VBF "
                f"split {split!r}"
            ),
        )

        require(
            int(
                row[
                    "candidate_schema_columns"
                ]
            )
            == 72,
            (
                f"{member_id}: candidate schema "
                "is not 72 columns"
            ),
        )

        require(
            row[
                "candidate_schema_sha256"
            ]
            == common72_sha256,
            (
                f"{member_id}: VBF schema "
                "SHA-256 mismatch"
            ),
        )

        candidate_location = row[
            "candidate_parquet"
        ]

        require(
            candidate_location.startswith(
                "/store/"
            ),
            (
                f"{member_id}: candidate path "
                "is not an EOS LFN"
            ),
        )

        candidate_sha256 = row[
            "candidate_parquet_sha256"
        ]

        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                candidate_sha256,
            )
            is not None,
            (
                f"{member_id}: invalid candidate "
                "SHA-256"
            ),
        )

        training_authorized = parse_bool(
            row["training_authorized"]
        )

        validation_authorized = parse_bool(
            row["validation_authorized"]
        )

        test_sealed = parse_bool(
            row["test_sealed"]
        )

        physics_yield_authorized = parse_bool(
            row[
                "physics_yield_authorized"
            ]
        )

        require(
            not physics_yield_authorized,
            (
                f"{member_id}: physics yield "
                "unexpectedly authorized"
            ),
        )

        validate_split_flags(
            mode=mode,
            member_id=member_id,
            split=split,
            training_authorized=
                training_authorized,
            validation_authorized=
                validation_authorized,
            test_sealed=test_sealed,
        )

        source_bundle = row[
            "source_remote_bundle"
        ]

        require(
            candidate_location
            not in candidate_locations,
            (
                f"{member_id}: duplicate "
                "candidate location"
            ),
        )

        require(
            source_bundle
            not in source_bundles,
            (
                f"{member_id}: duplicate "
                "source bundle"
            ),
        )

        candidate_locations.add(
            candidate_location
        )

        source_bundles.add(
            source_bundle
        )

        generated_by_mode[
            mode
        ] += generated_events

        candidates_by_mode[
            mode
        ] += candidate_rows

        generated_by_split[
            split
        ] += generated_events

        candidates_by_split[
            split
        ] += candidate_rows

        output_rows.append({
            "global_member_index":
                global_member_index,
            "signal_mode":
                mode,
            "mode_member_index":
                mode_member_index,
            "member_id":
                member_id,
            "source_campaign":
                row["source_campaign"],
            "generated_events":
                generated_events,
            "immutable_split":
                split,
            "candidate_rows":
                candidate_rows,
            "candidate_access":
                "eos_lfn",
            "candidate_parquet":
                candidate_location,
            "candidate_parquet_sha256":
                candidate_sha256,
            "candidate_schema_columns":
                72,
            "candidate_schema_sha256":
                common72_sha256,
            "common72_schema_sha256":
                common72_sha256,
            "source_bundle":
                source_bundle,
            "source_registry":
                str(
                    args.vbf_registry.resolve()
                ),
            "source_registry_sha256":
                source_registry_sha256[
                    mode
                ],
            "source_summary":
                str(
                    args.vbf_summary.resolve()
                ),
            "source_summary_sha256":
                source_summary_sha256[
                    mode
                ],
            "training_authorized":
                training_authorized,
            "validation_authorized":
                validation_authorized,
            "test_sealed":
                test_sealed,
            "physics_yield_authorized":
                False,
            "status":
                "pass",
        })

        global_member_index += 1

    require(
        len(output_rows)
        == EXPECTED_TOTAL_MEMBERS,
        (
            f"combined registry has "
            f"{len(output_rows)} members, not 110"
        ),
    )

    require(
        dict(generated_by_mode)
        == EXPECTED_GENERATED_BY_MODE,
        (
            "unexpected generated totals by mode: "
            f"{dict(generated_by_mode)}"
        ),
    )

    require(
        dict(candidates_by_mode)
        == EXPECTED_CANDIDATES_BY_MODE,
        (
            "unexpected candidate totals by mode: "
            f"{dict(candidates_by_mode)}"
        ),
    )

    require(
        dict(generated_by_split)
        == EXPECTED_GENERATED_BY_SPLIT,
        (
            "unexpected generated totals by split: "
            f"{dict(generated_by_split)}"
        ),
    )

    require(
        dict(candidates_by_split)
        == EXPECTED_CANDIDATES_BY_SPLIT,
        (
            "unexpected candidate totals by split: "
            f"{dict(candidates_by_split)}"
        ),
    )

    require(
        len(candidate_locations)
        == EXPECTED_TOTAL_MEMBERS,
        "candidate locations are not unique",
    )

    require(
        len(source_bundles)
        == EXPECTED_TOTAL_MEMBERS,
        "source bundles are not unique",
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    registry_output = (
        args.outdir
        / "hh4b_canonical_signal200k_registry.tsv"
    )

    with registry_output.open(
        "w",
        newline="",
    ) as handle:
        columns = list(
            output_rows[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "canonical_members":
            EXPECTED_TOTAL_MEMBERS,
        "signal_modes": [
            "ggf_hh4b",
            "vbf_hh4b",
        ],
        "generated_events":
            sum(
                generated_by_mode.values()
            ),
        "generated_events_by_mode":
            dict(generated_by_mode),
        "generated_events_by_split":
            dict(generated_by_split),
        "candidate_rows":
            sum(
                candidates_by_mode.values()
            ),
        "candidate_rows_by_mode":
            dict(candidates_by_mode),
        "candidate_rows_by_split":
            dict(candidates_by_split),
        "common_ml_schema_columns":
            72,
        "common_ml_schema_sha256":
            common72_sha256,
        "ggf_signal_schema_columns":
            75,
        "ggf_signal_schema_sha256":
            ggf_schema_sha256,
        "vbf_candidate_schema_columns":
            72,
        "vbf_candidate_schema_sha256":
            common72_sha256,
        "mode_membership_preserved":
            True,
        "mode_specific_normalization_required":
            True,
        "combined_by_generated_sample_size":
            False,
        "ggf_test_members_present":
            True,
        "ggf_test_members_sealed":
            True,
        "vbf_test_members_present":
            False,
        "vbf_test_split_invented":
            False,
        "physics_yield_authorized":
            False,
        "normalization_frozen":
            False,
        "reference_baseline_freeze_authorized":
            True,
        "full_signal_background_cutflow_authorized":
            False,
        "registry":
            str(registry_output),
        "source_registries": {
            "ggf_hh4b":
                str(
                    args.ggf_registry.resolve()
                ),
            "vbf_hh4b":
                str(
                    args.vbf_registry.resolve()
                ),
        },
        "source_registry_sha256":
            source_registry_sha256,
        "source_summary_sha256":
            source_summary_sha256,
    }

    summary_output = (
        args.outdir
        / "hh4b_canonical_signal200k_registry.json"
    )

    summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "HH4B_CANONICAL_SIGNAL200K_REGISTRY_VALID"
    )
    print(
        "HH4B_SIGNAL_MODES_PRESERVED"
    )
    print(
        "HH4B_SIGNAL_CANONICAL_MEMBERS=110"
    )
    print(
        "HH4B_SIGNAL_GENERATED_EVENTS=200000"
    )
    print(
        "HH4B_SIGNAL_CANDIDATE_ROWS=12395"
    )
    print(
        "HH4B_SIGNAL_COMMON72_SCHEMA_VALID"
    )
    print(
        "HH4B_SIGNAL_VBF_NO_TEST_SPLIT_PRESERVED"
    )
    print(
        "HH4B_SIGNAL_TEST_SEALED_GGF_ONLY"
    )
    print(
        "HH4B_SIGNAL_PHYSICS_YIELD_NOT_YET_AUTHORIZED"
    )
    print(
        "HH4B_SIGNAL_REFERENCE_BASELINE_FREEZE_AUTHORIZED"
    )
    print(
        "HH4B_FULL_SIGNAL_BACKGROUND_CUTFLOW_NOT_YET_AUTHORIZED"
    )
    print(
        f"summary_json={summary_output}"
    )
    print(
        f"registry_tsv={registry_output}"
    )


if __name__ == "__main__":
    main()
