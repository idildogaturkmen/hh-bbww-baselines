#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TTBAR_SALT = "final-background-canonical-ttbar270k-v1"
WAVEB_SALT = "final-background-canonical-waveb170k-v1"

TTBAR_TARGET_EVENTS = 270_000
TTBAR_VALIDATION_SHARDS = 5

WAVEB_TARGET_EVENTS = 170_000
WAVEB_SCALEOUT_SELECTED_SHARDS = 13

WAVEB_PILOT_CAMPAIGN = (
    "unified_background_5m_waveb_priority_"
    "pilots10k_20260721_v1"
)

WAVEB_SCALEOUT_CAMPAIGN = (
    "unified_background_5m_waveb_priority_"
    "scale400k_20260721_v1"
)

WAVEB_FAMILIES = (
    "tth_hbb",
    "ttz_zbb",
    "tttt",
    "vbf_hbb",
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def read_table(
    path: Path,
) -> list[dict[str, str]]:
    delimiter = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter=delimiter,
            )
        )


def hash_rank(
    salt: str,
    *values: Any,
) -> str:
    payload = "|".join([
        salt,
        *(
            str(value)
            for value in values
        ),
    ])

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


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


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_bool(
    value: Any,
) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


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

    ttbar_manifest_paths = [
        (
            store
            / "metadata"
            / "ttbar_100k_manifest.csv"
        ),
        (
            store
            / "metadata"
            / "ttbar_200k_manifest.csv"
        ),
    ]

    waveb_receipt_table = (
        repo
        / "outputs/agent_runs"
        / "final_background_missing10k_receipt_resolution_20260723_v1"
        / "waveb_wavec_complete_nonqa_deduplicated.tsv"
    )

    ttbar_accounting_path = (
        repo
        / "metadata/delphes"
        / "ttbar_legacy_300k_20260718"
        / "corrected_manifest_accounting.json"
    )

    waveb_plan_path = (
        repo
        / "metadata/production_plans"
        / "unified_background_5m_waveb_importance_plan_20260721.tsv"
    )

    required_inputs = [
        *ttbar_manifest_paths,
        waveb_receipt_table,
        ttbar_accounting_path,
        waveb_plan_path,
    ]

    for path in required_inputs:
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    outdir.mkdir(parents=True)

    # =====================================================
    # Canonical legacy ttbar 270k selection
    # =====================================================

    ttbar_rows: list[dict[str, Any]] = []

    for manifest_path in ttbar_manifest_paths:
        for row in read_table(
            manifest_path
        ):
            required_columns = {
                "campaign",
                "shard",
                "tag",
                "seed",
                "events_per_shard",
                "event_parquet",
                "candidate_parquet",
            }

            require(
                required_columns.issubset(row),
                (
                    f"{manifest_path}: missing "
                    "required columns"
                ),
            )

            events = int(
                row["events_per_shard"]
            )

            require(
                events == 10_000,
                (
                    f"{manifest_path}: unexpected "
                    f"event count {events}"
                ),
            )

            event_parquet = Path(
                row["event_parquet"]
            )

            candidate_parquet = Path(
                row["candidate_parquet"]
            )

            require(
                event_parquet.is_file(),
                (
                    "missing ttbar event parquet "
                    f"{event_parquet}"
                ),
            )

            require(
                candidate_parquet.is_file(),
                (
                    "missing ttbar candidate parquet "
                    f"{candidate_parquet}"
                ),
            )

            copy = dict(row)

            copy.update({
                "events": events,
                "source_manifest":
                    str(manifest_path),
                "source_manifest_sha256":
                    sha256_file(
                        manifest_path
                    ),
                "membership_rank":
                    hash_rank(
                        TTBAR_SALT,
                        row["campaign"],
                        row["shard"],
                        row["seed"],
                        row["tag"],
                    ),
            })

            ttbar_rows.append(copy)

    require(
        len(ttbar_rows) == 30,
        (
            "expected thirty legacy ttbar "
            f"shards, found {len(ttbar_rows)}"
        ),
    )

    require(
        sum(
            int(row["events"])
            for row in ttbar_rows
        )
        == 300_000,
        "legacy ttbar source total is not 300k",
    )

    for key in (
        "membership_rank",
        "event_parquet",
        "candidate_parquet",
    ):
        require(
            len({
                str(row[key])
                for row in ttbar_rows
            })
            == 30,
            f"duplicate legacy ttbar {key}",
        )

    ttbar_sorted = sorted(
        ttbar_rows,
        key=lambda row: row[
            "membership_rank"
        ],
    )

    ttbar_selected = [
        dict(row)
        for row in ttbar_sorted[:27]
    ]

    ttbar_holdout = [
        dict(row)
        for row in ttbar_sorted[27:]
    ]

    validation_order = sorted(
        ttbar_selected,
        key=lambda row: hash_rank(
            TTBAR_SALT,
            "split",
            row["campaign"],
            row["shard"],
            row["seed"],
        ),
    )

    validation_ids = {
        (
            row["campaign"],
            row["shard"],
            row["seed"],
        )
        for row in validation_order[
            :TTBAR_VALIDATION_SHARDS
        ]
    }

    for row in ttbar_selected:
        identity = (
            row["campaign"],
            row["shard"],
            row["seed"],
        )

        row["dataset_split"] = (
            "validation"
            if identity in validation_ids
            else "train"
        )

        row["dataset_role"] = (
            "canonical_validation"
            if identity in validation_ids
            else "canonical_train"
        )

        row["canonical_membership"] = True
        row["selection_policy"] = (
            "salted_sha256_rank_first_27_of_30"
        )
        row["physics_yield_authorized"] = False

    for row in ttbar_holdout:
        row["dataset_split"] = "holdout"
        row["dataset_role"] = (
            "noncanonical_holdout"
        )
        row["canonical_membership"] = False
        row["selection_policy"] = (
            "salted_sha256_rank_last_3_of_30"
        )
        row["physics_yield_authorized"] = False

    require(
        sum(
            int(row["events"])
            for row in ttbar_selected
        )
        == TTBAR_TARGET_EVENTS,
        "canonical ttbar total is not 270k",
    )

    require(
        Counter(
            row["dataset_split"]
            for row in ttbar_selected
        )
        == {
            "train": 22,
            "validation": 5,
        },
        "canonical ttbar split is not 22/5",
    )

    # =====================================================
    # Canonical Wave-B 170k selection
    # =====================================================

    receipt_rows = read_table(
        waveb_receipt_table
    )

    waveb_rows = [
        dict(row)
        for row in receipt_rows
        if row.get("campaign") in {
            WAVEB_PILOT_CAMPAIGN,
            WAVEB_SCALEOUT_CAMPAIGN,
        }
    ]

    require(
        len(waveb_rows) == 44,
        (
            "expected forty-four successful "
            f"Wave-B receipts, found {len(waveb_rows)}"
        ),
    )

    for row in waveb_rows:
        require(
            row["family"] in WAVEB_FAMILIES,
            (
                "unexpected Wave-B family "
                f"{row['family']}"
            ),
        )

        require(
            int(row["events"]) == 10_000,
            (
                f"{row['target_tag']}: "
                "expected 10k events"
            ),
        )

        require(
            row["stage"]
            == "complete_copied_and_verified",
            (
                f"{row['target_tag']}: "
                "incomplete stage"
            ),
        )

        require(
            str(row["exit_status"]) == "0",
            (
                f"{row['target_tag']}: "
                "nonzero exit status"
            ),
        )

        require(
            not parse_bool(
                row["qa_or_test"]
            ),
            (
                f"{row['target_tag']}: "
                "QA/test shard cannot be canonical"
            ),
        )

        require(
            parse_bool(
                row[
                    "canonical_candidate"
                ]
            ),
            (
                f"{row['target_tag']}: "
                "not marked as a canonical candidate"
            ),
        )

        require(
            row["remote_bundle"],
            (
                f"{row['target_tag']}: "
                "missing remote bundle"
            ),
        )

        require(
            row["bundle_sha256"],
            (
                f"{row['target_tag']}: "
                "missing bundle SHA"
            ),
        )

        row["membership_rank"] = hash_rank(
            WAVEB_SALT,
            row["campaign"],
            row["family"],
            row["shard_id"],
            row["seed"],
            row["remote_bundle"],
        )

    for key in (
        "remote_bundle",
        "bundle_sha256",
        "target_tag",
        "membership_rank",
    ):
        require(
            len({
                row[key]
                for row in waveb_rows
            })
            == 44,
            f"duplicate Wave-B {key}",
        )

    pilot_rows = [
        row
        for row in waveb_rows
        if (
            row["campaign"]
            == WAVEB_PILOT_CAMPAIGN
        )
    ]

    scaleout_rows = [
        row
        for row in waveb_rows
        if (
            row["campaign"]
            == WAVEB_SCALEOUT_CAMPAIGN
        )
    ]

    require(
        len(pilot_rows) == 4,
        "expected four Wave-B pilot shards",
    )

    require(
        len(scaleout_rows) == 40,
        "expected forty Wave-B scaleout shards",
    )

    require(
        Counter(
            row["family"]
            for row in pilot_rows
        )
        == {
            family: 1
            for family in WAVEB_FAMILIES
        },
        (
            "Wave-B pilots do not contain "
            "one shard per family"
        ),
    )

    pilot_train_families = sorted({
        row["family"]
        for row in pilot_rows
        if row["dataset_split"] == "train"
    })

    require(
        len(pilot_train_families) == 2,
        (
            "expected two train-family pilots "
            "for Wave-B"
        ),
    )

    extra_family = min(
        WAVEB_FAMILIES,
        key=lambda family: hash_rank(
            WAVEB_SALT,
            "extra_family",
            family,
        ),
    )

    validation_family = min(
        pilot_train_families,
        key=lambda family: hash_rank(
            WAVEB_SALT,
            "extra_validation_family",
            family,
        ),
    )

    scaleout_by_family: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in scaleout_rows:
        scaleout_by_family[
            row["family"]
        ].append(row)

    selected_scaleout: list[
        dict[str, Any]
    ] = []

    for family in WAVEB_FAMILIES:
        additional_quota = (
            4
            if family == extra_family
            else 3
        )

        family_rows = (
            scaleout_by_family[family]
        )

        train_rows = sorted(
            [
                row
                for row in family_rows
                if (
                    row["dataset_split"]
                    == "train"
                )
            ],
            key=lambda row: row[
                "membership_rank"
            ],
        )

        validation_rows = sorted(
            [
                row
                for row in family_rows
                if (
                    row["dataset_split"]
                    == "validation"
                )
            ],
            key=lambda row: row[
                "membership_rank"
            ],
        )

        require(
            len(train_rows) == 8,
            (
                f"{family}: expected eight "
                "scaleout train shards"
            ),
        )

        require(
            len(validation_rows) == 2,
            (
                f"{family}: expected two "
                "scaleout validation shards"
            ),
        )

        validation_quota = (
            1
            if family == validation_family
            else 0
        )

        train_quota = (
            additional_quota
            - validation_quota
        )

        selected_scaleout.extend(
            train_rows[:train_quota]
        )

        selected_scaleout.extend(
            validation_rows[
                :validation_quota
            ]
        )

    require(
        len(selected_scaleout)
        == WAVEB_SCALEOUT_SELECTED_SHARDS,
        (
            "Wave-B selected scaleout count "
            f"is {len(selected_scaleout)}, not 13"
        ),
    )

    selected_identity = {
        row["remote_bundle"]
        for row in selected_scaleout
    }

    waveb_selected = [
        dict(row)
        for row in (
            pilot_rows
            + selected_scaleout
        )
    ]

    waveb_holdout = [
        dict(row)
        for row in scaleout_rows
        if (
            row["remote_bundle"]
            not in selected_identity
        )
    ]

    for row in waveb_selected:
        row["canonical_membership"] = True
        row["selection_policy"] = (
            "all_four_pilots_plus_stratified_"
            "salted_sha256_scaleout_selection"
        )
        row["physics_yield_authorized"] = False

    for row in waveb_holdout:
        row["canonical_membership"] = False
        row["selection_policy"] = (
            "noncanonical_scaleout_reserve"
        )
        row["physics_yield_authorized"] = False

    require(
        len(waveb_selected) == 17,
        "canonical Wave-B registry is not 17 shards",
    )

    require(
        len(waveb_holdout) == 27,
        "Wave-B reserve is not 27 shards",
    )

    require(
        sum(
            int(row["events"])
            for row in waveb_selected
        )
        == WAVEB_TARGET_EVENTS,
        "canonical Wave-B total is not 170k",
    )

    selected_family_counts = Counter(
        row["family"]
        for row in waveb_selected
    )

    require(
        sorted(
            selected_family_counts.values()
        )
        == [4, 4, 4, 5],
        (
            "canonical Wave-B family allocation "
            f"is invalid: {selected_family_counts}"
        ),
    )

    selected_split_counts = Counter(
        row["dataset_split"]
        for row in waveb_selected
    )

    require(
        selected_split_counts
        == {
            "train": 14,
            "validation": 3,
        },
        (
            "canonical Wave-B split is not "
            f"14/3: {selected_split_counts}"
        ),
    )

    # =====================================================
    # Write immutable membership records
    # =====================================================

    ttbar_columns = [
        "canonical_membership",
        "campaign",
        "shard",
        "tag",
        "seed",
        "events",
        "dataset_split",
        "dataset_role",
        "event_parquet",
        "candidate_parquet",
        "source_manifest",
        "source_manifest_sha256",
        "membership_rank",
        "selection_policy",
        "physics_yield_authorized",
    ]

    waveb_columns = [
        "canonical_membership",
        "campaign",
        "family",
        "target_tag",
        "shard_id",
        "seed",
        "events",
        "dataset_split",
        "dataset_role",
        "remote_bundle",
        "bundle_sha256",
        "bundle_adler32",
        "receipt_path",
        "membership_rank",
        "selection_policy",
        "physics_yield_authorized",
    ]

    ttbar_registry_path = (
        outdir
        / "canonical_ttbar270k_registry.tsv"
    )

    ttbar_holdout_path = (
        outdir
        / "noncanonical_ttbar30k_holdout.tsv"
    )

    waveb_registry_path = (
        outdir
        / "canonical_waveb170k_registry.tsv"
    )

    waveb_reserve_path = (
        outdir
        / "noncanonical_waveb270k_reserve.tsv"
    )

    write_tsv(
        ttbar_registry_path,
        sorted(
            ttbar_selected,
            key=lambda row: (
                row["dataset_split"],
                row["membership_rank"],
            ),
        ),
        ttbar_columns,
    )

    write_tsv(
        ttbar_holdout_path,
        ttbar_holdout,
        ttbar_columns,
    )

    write_tsv(
        waveb_registry_path,
        sorted(
            waveb_selected,
            key=lambda row: (
                row["family"],
                row["dataset_split"],
                row["membership_rank"],
            ),
        ),
        waveb_columns,
    )

    write_tsv(
        waveb_reserve_path,
        sorted(
            waveb_holdout,
            key=lambda row: (
                row["family"],
                row["membership_rank"],
            ),
        ),
        waveb_columns,
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "canonical_subset_membership_frozen":
            True,
        "selection_semantics": (
            "new deterministic canonical definition; "
            "not recovery of a historical registry"
        ),
        "ttbar": {
            "source_events": 300_000,
            "canonical_events":
                TTBAR_TARGET_EVENTS,
            "canonical_shards": 27,
            "canonical_train_events":
                220_000,
            "canonical_validation_events":
                50_000,
            "noncanonical_holdout_events":
                30_000,
            "selection_salt":
                TTBAR_SALT,
            "registry":
                str(ttbar_registry_path),
            "holdout":
                str(ttbar_holdout_path),
        },
        "waveb": {
            "source_events": 440_000,
            "canonical_events":
                WAVEB_TARGET_EVENTS,
            "canonical_shards": 17,
            "canonical_train_events":
                140_000,
            "canonical_validation_events":
                30_000,
            "canonical_family_shards":
                dict(
                    sorted(
                        selected_family_counts.items()
                    )
                ),
            "extra_family":
                extra_family,
            "additional_validation_family":
                validation_family,
            "noncanonical_reserve_events":
                270_000,
            "selection_salt":
                WAVEB_SALT,
            "registry":
                str(waveb_registry_path),
            "reserve":
                str(waveb_reserve_path),
        },
        "canonical_existing_background_definition": {
            "qcd_events": 2_510_000,
            "ttbar_events": 270_000,
            "total_events": 2_780_000,
        },
        "canonical_final_background_definition": {
            "qcd": 2_510_000,
            "ttbar": 270_000,
            "wavea": 1_300_000,
            "waveb": 170_000,
            "wavec1": 220_000,
            "single_top": 330_000,
            "hbb": 150_000,
            "triboson": 50_000,
            "total": 5_000_000,
        },
        "unified_background_5m_registry_audit_authorized":
            True,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "canonical_background_subset_freeze.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "CANONICAL_TTBAR270K_SELECTION_FROZEN"
    )
    print(
        "CANONICAL_TTBAR270K_TRAIN_EVENTS=220000"
    )
    print(
        "CANONICAL_TTBAR270K_VALIDATION_EVENTS=50000"
    )
    print(
        "CANONICAL_WAVEB170K_SELECTION_FROZEN"
    )
    print(
        "CANONICAL_WAVEB170K_TRAIN_EVENTS=140000"
    )
    print(
        "CANONICAL_WAVEB170K_VALIDATION_EVENTS=30000"
    )
    print(
        "CANONICAL_SUBSET_MEMBERSHIP_FREEZE_VALID"
    )
    print(
        "CANONICAL_BACKGROUND_TARGET_EVENTS=5000000"
    )
    print(
        "UNIFIED_BACKGROUND_5M_REGISTRY_AUDIT_AUTHORIZED"
    )
    print(
        "NO_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
