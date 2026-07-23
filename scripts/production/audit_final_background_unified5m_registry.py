#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_COMPONENTS = {
    "qcd": 2_510_000,
    "ttbar": 270_000,
    "wavea": 1_300_000,
    "waveb": 170_000,
    "wavec1": 220_000,
    "single_top": 330_000,
    "hbb": 150_000,
    "triboson": 50_000,
}

EXPECTED_MEMBER_COUNTS = {
    "qcd": 261,
    "ttbar": 27,
    "wavea": 130,
    "waveb": 17,
    "wavec1": 22,
    "single_top": 33,
    "hbb": 15,
    "triboson": 15,
}

WAVEA_CAMPAIGNS = {
    "unified_background_5m_wavea_pilots_20260720_retry1": 4,
    "unified_background_5m_wavea_proven111_20260720_v1": 111,
    "unified_background_5m_wavea_zbbbb_remaining15_20260721_v1": 15,
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "boolean cannot be interpreted as an event count"
        )

    return int(float(str(value).strip()))


def sha256_file(path: Path) -> str:
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


def read_json(path: Path) -> dict[str, Any]:
    require(
        path.is_file(),
        f"missing JSON input {path}",
    )

    document = json.loads(
        path.read_text()
    )

    require(
        isinstance(document, dict),
        f"{path}: expected a JSON object",
    )

    return document


def read_table(
    path: Path,
) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"missing table {path}",
    )

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


def first_value(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = "",
) -> Any:
    for key in keys:
        if key in row:
            value = row[key]

            if value not in (
                None,
                "",
            ):
                return value

    return default


def receipt_event_count(
    receipt: dict[str, Any],
) -> int:
    value = first_value(
        receipt,
        (
            "root_events",
            "accepted_events",
            "generated_events",
            "n_events",
            "events",
            "hepmc_events",
        ),
        None,
    )

    require(
        value is not None,
        "receipt has no generated-event field",
    )

    return as_int(value)


def receipt_is_successful(
    receipt: dict[str, Any],
) -> bool:
    stage = str(
        receipt.get("stage", "")
    )

    exit_value = first_value(
        receipt,
        (
            "exit_status",
            "exit_code",
        ),
        0,
    )

    try:
        exit_code = as_int(
            exit_value
        )
    except Exception:
        return False

    return (
        stage
        == "complete_copied_and_verified"
        and exit_code == 0
        and receipt_event_count(
            receipt
        ) > 0
    )


def registry_row_from_receipt(
    *,
    component: str,
    source_kind: str,
    source_audit: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    receipt = read_json(
        receipt_path
    )

    require(
        receipt_is_successful(
            receipt
        ),
        (
            f"{receipt_path}: receipt is not "
            "complete and successful"
        ),
    )

    if (
        "physics_yield_authorized"
        in receipt
    ):
        require(
            receipt[
                "physics_yield_authorized"
            ]
            is False,
            (
                f"{receipt_path}: receipt "
                "incorrectly authorizes physics yield"
            ),
        )

    campaign = str(
        receipt.get(
            "campaign",
            receipt_path.parents[1].name,
        )
    )

    family = str(
        receipt.get("family", "")
    )

    target_tag = str(
        first_value(
            receipt,
            (
                "target_tag",
                "tag",
                "run_name",
            ),
            "",
        )
    )

    searchable = " ".join([
        campaign,
        family,
        target_tag,
        str(receipt_path),
    ]).lower()

    for forbidden in (
        "canary",
        "smoke",
        "dryrun",
    ):
        require(
            forbidden not in searchable,
            (
                f"{receipt_path}: QA/test receipt "
                f"contains forbidden marker {forbidden}"
            ),
        )

    remote_bundle = str(
        receipt.get(
            "remote_bundle",
            "",
        )
    )

    bundle_sha256 = str(
        receipt.get(
            "bundle_sha256",
            "",
        )
    )

    require(
        remote_bundle.startswith(
            "/store/"
        ),
        (
            f"{receipt_path}: missing or invalid "
            "remote bundle"
        ),
    )

    require(
        len(bundle_sha256) == 64,
        (
            f"{receipt_path}: missing or invalid "
            "bundle SHA-256"
        ),
    )

    dataset_split = str(
        receipt.get(
            "dataset_split",
            "",
        )
    )

    require(
        dataset_split
        in {
            "train",
            "validation",
        },
        (
            f"{receipt_path}: invalid dataset split "
            f"{dataset_split!r}"
        ),
    )

    return {
        "component": component,
        "family": family,
        "source_kind":
            source_kind,
        "source_campaign":
            campaign,
        "source_audit":
            str(source_audit),
        "target_tag":
            target_tag,
        "shard_id":
            str(
                receipt.get(
                    "shard_id",
                    "",
                )
            ),
        "seed":
            str(
                receipt.get(
                    "seed",
                    "",
                )
            ),
        "pythia_seed":
            str(
                receipt.get(
                    "pythia_seed",
                    "",
                )
            ),
        "dataset_split":
            dataset_split,
        "dataset_role":
            str(
                receipt.get(
                    "dataset_role",
                    "",
                )
            ),
        "events":
            receipt_event_count(
                receipt
            ),
        "candidate_rows":
            as_int(
                receipt.get(
                    "candidate_rows",
                    0,
                )
            ),
        "receipt_path":
            str(receipt_path),
        "remote_bundle":
            remote_bundle,
        "bundle_sha256":
            bundle_sha256,
        "bundle_adler32":
            str(
                receipt.get(
                    "bundle_adler32",
                    "",
                )
            ).lower(),
        "local_event_parquet":
            "",
        "local_candidate_parquet":
            "",
        "canonical_identity":
            "bundle:"
            + remote_bundle,
        "canonical_membership":
            True,
        "count_toward_background_target":
            True,
        "physics_yield_authorized":
            False,
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

    # -----------------------------------------------------
    # Authoritative source paths
    # -----------------------------------------------------

    subset_freeze_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_canonical_subset_freeze_20260723_v1"
        / "canonical_background_subset_freeze.json"
    )

    qcd_audit_path = (
        repo
        / "outputs/agent_runs"
        / "qcd_adaptive_2p51m_full_audit_20260718_retry1"
        / "qcd_1p01m_cross_layer_validation.json"
    )

    wavea_audit_path = (
        repo
        / "outputs/agent_runs"
        / "unified_background_5m_postscale_receipt_audit_20260721"
        / "new_1p30m_receipt_audit.json"
    )

    wavec_pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / "wavec1_background_pilot40k_audit_20260721_v2"
        / "wavec1_background_pilot40k_audit.json"
    )

    wavec_scaleout_audit_path = (
        repo
        / "outputs/agent_runs"
        / "wavec1_background_scaleout180k_audit_20260722_v1"
        / "wavec1_background_scaleout180k_audit.json"
    )

    wavec_checkpoint_path = (
        repo
        / "outputs/agent_runs"
        / "background_remaining530k_readiness_20260722_v1"
        / "wavec1_validated220k.json"
    )

    single_top_pilot_path = (
        repo
        / "outputs/agent_runs"
        / "background_single_top_pilot50k_audit_20260722_v1"
        / "background_single_top_pilot50k_audit.json"
    )

    single_top_scaleout_path = (
        repo
        / "outputs/agent_runs"
        / "background_single_top_scaleout280k_final_audit_20260722_v2"
        / "background_single_top_scaleout280k_audit.json"
    )

    hbb_pilot_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_pilots10k_audit_20260722_v1"
        / "hbb_pilots10k_final_audit.json"
    )

    hbb_scaleout_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_scaleouts130k_audit_20260722_v1"
        / "hbb_scaleouts130k_final_audit.json"
    )

    triboson_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson50k_registry_audit_20260723_v1"
        / "triboson_final50k_registry.json"
    )

    subset_freeze = read_json(
        subset_freeze_path
    )

    qcd_audit = read_json(
        qcd_audit_path
    )

    wavea_audit = read_json(
        wavea_audit_path
    )

    wavec_pilot_audit = read_json(
        wavec_pilot_audit_path
    )

    wavec_scaleout_audit = read_json(
        wavec_scaleout_audit_path
    )

    wavec_checkpoint = read_json(
        wavec_checkpoint_path
    )

    single_top_pilot = read_json(
        single_top_pilot_path
    )

    single_top_scaleout = read_json(
        single_top_scaleout_path
    )

    hbb_pilot = read_json(
        hbb_pilot_path
    )

    hbb_scaleout = read_json(
        hbb_scaleout_path
    )

    triboson = read_json(
        triboson_path
    )

    require(
        subset_freeze.get(
            "status"
        )
        == "pass",
        "canonical subset freeze did not pass",
    )

    require(
        subset_freeze.get(
            "canonical_subset_membership_frozen"
        )
        is True,
        "canonical subset membership is not frozen",
    )

    require(
        subset_freeze[
            "canonical_final_background_definition"
        ]["total"]
        == 5_000_000,
        "canonical target definition is not 5M",
    )

    # -----------------------------------------------------
    # Global receipt index
    # -----------------------------------------------------

    receipt_documents: list[
        tuple[Path, dict[str, Any]]
    ] = []

    receipt_by_remote: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ] = defaultdict(list)

    receipt_by_sha: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ] = defaultdict(list)

    for path in sorted(
        (
            store
            / "condor_return"
        ).glob(
            "*/receipts/*.json"
        )
    ):
        try:
            document = json.loads(
                path.read_text()
            )
        except Exception:
            continue

        if not isinstance(
            document,
            dict,
        ):
            continue

        receipt_documents.append(
            (
                path,
                document,
            )
        )

        remote = str(
            document.get(
                "remote_bundle",
                "",
            )
        )

        bundle_sha = str(
            document.get(
                "bundle_sha256",
                "",
            )
        )

        if remote:
            receipt_by_remote[
                remote
            ].append(
                (
                    path,
                    document,
                )
            )

        if bundle_sha:
            receipt_by_sha[
                bundle_sha
            ].append(
                (
                    path,
                    document,
                )
            )

    def resolve_receipt(
        *,
        receipt_path_value: Any = "",
        remote_bundle: str = "",
        bundle_sha256: str = "",
        campaign: str = "",
    ) -> Path:
        if receipt_path_value:
            path = Path(
                str(
                    receipt_path_value
                )
            )

            require(
                path.is_file(),
                f"missing receipt {path}",
            )

            return path

        candidates: list[
            tuple[Path, dict[str, Any]]
        ] = []

        if remote_bundle:
            candidates.extend(
                receipt_by_remote.get(
                    remote_bundle,
                    [],
                )
            )

        if (
            not candidates
            and bundle_sha256
        ):
            candidates.extend(
                receipt_by_sha.get(
                    bundle_sha256,
                    [],
                )
            )

        if campaign:
            candidates = [
                item
                for item in candidates
                if str(
                    item[1].get(
                        "campaign",
                        item[0].parents[1].name,
                    )
                )
                == campaign
            ]

        unique_paths = {
            item[0]
            for item in candidates
        }

        require(
            len(unique_paths) == 1,
            (
                "could not uniquely resolve receipt: "
                f"remote={remote_bundle!r}, "
                f"sha={bundle_sha256!r}, "
                f"campaign={campaign!r}, "
                f"matches={len(unique_paths)}"
            ),
        )

        return next(
            iter(
                unique_paths
            )
        )

    registry_rows: list[
        dict[str, Any]
    ] = []

    # =====================================================
    # QCD canonical 2.51M
    # =====================================================

    require(
        qcd_audit.get("status")
        == "pass",
        "QCD 2.51M audit did not pass",
    )

    require(
        qcd_audit.get(
            "processed_events"
        )
        == 2_510_000,
        "QCD processed-event total is not 2.51M",
    )

    require(
        qcd_audit.get(
            "processed_shards"
        )
        == 261,
        "QCD processed-shard total is not 261",
    )

    qcd_registry_path = Path(
        qcd_audit["registry"]
    )

    require(
        sha256_file(
            qcd_registry_path
        )
        == qcd_audit[
            "registry_sha256"
        ],
        "QCD registry SHA mismatch",
    )

    qcd_validation_path = Path(
        qcd_audit[
            "artifacts"
        ][
            "shard_validation"
        ]
    )

    qcd_registry_rows = read_table(
        qcd_registry_path
    )

    qcd_validation_rows = read_table(
        qcd_validation_path
    )

    require(
        len(qcd_registry_rows)
        == 261,
        "QCD registry does not have 261 rows",
    )

    require(
        len(qcd_validation_rows)
        == 261,
        (
            "QCD shard validation does not "
            "have 261 rows"
        ),
    )

    qcd_validation_by_remote = {
        row["remote_bundle"]: row
        for row in qcd_validation_rows
    }

    require(
        len(qcd_validation_by_remote)
        == 261,
        (
            "QCD validation contains duplicate "
            "remote bundles"
        ),
    )

    for row in qcd_registry_rows:
        remote = row[
            "remote_bundle"
        ]

        require(
            remote
            in qcd_validation_by_remote,
            (
                "QCD registry member missing "
                f"validation row: {remote}"
            ),
        )

        validation = (
            qcd_validation_by_remote[
                remote
            ]
        )

        require(
            as_bool(
                validation[
                    "all_checks_pass"
                ]
            ),
            (
                "QCD member failed cross-layer "
                f"validation: {remote}"
            ),
        )

        events = as_int(
            row["n_events"]
        )

        require(
            as_int(
                validation[
                    "n_generated"
                ]
            )
            == events,
            (
                "QCD registry/validation event "
                f"mismatch for {remote}"
            ),
        )

        split = row[
            "dataset_split"
        ]

        require(
            split
            in {
                "train",
                "validation",
                "test",
            },
            f"QCD invalid split {split!r}",
        )

        receipt_path = Path(
            row[
                "receipt_path"
            ]
        )

        require(
            receipt_path.is_file(),
            (
                "QCD receipt path is missing: "
                f"{receipt_path}"
            ),
        )

        registry_rows.append({
            "component": "qcd",
            "family":
                "qcd_hardqcd",
            "source_kind":
                "canonical_qcd_registry",
            "source_campaign":
                row["campaign"],
            "source_audit":
                str(qcd_audit_path),
            "target_tag":
                str(
                    validation.get(
                        "tag",
                        "",
                    )
                ),
            "shard_id":
                row["shard_id"],
            "seed":
                row["seed"],
            "pythia_seed":
                "",
            "dataset_split":
                split,
            "dataset_role":
                (
                    "canonical_train"
                    if split == "train"
                    else (
                        "canonical_validation"
                        if split == "validation"
                        else "canonical_test"
                    )
                ),
            "events":
                events,
            "candidate_rows":
                as_int(
                    validation[
                        "n_candidates"
                    ]
                ),
            "receipt_path":
                str(receipt_path),
            "remote_bundle":
                remote,
            "bundle_sha256":
                validation[
                    "bundle_sha256"
                ],
            "bundle_adler32":
                row[
                    "adler32"
                ].lower(),
            "local_event_parquet":
                "",
            "local_candidate_parquet":
                "",
            "canonical_identity":
                "bundle:"
                + remote,
            "canonical_membership":
                True,
            "count_toward_background_target":
                True,
            "physics_yield_authorized":
                False,
        })

    # =====================================================
    # Canonical ttbar 270k
    # =====================================================

    ttbar_registry_path = Path(
        subset_freeze[
            "ttbar"
        ][
            "registry"
        ]
    )

    ttbar_rows = read_table(
        ttbar_registry_path
    )

    require(
        len(ttbar_rows) == 27,
        "canonical ttbar registry is not 27 rows",
    )

    for row in ttbar_rows:
        require(
            as_bool(
                row[
                    "canonical_membership"
                ]
            ),
            "noncanonical ttbar row entered registry",
        )

        events = as_int(
            row["events"]
        )

        event_parquet = Path(
            row[
                "event_parquet"
            ]
        )

        candidate_parquet = Path(
            row[
                "candidate_parquet"
            ]
        )

        require(
            event_parquet.is_file(),
            (
                "missing canonical ttbar "
                f"event parquet {event_parquet}"
            ),
        )

        require(
            candidate_parquet.is_file(),
            (
                "missing canonical ttbar "
                f"candidate parquet {candidate_parquet}"
            ),
        )

        registry_rows.append({
            "component": "ttbar",
            "family":
                "ttbar_inclusive",
            "source_kind":
                "canonical_legacy_ttbar_subset",
            "source_campaign":
                row["campaign"],
            "source_audit":
                str(subset_freeze_path),
            "target_tag":
                row["tag"],
            "shard_id":
                row["shard"],
            "seed":
                row["seed"],
            "pythia_seed":
                "",
            "dataset_split":
                row[
                    "dataset_split"
                ],
            "dataset_role":
                row[
                    "dataset_role"
                ],
            "events":
                events,
            "candidate_rows":
                "",
            "receipt_path":
                "",
            "remote_bundle":
                "",
            "bundle_sha256":
                "",
            "bundle_adler32":
                "",
            "local_event_parquet":
                str(event_parquet),
            "local_candidate_parquet":
                str(candidate_parquet),
            "canonical_identity":
                "event_parquet:"
                + str(event_parquet),
            "canonical_membership":
                True,
            "count_toward_background_target":
                True,
            "physics_yield_authorized":
                False,
        })

    # =====================================================
    # Wave-A canonical 1.30M
    # =====================================================

    require(
        wavea_audit.get(
            "new_validated_events"
        )
        == 1_300_000,
        "Wave-A validated total is not 1.30M",
    )

    require(
        wavea_audit.get(
            "preexisting_inventory_events"
        )
        == 2_780_000,
        (
            "Wave-A preexisting inventory "
            "checkpoint is not 2.78M"
        ),
    )

    require(
        wavea_audit.get(
            "current_inventory_accounting_events"
        )
        == 4_080_000,
        (
            "Wave-A accounting checkpoint "
            "is not 4.08M"
        ),
    )

    if "structural_problems" in wavea_audit:
        require(
            wavea_audit[
                "structural_problems"
            ]
            in (
                [],
                {},
            ),
            "Wave-A audit reports structural problems",
        )

    wavea_rows: list[
        dict[str, Any]
    ] = []

    for (
        campaign,
        expected_successes,
    ) in WAVEA_CAMPAIGNS.items():
        receipt_dir = (
            store
            / "condor_return"
            / campaign
            / "receipts"
        )

        require(
            receipt_dir.is_dir(),
            (
                "missing Wave-A receipt directory "
                f"{receipt_dir}"
            ),
        )

        successful_paths: list[
            Path
        ] = []

        for path in sorted(
            receipt_dir.glob(
                "*.json"
            )
        ):
            try:
                receipt = json.loads(
                    path.read_text()
                )
            except Exception:
                continue

            if (
                isinstance(
                    receipt,
                    dict,
                )
                and receipt_is_successful(
                    receipt
                )
            ):
                successful_paths.append(
                    path
                )

        require(
            len(successful_paths)
            == expected_successes,
            (
                f"{campaign}: expected "
                f"{expected_successes} successful "
                f"receipts, found "
                f"{len(successful_paths)}"
            ),
        )

        for path in successful_paths:
            wavea_rows.append(
                registry_row_from_receipt(
                    component="wavea",
                    source_kind=
                        "wavea_audited_receipt",
                    source_audit=
                        wavea_audit_path,
                    receipt_path=path,
                )
            )

    require(
        len(wavea_rows) == 130,
        "Wave-A canonical membership is not 130 rows",
    )

    require(
        sum(
            row["events"]
            for row in wavea_rows
        )
        == 1_300_000,
        "Wave-A canonical total is not 1.30M",
    )

    wavea_family_events = Counter()

    for row in wavea_rows:
        wavea_family_events[
            row["family"]
        ] += row["events"]

    require(
        wavea_family_events
        == {
            "qcd_bbbb_general": 150_000,
            "qcd_bbbb_iht400to600": 150_000,
            "ttbar_inclusive": 750_000,
            "zbbbb": 250_000,
        },
        (
            "Wave-A family allocation mismatch: "
            f"{wavea_family_events}"
        ),
    )

    registry_rows.extend(
        wavea_rows
    )

    # =====================================================
    # Wave-B canonical 170k
    # =====================================================

    waveb_registry_path = Path(
        subset_freeze[
            "waveb"
        ][
            "registry"
        ]
    )

    waveb_members = read_table(
        waveb_registry_path
    )

    require(
        len(waveb_members) == 17,
        "canonical Wave-B registry is not 17 rows",
    )

    waveb_rows: list[
        dict[str, Any]
    ] = []

    for member in waveb_members:
        require(
            as_bool(
                member[
                    "canonical_membership"
                ]
            ),
            "noncanonical Wave-B row entered registry",
        )

        receipt_path = resolve_receipt(
            receipt_path_value=
                member[
                    "receipt_path"
                ],
        )

        row = registry_row_from_receipt(
            component="waveb",
            source_kind=
                "canonical_waveb_subset",
            source_audit=
                subset_freeze_path,
            receipt_path=
                receipt_path,
        )

        require(
            row["events"]
            == as_int(
                member[
                    "events"
                ]
            ),
            (
                "Wave-B receipt/registry "
                "event mismatch"
            ),
        )

        require(
            row["remote_bundle"]
            == member[
                "remote_bundle"
            ],
            (
                "Wave-B receipt/registry "
                "bundle mismatch"
            ),
        )

        require(
            row["bundle_sha256"]
            == member[
                "bundle_sha256"
            ],
            (
                "Wave-B receipt/registry "
                "SHA mismatch"
            ),
        )

        require(
            row["dataset_split"]
            == member[
                "dataset_split"
            ],
            (
                "Wave-B receipt/registry "
                "split mismatch"
            ),
        )

        waveb_rows.append(
            row
        )

    require(
        sum(
            row["events"]
            for row in waveb_rows
        )
        == 170_000,
        "Wave-B canonical total is not 170k",
    )

    require(
        Counter(
            row["dataset_split"]
            for row in waveb_rows
        )
        == {
            "train": 14,
            "validation": 3,
        },
        "Wave-B split is not 14/3",
    )

    registry_rows.extend(
        waveb_rows
    )

    # =====================================================
    # Wave-C1 canonical 220k
    # =====================================================

    require(
        wavec_pilot_audit.get(
            "status"
        )
        == "pass",
        "Wave-C1 pilot audit did not pass",
    )

    require(
        wavec_pilot_audit.get(
            "technical_pilot_valid"
        )
        is True,
        "Wave-C1 pilot is not technically valid",
    )

    require(
        wavec_pilot_audit.get(
            "validated_events"
        )
        == 40_000,
        "Wave-C1 pilot total is not 40k",
    )

    require(
        wavec_scaleout_audit.get(
            "status"
        )
        == "pass",
        "Wave-C1 scale-out audit did not pass",
    )

    require(
        wavec_scaleout_audit.get(
            "technical_scaleout_valid"
        )
        is True,
        "Wave-C1 scale-out is not technically valid",
    )

    require(
        wavec_scaleout_audit.get(
            "validated_events"
        )
        == 180_000,
        "Wave-C1 scale-out total is not 180k",
    )

    require(
        wavec_checkpoint.get(
            "validated_events"
        )
        == 220_000,
        "Wave-C1 combined checkpoint is not 220k",
    )

    require(
        wavec_checkpoint.get(
            "training_authorized"
        )
        is True,
        "Wave-C1 checkpoint does not authorize training",
    )

    require(
        wavec_checkpoint.get(
            "physics_yield_authorized"
        )
        is False,
        (
            "Wave-C1 checkpoint incorrectly "
            "authorizes physics yield"
        ),
    )

    wavec_rows: list[
        dict[str, Any]
    ] = []

    for (
        audit_document,
        audit_path,
        expected_rows,
        source_kind,
    ) in (
        (
            wavec_pilot_audit,
            wavec_pilot_audit_path,
            4,
            "wavec1_pilot",
        ),
        (
            wavec_scaleout_audit,
            wavec_scaleout_audit_path,
            18,
            "wavec1_scaleout",
        ),
    ):
        table_path = Path(
            audit_document[
                "audit_tsv"
            ]
        )

        table_rows = read_table(
            table_path
        )

        require(
            len(table_rows)
            == expected_rows,
            (
                f"{table_path}: expected "
                f"{expected_rows} rows"
            ),
        )

        for audit_row in table_rows:
            receipt_path = resolve_receipt(
                receipt_path_value=
                    first_value(
                        audit_row,
                        (
                            "receipt",
                            "receipt_path",
                        ),
                        "",
                    ),
                remote_bundle=
                    str(
                        audit_row.get(
                            "remote_bundle",
                            "",
                        )
                    ),
            )

            row = registry_row_from_receipt(
                component="wavec1",
                source_kind=
                    source_kind,
                source_audit=
                    audit_path,
                receipt_path=
                    receipt_path,
            )

            expected_events = as_int(
                first_value(
                    audit_row,
                    (
                        "generated_events",
                        "events",
                    ),
                )
            )

            require(
                row["events"]
                == expected_events,
                (
                    "Wave-C1 audit/receipt "
                    "event mismatch"
                ),
            )

            if audit_row.get(
                "remote_bundle"
            ):
                require(
                    row["remote_bundle"]
                    == audit_row[
                        "remote_bundle"
                    ],
                    (
                        "Wave-C1 audit/receipt "
                        "bundle mismatch"
                    ),
                )

            wavec_rows.append(
                row
            )

    require(
        len(wavec_rows) == 22,
        "Wave-C1 canonical membership is not 22 rows",
    )

    require(
        sum(
            row["events"]
            for row in wavec_rows
        )
        == 220_000,
        "Wave-C1 canonical total is not 220k",
    )

    wavec_family_events = Counter()

    for row in wavec_rows:
        wavec_family_events[
            row["family"]
        ] += row["events"]

    require(
        wavec_family_events
        == {
            "ttw": 60_000,
            "wh_hbb": 60_000,
            "ww": 40_000,
            "wz_zbb": 60_000,
        },
        (
            "Wave-C1 family totals mismatch: "
            f"{wavec_family_events}"
        ),
    )

    registry_rows.extend(
        wavec_rows
    )

    # =====================================================
    # Single-top canonical 330k
    # =====================================================

    require(
        single_top_pilot.get(
            "status"
        )
        == "pass",
        "single-top pilot audit did not pass",
    )

    require(
        single_top_pilot.get(
            "pilot_valid"
        )
        is True,
        "single-top pilot is not valid",
    )

    require(
        single_top_pilot.get(
            "counted_background_events_validated"
        )
        == 50_000,
        "single-top pilot total is not 50k",
    )

    require(
        single_top_scaleout.get(
            "status"
        )
        == "pass",
        "single-top scale-out audit did not pass",
    )

    require(
        single_top_scaleout.get(
            "scaleout_280k_valid"
        )
        is True,
        "single-top scale-out is not valid",
    )

    require(
        single_top_scaleout.get(
            "counted_background_events_validated"
        )
        == 280_000,
        "single-top scale-out total is not 280k",
    )

    single_top_rows: list[
        dict[str, Any]
    ] = []

    for (
        audit_document,
        audit_path,
        expected_rows,
        source_kind,
    ) in (
        (
            single_top_pilot,
            single_top_pilot_path,
            5,
            "single_top_pilot",
        ),
        (
            single_top_scaleout,
            single_top_scaleout_path,
            28,
            "single_top_scaleout",
        ),
    ):
        table_path = Path(
            audit_document[
                "audit_tsv"
            ]
        )

        table_rows = read_table(
            table_path
        )

        require(
            len(table_rows)
            == expected_rows,
            (
                f"{table_path}: expected "
                f"{expected_rows} rows"
            ),
        )

        for audit_row in table_rows:
            receipt_path = resolve_receipt(
                receipt_path_value=
                    audit_row.get(
                        "receipt_path",
                        "",
                    ),
                remote_bundle=
                    str(
                        audit_row.get(
                            "remote_bundle",
                            "",
                        )
                    ),
                bundle_sha256=
                    str(
                        audit_row.get(
                            "bundle_sha256",
                            "",
                        )
                    ),
                campaign=
                    str(
                        audit_row.get(
                            "campaign",
                            audit_document.get(
                                "campaign",
                                "",
                            ),
                        )
                    ),
            )

            row = registry_row_from_receipt(
                component="single_top",
                source_kind=
                    source_kind,
                source_audit=
                    audit_path,
                receipt_path=
                    receipt_path,
            )

            require(
                row["events"]
                == as_int(
                    audit_row[
                        "events"
                    ]
                ),
                (
                    "single-top audit/receipt "
                    "event mismatch"
                ),
            )

            single_top_rows.append(
                row
            )

    require(
        len(single_top_rows)
        == 33,
        (
            "single-top canonical membership "
            "is not 33 rows"
        ),
    )

    require(
        sum(
            row["events"]
            for row in single_top_rows
        )
        == 330_000,
        "single-top canonical total is not 330k",
    )

    registry_rows.extend(
        single_top_rows
    )

    # =====================================================
    # Hbb canonical 150k
    # =====================================================

    require(
        hbb_pilot.get(
            "status"
        )
        == "pass",
        "Hbb pilot audit did not pass",
    )

    require(
        hbb_pilot.get(
            "hbb_pilots20k_final_audit_valid"
        )
        is True,
        "Hbb pilot 20k audit is not valid",
    )

    require(
        hbb_pilot.get(
            "validated_pilot_events"
        )
        == 20_000,
        "Hbb pilot total is not 20k",
    )

    require(
        hbb_scaleout.get(
            "status"
        )
        == "pass",
        "Hbb scale-out audit did not pass",
    )

    require(
        hbb_scaleout.get(
            "hbb_scaleouts130k_final_audit_valid"
        )
        is True,
        "Hbb scale-out 130k audit is not valid",
    )

    require(
        hbb_scaleout.get(
            "validated_hbb_scaleout_events"
        )
        == 130_000,
        "Hbb scale-out total is not 130k",
    )

    hbb_rows: list[
        dict[str, Any]
    ] = []

    for (
        audit_rows,
        audit_path,
        source_kind,
    ) in (
        (
            hbb_pilot[
                "pilots"
            ],
            hbb_pilot_path,
            "hbb_pilot",
        ),
        (
            hbb_scaleout[
                "shards"
            ],
            hbb_scaleout_path,
            "hbb_scaleout",
        ),
    ):
        for audit_row in audit_rows:
            receipt_path = resolve_receipt(
                remote_bundle=
                    str(
                        audit_row.get(
                            "remote_bundle",
                            "",
                        )
                    ),
                bundle_sha256=
                    str(
                        audit_row[
                            "bundle_sha256"
                        ]
                    ),
                campaign=
                    str(
                        audit_row[
                            "campaign"
                        ]
                    ),
            )

            row = registry_row_from_receipt(
                component="hbb",
                source_kind=
                    source_kind,
                source_audit=
                    audit_path,
                receipt_path=
                    receipt_path,
            )

            expected_events = as_int(
                audit_row[
                    "events"
                ]
            )

            require(
                row["events"]
                == expected_events,
                "Hbb audit/receipt event mismatch",
            )

            require(
                as_int(
                    audit_row[
                        "direct_hbb_truth_events"
                    ]
                )
                == expected_events,
                "Hbb direct-truth total mismatch",
            )

            hbb_rows.append(
                row
            )

    require(
        len(hbb_rows) == 15,
        "Hbb canonical membership is not 15 rows",
    )

    require(
        sum(
            row["events"]
            for row in hbb_rows
        )
        == 150_000,
        "Hbb canonical total is not 150k",
    )

    hbb_family_events = Counter()

    for row in hbb_rows:
        hbb_family_events[
            row["family"]
        ] += row["events"]

    require(
        hbb_family_events
        == {
            "ggh_hbb": 100_000,
            "bbh_hbb_4fs": 50_000,
        },
        (
            "Hbb family totals mismatch: "
            f"{hbb_family_events}"
        ),
    )

    registry_rows.extend(
        hbb_rows
    )

    # =====================================================
    # Triboson canonical 50k
    # =====================================================

    require(
        triboson.get(
            "status"
        )
        == "pass",
        "triboson registry audit did not pass",
    )

    require(
        triboson.get(
            "triboson_final50k_registry_valid"
        )
        is True,
        "triboson final 50k registry is not valid",
    )

    require(
        triboson.get(
            "canonical_triboson50k_membership_frozen"
        )
        is True,
        "triboson membership is not frozen",
    )

    triboson_rows: list[
        dict[str, Any]
    ] = []

    for member in triboson[
        "registry"
    ]:
        receipt_path = resolve_receipt(
            receipt_path_value=
                member[
                    "source_receipt"
                ],
        )

        row = registry_row_from_receipt(
            component="triboson",
            source_kind=
                "canonical_triboson_registry",
            source_audit=
                triboson_path,
            receipt_path=
                receipt_path,
        )

        require(
            row["events"]
            == as_int(
                member[
                    "accepted_events"
                ]
            ),
            (
                "triboson registry/receipt "
                "event mismatch"
            ),
        )

        require(
            row["remote_bundle"]
            == member[
                "remote_bundle"
            ],
            (
                "triboson registry/receipt "
                "bundle mismatch"
            ),
        )

        require(
            row["bundle_sha256"]
            == member[
                "bundle_sha256"
            ],
            (
                "triboson registry/receipt "
                "SHA mismatch"
            ),
        )

        triboson_rows.append(
            row
        )

    require(
        len(triboson_rows)
        == 15,
        (
            "triboson canonical membership "
            "is not 15 rows"
        ),
    )

    require(
        sum(
            row["events"]
            for row in triboson_rows
        )
        == 50_000,
        "triboson canonical total is not 50k",
    )

    registry_rows.extend(
        triboson_rows
    )

    # =====================================================
    # Unified invariants
    # =====================================================

    component_events = Counter()
    component_members = Counter()
    split_events = Counter()
    split_members = Counter()
    family_events = Counter()

    for row in registry_rows:
        component_events[
            row["component"]
        ] += as_int(
            row["events"]
        )

        component_members[
            row["component"]
        ] += 1

        split_events[
            row["dataset_split"]
        ] += as_int(
            row["events"]
        )

        split_members[
            row["dataset_split"]
        ] += 1

        family_events[
            (
                row["component"],
                row["family"],
            )
        ] += as_int(
            row["events"]
        )

        require(
            row[
                "dataset_split"
            ]
            in {
                "train",
                "validation",
                "test",
            },
            (
                "invalid unified split for "
                f"{row['canonical_identity']}"
            ),
        )

        require(
            row[
                "canonical_membership"
            ]
            is True,
            "noncanonical row entered unified registry",
        )

        require(
            row[
                "count_toward_background_target"
            ]
            is True,
            (
                "uncounted row entered "
                "unified registry"
            ),
        )

        require(
            row[
                "physics_yield_authorized"
            ]
            is False,
            (
                "row incorrectly authorizes "
                "physics yield"
            ),
        )

    require(
        dict(
            component_events
        )
        == EXPECTED_COMPONENTS,
        (
            "unified component event totals mismatch: "
            f"{dict(component_events)}"
        ),
    )

    require(
        dict(
            component_members
        )
        == EXPECTED_MEMBER_COUNTS,
        (
            "unified component member counts mismatch: "
            f"{dict(component_members)}"
        ),
    )

    require(
        len(registry_rows) == 520,
        (
            "unified registry does not contain "
            f"520 members: {len(registry_rows)}"
        ),
    )

    total_events = sum(
        as_int(
            row["events"]
        )
        for row in registry_rows
    )

    require(
        total_events == 5_000_000,
        (
            "unified generated-event total "
            f"is {total_events}, not 5M"
        ),
    )

    identities = [
        row[
            "canonical_identity"
        ]
        for row in registry_rows
    ]

    require(
        len(set(identities))
        == len(identities),
        (
            "unified registry contains duplicate "
            "canonical identities"
        ),
    )

    remote_bundles = [
        row["remote_bundle"]
        for row in registry_rows
        if row["remote_bundle"]
    ]

    require(
        len(set(remote_bundles))
        == len(remote_bundles),
        (
            "unified registry contains duplicate "
            "remote bundles"
        ),
    )

    local_event_parquets = [
        row[
            "local_event_parquet"
        ]
        for row in registry_rows
        if row[
            "local_event_parquet"
        ]
    ]

    require(
        len(
            set(
                local_event_parquets
            )
        )
        == len(
            local_event_parquets
        ),
        (
            "unified registry contains duplicate "
            "local event Parquet paths"
        ),
    )

    # -----------------------------------------------------
    # Write immutable registry and summaries
    # -----------------------------------------------------

    outdir.mkdir(
        parents=True
    )

    registry_columns = [
        "component",
        "family",
        "source_kind",
        "source_campaign",
        "source_audit",
        "target_tag",
        "shard_id",
        "seed",
        "pythia_seed",
        "dataset_split",
        "dataset_role",
        "events",
        "candidate_rows",
        "receipt_path",
        "remote_bundle",
        "bundle_sha256",
        "bundle_adler32",
        "local_event_parquet",
        "local_candidate_parquet",
        "canonical_identity",
        "canonical_membership",
        "count_toward_background_target",
        "physics_yield_authorized",
    ]

    registry_path = (
        outdir
        / "unified_background_5m_registry.tsv"
    )

    sorted_registry = sorted(
        registry_rows,
        key=lambda row: (
            row["component"],
            row["family"],
            row["dataset_split"],
            row["source_campaign"],
            str(row["shard_id"]),
            row["canonical_identity"],
        ),
    )

    write_tsv(
        registry_path,
        sorted_registry,
        registry_columns,
    )

    component_rows = []

    for component in (
        "qcd",
        "ttbar",
        "wavea",
        "waveb",
        "wavec1",
        "single_top",
        "hbb",
        "triboson",
    ):
        component_rows.append({
            "component":
                component,
            "canonical_members":
                component_members[
                    component
                ],
            "canonical_events":
                component_events[
                    component
                ],
            "train_events":
                sum(
                    as_int(
                        row["events"]
                    )
                    for row in registry_rows
                    if (
                        row[
                            "component"
                        ]
                        == component
                        and row[
                            "dataset_split"
                        ]
                        == "train"
                    )
                ),
            "validation_events":
                sum(
                    as_int(
                        row["events"]
                    )
                    for row in registry_rows
                    if (
                        row[
                            "component"
                        ]
                        == component
                        and row[
                            "dataset_split"
                        ]
                        == "validation"
                    )
                ),
            "test_events":
                sum(
                    as_int(
                        row["events"]
                    )
                    for row in registry_rows
                    if (
                        row[
                            "component"
                        ]
                        == component
                        and row[
                            "dataset_split"
                        ]
                        == "test"
                    )
                ),
        })

    component_path = (
        outdir
        / "unified_background_5m_component_summary.tsv"
    )

    write_tsv(
        component_path,
        component_rows,
        [
            "component",
            "canonical_members",
            "canonical_events",
            "train_events",
            "validation_events",
            "test_events",
        ],
    )

    family_rows = [
        {
            "component":
                component,
            "family":
                family,
            "canonical_events":
                events,
        }
        for (
            component,
            family,
        ), events in sorted(
            family_events.items()
        )
    ]

    family_path = (
        outdir
        / "unified_background_5m_family_summary.tsv"
    )

    write_tsv(
        family_path,
        family_rows,
        [
            "component",
            "family",
            "canonical_events",
        ],
    )

    source_paths = [
        subset_freeze_path,
        qcd_audit_path,
        wavea_audit_path,
        wavec_pilot_audit_path,
        wavec_scaleout_audit_path,
        wavec_checkpoint_path,
        single_top_pilot_path,
        single_top_scaleout_path,
        hbb_pilot_path,
        hbb_scaleout_path,
        triboson_path,
    ]

    summary = {
        "schema_version": 1,
        "status": "pass",
        "unified_background_5m_registry_valid":
            True,
        "unified_background_5m_membership_frozen":
            True,
        "canonical_background_events":
            total_events,
        "canonical_background_members":
            len(registry_rows),
        "canonical_train_events":
            split_events["train"],
        "canonical_validation_events":
            split_events[
                "validation"
            ],
        "canonical_test_events":
            split_events[
                "test"
            ],
        "canonical_train_members":
            split_members["train"],
        "canonical_validation_members":
            split_members[
                "validation"
            ],
        "canonical_test_members":
            split_members[
                "test"
            ],
        "canonical_events_by_component":
            dict(
                component_events
            ),
        "canonical_members_by_component":
            dict(
                component_members
            ),
        "registry_tsv":
            str(registry_path),
        "registry_sha256":
            sha256_file(
                registry_path
            ),
        "component_summary_tsv":
            str(component_path),
        "component_summary_sha256":
            sha256_file(
                component_path
            ),
        "family_summary_tsv":
            str(family_path),
        "family_summary_sha256":
            sha256_file(
                family_path
            ),
        "source_documents": [
            {
                "path": str(path),
                "sha256":
                    sha256_file(path),
            }
            for path in source_paths
        ],
        "classifier_table_preparation_authorized":
            True,
        "normalization_freeze_authorized":
            True,
        "raw_generator_xsec_normalization_authorized":
            False,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "unified_background_5m_registry.json"
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
        "UNIFIED_BACKGROUND_5M_REGISTRY_VALID"
    )
    print(
        "UNIFIED_BACKGROUND_5M_MEMBERSHIP_FROZEN"
    )
    print(
        "UNIFIED_BACKGROUND_5M_EVENTS=5000000"
    )
    print(
        "UNIFIED_BACKGROUND_5M_MEMBERS=520"
    )
    print(
        "UNIFIED_BACKGROUND_5M_TRAIN_EVENTS="
        f"{split_events['train']}"
    )
    print(
        "UNIFIED_BACKGROUND_5M_VALIDATION_EVENTS="
        f"{split_events['validation']}"
    )
    print(
        "UNIFIED_BACKGROUND_5M_TEST_EVENTS="
        f"{split_events['test']}"
    )
    print(
        "BACKGROUND_CLASSIFIER_TABLE_PREPARATION_AUTHORIZED"
    )
    print(
        "BACKGROUND_NORMALIZATION_FREEZE_AUTHORIZED"
    )
    print(
        "NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"registry_tsv={registry_path}"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
