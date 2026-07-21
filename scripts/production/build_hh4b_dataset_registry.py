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


def truthy(value: object) -> bool:
    return str(
        value or ""
    ).strip().lower() in TRUE_VALUES


def integer(
    value: object,
    default: int = 0,
) -> int:
    text = str(
        value or ""
    ).strip()

    if not text:
        return default

    return int(text)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(
            f"ERROR: missing TSV: {path}"
        )

    with path.open(
        newline=""
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(
            f"ERROR: missing CSV: {path}"
        )

    with path.open(
        newline=""
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def exact_success(
    row: dict[str, str],
) -> bool:
    expected = integer(
        row.get("events"),
        0,
    )

    return (
        expected > 0
        and row.get("stage")
        == "complete_copied_and_verified"
        and integer(
            row.get("exit_status"),
            -1,
        )
        == 0
        and integer(
            row.get("lhe_events"),
            -1,
        )
        == expected
        and integer(
            row.get("hepmc_events"),
            -1,
        )
        == expected
        and integer(
            row.get("root_events"),
            -1,
        )
        == expected
    )


def fallback_group(family: str) -> str:
    explicit = {
        "ttbar_inclusive": "top",
        "zbbbb": "zjets_heavy_flavor",
        "qcd_bbbb_general": "qcd",
        "qcd_bbbb_iht400to600": "qcd",
        "zh4b_forced": "single_higgs",
        "zz4b_forced": "diboson",
    }

    if family in explicit:
        return explicit[family]

    if family.startswith("qcd"):
        return "qcd"

    if family.startswith("tt"):
        return "top"

    return "background"


def clean_note(text: str) -> str:
    return " ".join(
        str(text or "").split()
    )


def markdown_escape(value: object) -> str:
    return str(
        value
    ).replace(
        "|",
        "\\|",
    ).replace(
        "\n",
        " ",
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
        "--promotion-overlay",
        required=True,
    )

    parser.add_argument(
        "--scale-manifest",
        required=True,
    )

    parser.add_argument(
        "--waveb-plan",
        required=True,
    )

    parser.add_argument(
        "--manual-catalog",
        required=True,
    )

    parser.add_argument(
        "--output-tsv",
        required=True,
    )

    parser.add_argument(
        "--output-markdown",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    args = parser.parse_args()

    inventory_path = Path(
        args.inventory
    ).resolve()

    contract_path = Path(
        args.contract
    ).resolve()

    overlay_path = Path(
        args.promotion_overlay
    ).resolve()

    scale_manifest_path = Path(
        args.scale_manifest
    ).resolve()

    waveb_plan_path = Path(
        args.waveb_plan
    ).resolve()

    manual_catalog_path = Path(
        args.manual_catalog
    ).resolve()

    output_tsv = Path(
        args.output_tsv
    ).resolve()

    output_markdown = Path(
        args.output_markdown
    ).resolve()

    output_json = Path(
        args.output_json
    ).resolve()

    inventory = read_tsv(
        inventory_path
    )

    contract_rows = read_tsv(
        contract_path
    )

    overlay_rows = read_tsv(
        overlay_path
    )

    scale_rows = read_csv(
        scale_manifest_path
    )

    waveb_rows = read_tsv(
        waveb_plan_path
    )

    manual_rows = read_tsv(
        manual_catalog_path
    )

    contract_by_family = {
        row["family"]: row
        for row in contract_rows
    }

    overlay_by_key = {
        (
            row["campaign"],
            row["target_tag"],
        ): row
        for row in overlay_rows
    }

    if len(overlay_by_key) != len(
        overlay_rows
    ):
        raise SystemExit(
            "ERROR: duplicate promotion-overlay keys"
        )

    plan_by_family = {
        row["family"]: row
        for row in waveb_rows
    }

    manual_by_family = {
        row["family"]: row
        for row in manual_rows
    }

    if len(manual_by_family) != len(
        manual_rows
    ):
        raise SystemExit(
            "ERROR: duplicate manual-catalog families"
        )

    stats: dict[
        str,
        dict[str, object],
    ] = defaultdict(
        lambda: {
            "existing_rows": 0,
            "successful_rows": 0,
            "failed_or_incomplete_rows": 0,
            "existing_successful_events": 0,
            "existing_candidate_rows": 0,
            "train_existing_events": 0,
            "validation_existing_events": 0,
            "native_counted_events": 0,
            "promoted_counted_events": 0,
            "submitted_pending_events": 0,
            "submitted_train_events": 0,
            "submitted_validation_events": 0,
            "submitted_pending_counted_events": 0,
            "campaigns": set(),
            "accounting_classes": set(),
        }
    )

    for row in inventory:
        family = row.get(
            "family",
            "",
        ).strip()

        if not family:
            continue

        entry = stats[family]

        entry["existing_rows"] += 1

        entry["campaigns"].add(
            row.get(
                "campaign",
                "",
            )
        )

        entry["accounting_classes"].add(
            row.get(
                "accounting_class",
                "",
            )
        )

        if not exact_success(row):
            entry[
                "failed_or_incomplete_rows"
            ] += 1

            continue

        events = integer(
            row.get("events"),
            0,
        )

        candidates = integer(
            row.get(
                "candidate_rows",
            ),
            0,
        )

        entry["successful_rows"] += 1

        entry[
            "existing_successful_events"
        ] += events

        entry[
            "existing_candidate_rows"
        ] += candidates

        split = row.get(
            "dataset_split",
            "",
        )

        if split == "train":
            entry[
                "train_existing_events"
            ] += events

        elif split == "validation":
            entry[
                "validation_existing_events"
            ] += events

        key = (
            row.get(
                "campaign",
                "",
            ),
            row.get(
                "target_tag",
                "",
            ),
        )

        if truthy(
            row.get(
                "count_toward_5m",
            )
        ):
            entry[
                "native_counted_events"
            ] += events

        elif key in overlay_by_key:
            overlay = overlay_by_key[
                key
            ]

            if not truthy(
                overlay.get(
                    "count_toward_5m",
                )
            ):
                raise SystemExit(
                    "ERROR: promotion overlay entry "
                    f"is not counted: {key}"
                )

            entry[
                "promoted_counted_events"
            ] += events

    for row in scale_rows:
        family = row.get(
            "family",
            "",
        ).strip()

        if not family:
            raise SystemExit(
                "ERROR: scale manifest row lacks family"
            )

        entry = stats[family]

        events = integer(
            row.get("events"),
            0,
        )

        if events <= 0:
            raise SystemExit(
                f"ERROR: invalid pending events for {family}"
            )

        entry[
            "submitted_pending_events"
        ] += events

        split = row.get(
            "dataset_split",
            "",
        )

        if split == "train":
            entry[
                "submitted_train_events"
            ] += events

        elif split == "validation":
            entry[
                "submitted_validation_events"
            ] += events

        else:
            raise SystemExit(
                "ERROR: unexpected scale split: "
                f"{family}: {split}"
            )

        if truthy(
            row.get(
                "count_toward_5m",
            )
        ):
            entry[
                "submitted_pending_counted_events"
            ] += events

        entry["campaigns"].add(
            row.get(
                "campaign",
                "",
            )
        )

    families = sorted(
        set(stats)
        | set(contract_by_family)
        | set(plan_by_family)
        | set(manual_by_family)
    )

    records = []

    for family in families:
        entry = stats[family]

        contract = contract_by_family.get(
            family,
            {},
        )

        plan = plan_by_family.get(
            family,
            {},
        )

        manual = manual_by_family.get(
            family,
            {},
        )

        existing_events = int(
            entry[
                "existing_successful_events"
            ]
        )

        native_counted = int(
            entry[
                "native_counted_events"
            ]
        )

        promoted_counted = int(
            entry[
                "promoted_counted_events"
            ]
        )

        counted_existing = (
            native_counted
            + promoted_counted
        )

        submitted_pending = int(
            entry[
                "submitted_pending_events"
            ]
        )

        submitted_pending_counted = int(
            entry[
                "submitted_pending_counted_events"
            ]
        )

        projected_counted = (
            counted_existing
            + submitted_pending_counted
        )

        if manual:
            role = manual[
                "role"
            ]

            group = manual[
                "process_group"
            ]

            display_name = manual[
                "display_name"
            ]

        else:
            role = "background"

            group = (
                plan.get("group")
                or fallback_group(
                    family
                )
            )

            display_name = family

        if manual:
            training_authorized = manual[
                "training_authorized"
            ]

            physics_authorized = manual[
                "physics_yield_authorized"
            ]

            normalization_rule = manual[
                "normalization_rule"
            ]

            overlap_group = manual[
                "overlap_group"
            ]

        else:
            training_authorized = contract.get(
                "training_authorized",
                "False",
            )

            physics_authorized = contract.get(
                "physics_yield_authorized",
                "False",
            )

            normalization_rule = (
                contract.get(
                    "normalization_rule"
                )
                or plan.get(
                    "normalization_rule"
                )
                or "REVIEW_REQUIRED"
            )

            overlap_group = (
                contract.get(
                    "overlap_group"
                )
                or plan.get(
                    "overlap_group"
                )
                or "REVIEW_REQUIRED"
            )

        if (
            existing_events > 0
            and submitted_pending > 0
        ):
            dataset_status = (
                "receipt_backed_existing_plus_"
                "submitted_pending_receipt"
            )

        elif existing_events > 0:
            dataset_status = (
                "receipt_backed_existing"
            )

        elif submitted_pending > 0:
            dataset_status = (
                "submitted_pending_receipt"
            )

        elif manual:
            dataset_status = manual[
                "dataset_status"
            ]

        elif plan:
            dataset_status = (
                "planned_template_or_pilot_pending"
            )

        elif int(
            entry["existing_rows"]
        ) > 0:
            dataset_status = (
                "failed_or_incomplete_only"
            )

        else:
            dataset_status = (
                "registry_entry"
            )

        notes = []

        if manual.get("notes"):
            notes.append(
                clean_note(
                    manual["notes"]
                )
            )

        if contract.get(
            "policy_status"
        ):
            notes.append(
                "policy_status="
                + contract[
                    "policy_status"
                ]
            )

        if submitted_pending > 0:
            notes.append(
                "submitted production is pending "
                "receipt and EOS validation"
            )

        if plan and existing_events == 0:
            notes.append(
                "Wave-B family exists in the "
                "production plan but has no "
                "successful receipt-backed event "
                "in this snapshot"
            )

        records.append({
            "family": family,
            "display_name": display_name,
            "role": role,
            "process_group": group,
            "dataset_status": dataset_status,
            "existing_rows": int(
                entry[
                    "existing_rows"
                ]
            ),
            "successful_rows": int(
                entry[
                    "successful_rows"
                ]
            ),
            "failed_or_incomplete_rows": int(
                entry[
                    "failed_or_incomplete_rows"
                ]
            ),
            "existing_successful_events": (
                existing_events
            ),
            "train_existing_events": int(
                entry[
                    "train_existing_events"
                ]
            ),
            "validation_existing_events": int(
                entry[
                    "validation_existing_events"
                ]
            ),
            "existing_candidate_rows": int(
                entry[
                    "existing_candidate_rows"
                ]
            ),
            "native_counted_events": native_counted,
            "promoted_counted_events": (
                promoted_counted
            ),
            "counted_existing_events": (
                counted_existing
            ),
            "submitted_pending_events": (
                submitted_pending
            ),
            "submitted_train_events": int(
                entry[
                    "submitted_train_events"
                ]
            ),
            "submitted_validation_events": int(
                entry[
                    "submitted_validation_events"
                ]
            ),
            "submitted_pending_counted_events": (
                submitted_pending_counted
            ),
            "projected_counted_after_success": (
                projected_counted
            ),
            "training_authorized": (
                training_authorized
            ),
            "physics_yield_authorized": (
                physics_authorized
            ),
            "overlap_group": overlap_group,
            "normalization_rule": (
                normalization_rule
            ),
            "campaign_count": len({
                value
                for value in entry[
                    "campaigns"
                ]
                if value
            }),
            "accounting_classes": ",".join(
                sorted({
                    value
                    for value in entry[
                        "accounting_classes"
                    ]
                    if value
                })
            ),
            "notes": "; ".join(
                note
                for note in notes
                if note
            ),
        })

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
            fieldnames=list(
                records[0]
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(records)

    receipt_backed_events = sum(
        row[
            "existing_successful_events"
        ]
        for row in records
    )

    receipt_backed_candidates = sum(
        row[
            "existing_candidate_rows"
        ]
        for row in records
    )

    native_counted_events = sum(
        row[
            "native_counted_events"
        ]
        for row in records
    )

    promoted_counted_events = sum(
        row[
            "promoted_counted_events"
        ]
        for row in records
    )

    counted_existing_events = sum(
        row[
            "counted_existing_events"
        ]
        for row in records
    )

    submitted_pending_events = sum(
        row[
            "submitted_pending_events"
        ]
        for row in records
    )

    submitted_pending_counted_events = sum(
        row[
            "submitted_pending_counted_events"
        ]
        for row in records
    )

    projected_counted_events = (
        counted_existing_events
        + submitted_pending_counted_events
    )

    target_events = 5_000_000

    remaining_after_success = max(
        0,
        target_events
        - projected_counted_events,
    )

    summary = {
        "schema_version": 1,
        "snapshot_status": (
            "scale400k_submitted_pending_receipts"
        ),
        "family_records": len(
            records
        ),
        "target_events": (
            target_events
        ),
        "receipt_backed_successful_events": (
            receipt_backed_events
        ),
        "receipt_backed_candidate_rows": (
            receipt_backed_candidates
        ),
        "native_counted_events": (
            native_counted_events
        ),
        "promoted_counted_events": (
            promoted_counted_events
        ),
        "counted_existing_events": (
            counted_existing_events
        ),
        "submitted_pending_events": (
            submitted_pending_events
        ),
        "submitted_pending_counted_events": (
            submitted_pending_counted_events
        ),
        "projected_counted_after_success": (
            projected_counted_events
        ),
        "remaining_after_success": (
            remaining_after_success
        ),
        "important_caveats": [
            (
                "Submitted events are not counted as "
                "completed until receipt and EOS audits pass."
            ),
            (
                "Physics-yield authorization is separate "
                "from training authorization."
            ),
            (
                "Manual signal and diagnostic entries "
                "require exact lineage discovery."
            ),
            (
                "No final-test production is authorized."
            ),
        ],
        "registry_tsv": str(
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

    output_markdown.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown = [
        "# HH→4b Dataset Registry Snapshot",
        "",
        "Snapshot date: 2026-07-21",
        "",
        "## Accounting summary",
        "",
        (
            f"- Receipt-backed successful events: "
            f"{receipt_backed_events:,}"
        ),
        (
            f"- Native counted events: "
            f"{native_counted_events:,}"
        ),
        (
            f"- Promoted legacy counted events: "
            f"{promoted_counted_events:,}"
        ),
        (
            f"- Counted completed events after overlay: "
            f"{counted_existing_events:,}"
        ),
        (
            f"- Submitted events pending validation: "
            f"{submitted_pending_events:,}"
        ),
        (
            f"- Projected counted events after successful "
            f"scale-out validation: "
            f"{projected_counted_events:,}"
        ),
        (
            f"- Provisional remaining events to 5M: "
            f"{remaining_after_success:,}"
        ),
        "",
        (
            "Submitted events remain pending and are not "
            "classified as completed in this snapshot."
        ),
        "",
        "## Sample-family registry",
        "",
        (
            "| Family | Display name | Role | Group | "
            "Existing successful | Counted completed | "
            "Submitted pending | Projected counted | "
            "Candidates | Status | Training | Physics yields |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|"
            "---:|---|---|---|"
        ),
    ]

    for row in records:
        markdown.append(
            "| "
            + " | ".join([
                markdown_escape(
                    row["family"]
                ),
                markdown_escape(
                    row["display_name"]
                ),
                markdown_escape(
                    row["role"]
                ),
                markdown_escape(
                    row["process_group"]
                ),
                f"{row['existing_successful_events']:,}",
                f"{row['counted_existing_events']:,}",
                f"{row['submitted_pending_events']:,}",
                (
                    f"{row['projected_counted_after_success']:,}"
                ),
                f"{row['existing_candidate_rows']:,}",
                markdown_escape(
                    row["dataset_status"]
                ),
                markdown_escape(
                    row["training_authorized"]
                ),
                markdown_escape(
                    row["physics_yield_authorized"]
                ),
            ])
            + " |"
        )

    markdown.extend([
        "",
        "## Interpretation rules",
        "",
        (
            "- `receipt_backed_existing` requires exact "
            "LHE, HepMC, ROOT, receipt, and EOS lineage."
        ),
        (
            "- `submitted_pending_receipt` is planned or "
            "running production and is not yet completed."
        ),
        (
            "- Training authorization does not imply "
            "physical-yield authorization."
        ),
        (
            "- `REVIEW_REQUIRED` entries must not be used "
            "silently in final training or inference."
        ),
        (
            "- Final-test production remains unauthorized."
        ),
        "",
        "## Provenance",
        "",
        f"- Inventory: `{inventory_path.name}`",
        f"- Contract: `{contract_path.name}`",
        f"- Promotion overlay: `{overlay_path.name}`",
        f"- Scale manifest: `{scale_manifest_path.name}`",
        f"- Wave-B plan: `{waveb_plan_path.name}`",
        f"- Manual catalog: `{manual_catalog_path.name}`",
        "",
    ])

    output_markdown.write_text(
        "\n".join(markdown)
        + "\n"
    )

    print(
        f"registry_families={len(records)}"
    )

    print(
        "receipt_backed_successful_events="
        f"{receipt_backed_events}"
    )

    print(
        "native_counted_events="
        f"{native_counted_events}"
    )

    print(
        "promoted_counted_events="
        f"{promoted_counted_events}"
    )

    print(
        "counted_existing_events="
        f"{counted_existing_events}"
    )

    print(
        "submitted_pending_events="
        f"{submitted_pending_events}"
    )

    print(
        "projected_counted_after_success="
        f"{projected_counted_events}"
    )

    print(
        "remaining_after_success="
        f"{remaining_after_success}"
    )

    print(f"registry_tsv={output_tsv}")
    print(
        f"registry_markdown={output_markdown}"
    )
    print(f"summary_json={output_json}")
    print(
        "HH4B_DATASET_REGISTRY_SNAPSHOT_VALID"
    )


if __name__ == "__main__":
    main()
