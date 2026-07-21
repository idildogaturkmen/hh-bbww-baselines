#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_FAMILY = "ttbar_inclusive"

EXPECTED_CAMPAIGNS = {
    "unified_background_5m_wavea_pilots_20260720_retry1": {
        "rows": 1,
        "events": 10000,
        "split_counts": {
            "train": 1,
        },
    },
    "unified_background_5m_wavea_proven111_20260720_v1": {
        "rows": 74,
        "events": 740000,
        "split_counts": {
            "train": 59,
            "validation": 15,
        },
    },
}

EXPECTED_FAILED_CAMPAIGN = (
    "unified_background_5m_wavea_pilots_20260720"
)

EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

EXPECTED_PAYLOAD_GIT_HEAD = (
    "579322014ef2fd19c959e94a76ccc458d5d941af"
)

EXPECTED_ROWS = 75
EXPECTED_EVENTS_PER_ROW = 10000
EXPECTED_TOTAL_EVENTS = 750000

EXPECTED_SPLIT_COUNTS = {
    "train": 60,
    "validation": 15,
}


def as_int(
    value: object,
    default: int | None = None,
) -> int:
    text = str(
        value
        if value is not None
        else ""
    ).strip()

    if not text:
        if default is None:
            raise ValueError(
                "Required integer value is empty"
            )

        return default

    return int(text)


def is_exact_success(
    row: dict[str, str],
) -> bool:
    expected_events = as_int(
        row.get("events"),
        0,
    )

    return (
        row.get("stage")
        == "complete_copied_and_verified"
        and as_int(
            row.get("exit_status"),
            -1,
        )
        == 0
        and expected_events > 0
        and as_int(
            row.get("lhe_events"),
            -1,
        )
        == expected_events
        and as_int(
            row.get("hepmc_events"),
            -1,
        )
        == expected_events
        and as_int(
            row.get("root_events"),
            -1,
        )
        == expected_events
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory",
        required=True,
    )

    parser.add_argument(
        "--ledger",
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

    ledger_path = Path(
        args.ledger
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    if not inventory_path.is_file():
        raise SystemExit(
            f"ERROR: missing inventory: {inventory_path}"
        )

    if not ledger_path.is_file():
        raise SystemExit(
            f"ERROR: missing ledger: {ledger_path}"
        )

    with inventory_path.open(
        newline=""
    ) as handle:
        inventory = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    with ledger_path.open(
        newline=""
    ) as handle:
        ledger = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    ledger_by_key = {}

    for row in ledger:
        key = (
            row["campaign"],
            row["target_tag"],
        )

        if key in ledger_by_key:
            raise SystemExit(
                "ERROR: duplicate campaign/target_tag "
                f"in ledger: {key}"
            )

        ledger_by_key[key] = row

    all_ttbar_rows = [
        row
        for row in inventory
        if row.get("family")
        == EXPECTED_FAMILY
    ]

    selected_rows = [
        row
        for row in all_ttbar_rows
        if row.get("campaign")
        in EXPECTED_CAMPAIGNS
    ]

    excluded_rows = [
        row
        for row in all_ttbar_rows
        if row.get("campaign")
        not in EXPECTED_CAMPAIGNS
    ]

    errors: list[str] = []

    if len(selected_rows) != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} selected rows, "
            f"found {len(selected_rows)}"
        )

    total_events = sum(
        as_int(
            row.get("events"),
            0,
        )
        for row in selected_rows
    )

    if total_events != EXPECTED_TOTAL_EVENTS:
        errors.append(
            f"expected {EXPECTED_TOTAL_EVENTS} events, "
            f"found {total_events}"
        )

    if any(
        as_int(
            row.get("events"),
            0,
        )
        != EXPECTED_EVENTS_PER_ROW
        for row in selected_rows
    ):
        errors.append(
            "not every selected shard contains 10000 events"
        )

    selected_split_counts = Counter(
        row.get(
            "dataset_split",
            "",
        )
        for row in selected_rows
    )

    if (
        dict(selected_split_counts)
        != EXPECTED_SPLIT_COUNTS
    ):
        errors.append(
            "wrong combined split counts: "
            f"{dict(selected_split_counts)}"
        )

    rows_by_campaign = defaultdict(list)

    for row in selected_rows:
        rows_by_campaign[
            row["campaign"]
        ].append(row)

    campaign_summary = {}

    for campaign, expected in EXPECTED_CAMPAIGNS.items():
        campaign_rows = rows_by_campaign[
            campaign
        ]

        campaign_events = sum(
            as_int(
                row.get("events"),
                0,
            )
            for row in campaign_rows
        )

        campaign_split_counts = Counter(
            row.get(
                "dataset_split",
                "",
            )
            for row in campaign_rows
        )

        campaign_summary[campaign] = {
            "rows": len(
                campaign_rows
            ),
            "events": campaign_events,
            "split_counts": dict(
                sorted(
                    campaign_split_counts.items()
                )
            ),
        }

        if (
            len(campaign_rows)
            != expected["rows"]
        ):
            errors.append(
                f"{campaign}: expected "
                f"{expected['rows']} rows, "
                f"found {len(campaign_rows)}"
            )

        if (
            campaign_events
            != expected["events"]
        ):
            errors.append(
                f"{campaign}: expected "
                f"{expected['events']} events, "
                f"found {campaign_events}"
            )

        if (
            dict(campaign_split_counts)
            != expected["split_counts"]
        ):
            errors.append(
                f"{campaign}: wrong split counts: "
                f"{dict(campaign_split_counts)}"
            )

    failed_campaign_rows = [
        row
        for row in excluded_rows
        if row.get("campaign")
        == EXPECTED_FAILED_CAMPAIGN
    ]

    if len(failed_campaign_rows) != 1:
        errors.append(
            "expected exactly one excluded failed "
            f"pilot row, found {len(failed_campaign_rows)}"
        )
    elif is_exact_success(
        failed_campaign_rows[0]
    ):
        errors.append(
            "excluded failed pilot unexpectedly "
            "passes the exact-success checks"
        )

    unexpected_excluded_campaigns = sorted({
        row.get(
            "campaign",
            "",
        )
        for row in excluded_rows
        if row.get("campaign")
        != EXPECTED_FAILED_CAMPAIGN
    })

    if unexpected_excluded_campaigns:
        errors.append(
            "unexpected excluded ttbar campaigns: "
            + ", ".join(
                unexpected_excluded_campaigns
            )
        )

    target_tags = [
        row["target_tag"]
        for row in selected_rows
    ]

    remote_bundles = [
        row["remote_bundle"]
        for row in selected_rows
    ]

    receipt_paths = [
        row["receipt"]
        for row in selected_rows
    ]

    if len(set(target_tags)) != len(target_tags):
        errors.append(
            "duplicate target tags across the "
            "selected successful campaigns"
        )

    if (
        len(set(remote_bundles))
        != len(remote_bundles)
    ):
        errors.append(
            "duplicate EOS bundles across the "
            "selected successful campaigns"
        )

    if (
        len(set(receipt_paths))
        != len(receipt_paths)
    ):
        errors.append(
            "duplicate receipt paths across the "
            "selected successful campaigns"
        )

    payload_heads = {
        row.get(
            "payload_git_head",
            "",
        )
        for row in selected_rows
    }

    if payload_heads != {
        EXPECTED_PAYLOAD_GIT_HEAD
    }:
        errors.append(
            "unexpected payload heads: "
            f"{sorted(payload_heads)}"
        )

    card_hashes = {
        row.get(
            "delphes_card_sha256",
            "",
        )
        for row in selected_rows
    }

    if card_hashes != {
        EXPECTED_CARD_SHA256
    }:
        errors.append(
            "unexpected Delphes-card hashes: "
            f"{sorted(card_hashes)}"
        )

    candidate_rows = 0
    generator_xsecs = []

    for row in selected_rows:
        campaign = row["campaign"]
        target_tag = row["target_tag"]
        expected_events = as_int(
            row.get("events"),
            0,
        )

        checks = {
            "family": (
                row.get("family")
                == EXPECTED_FAMILY
            ),
            "campaign_selected": (
                campaign
                in EXPECTED_CAMPAIGNS
            ),
            "dataset_split": (
                row.get("dataset_split")
                in {
                    "train",
                    "validation",
                }
            ),
            "dataset_role": (
                row.get("dataset_role")
                == "physical_inference"
            ),
            "exact_success": (
                is_exact_success(row)
            ),
            "accounting_class": (
                row.get("accounting_class")
                == "valid_not_counted"
            ),
            "source_count_false": (
                row.get("count_toward_5m")
                == "False"
            ),
            "eos_bundle": (
                row.get(
                    "remote_bundle",
                    "",
                ).startswith(
                    "/store/user/"
                )
            ),
            "receipt_exists": (
                Path(
                    row.get(
                        "receipt",
                        "",
                    )
                ).is_file()
            ),
            "manifest_campaign_match": (
                campaign
                in row.get(
                    "manifest",
                    "",
                )
            ),
            "event_count": (
                expected_events
                == EXPECTED_EVENTS_PER_ROW
            ),
        }

        failed_checks = [
            key
            for key, passed
            in checks.items()
            if not passed
        ]

        if failed_checks:
            errors.append(
                f"{campaign}/{target_tag}: "
                + ", ".join(
                    failed_checks
                )
            )

        ledger_row = ledger_by_key.get(
            (
                campaign,
                target_tag,
            )
        )

        if ledger_row is None:
            errors.append(
                "missing ledger row: "
                f"{campaign}/{target_tag}"
            )
        else:
            ledger_checks = {
                "successful": (
                    ledger_row.get(
                        "successful"
                    )
                    == "True"
                ),
                "qa_only_false": (
                    ledger_row.get(
                        "qa_only"
                    )
                    == "False"
                ),
                "training_authorized": (
                    ledger_row.get(
                        "training_authorized"
                    )
                    == "True"
                ),
                "physics_not_authorized": (
                    ledger_row.get(
                        "physics_yield_authorized"
                    )
                    == "False"
                ),
                "not_currently_counted": (
                    ledger_row.get(
                        "count_toward_5m"
                    )
                    == "False"
                ),
                "w_phys_empty": (
                    ledger_row.get(
                        "w_phys",
                        "",
                    )
                    == ""
                ),
            }

            failed_ledger_checks = [
                key
                for key, passed
                in ledger_checks.items()
                if not passed
            ]

            if failed_ledger_checks:
                errors.append(
                    "ledger "
                    f"{campaign}/{target_tag}: "
                    + ", ".join(
                        failed_ledger_checks
                    )
                )

        candidate_rows += as_int(
            row.get(
                "candidate_rows",
                0,
            ),
            0,
        )

        xsec_text = str(
            row.get(
                "generator_xsec_pb",
                "",
            )
        ).strip()

        if not xsec_text:
            errors.append(
                "missing generator cross section: "
                f"{campaign}/{target_tag}"
            )
        else:
            xsec = float(
                xsec_text
            )

            if xsec <= 0:
                errors.append(
                    "nonpositive generator cross section: "
                    f"{campaign}/{target_tag}"
                )
            else:
                generator_xsecs.append(
                    xsec
                )

    promotion_candidate = (
        len(errors) == 0
    )

    summary = {
        "schema_version": 3,
        "family": EXPECTED_FAMILY,
        "selected_campaigns": sorted(
            EXPECTED_CAMPAIGNS
        ),
        "campaign_summary": (
            campaign_summary
        ),
        "promotion_candidate": (
            promotion_candidate
        ),
        "selected_rows": len(
            selected_rows
        ),
        "selected_events": (
            total_events
        ),
        "candidate_rows": (
            candidate_rows
        ),
        "combined_split_counts": dict(
            sorted(
                selected_split_counts.items()
            )
        ),
        "unique_target_tags": len(
            set(target_tags)
        ),
        "unique_remote_bundles": len(
            set(remote_bundles)
        ),
        "unique_receipts": len(
            set(receipt_paths)
        ),
        "payload_git_heads": sorted(
            payload_heads
        ),
        "delphes_card_sha256": sorted(
            card_hashes
        ),
        "generator_xsec_pb_min": (
            min(generator_xsecs)
            if generator_xsecs
            else None
        ),
        "generator_xsec_pb_max": (
            max(generator_xsecs)
            if generator_xsecs
            else None
        ),
        "excluded_failed_campaign": (
            EXPECTED_FAILED_CAMPAIGN
        ),
        "excluded_failed_rows": len(
            failed_campaign_rows
        ),
        "physics_yield_authorized": (
            False
        ),
        "training_authorized": (
            True
        ),
        "proposed_count_toward_5m": (
            promotion_candidate
        ),
        "errors": errors,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "selected_campaigns="
        f"{len(EXPECTED_CAMPAIGNS)}"
    )
    print(
        f"selected_ttbar_rows={len(selected_rows)}"
    )
    print(
        f"selected_ttbar_events={total_events}"
    )
    print(
        "selected_ttbar_candidate_rows="
        f"{candidate_rows}"
    )
    print(
        "combined_split_counts="
        f"{dict(sorted(selected_split_counts.items()))}"
    )
    print(
        "excluded_failed_rows="
        f"{len(failed_campaign_rows)}"
    )
    print(
        "unique_target_tags="
        f"{len(set(target_tags))}"
    )
    print(
        "unique_remote_bundles="
        f"{len(set(remote_bundles))}"
    )
    print(
        "unique_receipts="
        f"{len(set(receipt_paths))}"
    )
    print(
        "payload_git_heads="
        f"{len(payload_heads)}"
    )
    print(
        "card_hashes="
        f"{len(card_hashes)}"
    )
    print(
        "promotion_candidate="
        f"{promotion_candidate}"
    )
    print(f"audit={output_path}")

    if errors:
        for error in errors:
            print(
                f"ERROR: {error}"
            )

        raise SystemExit(2)

    print(
        "TTBAR_750K_LEGACY_PROMOTION_CANDIDATE_VALID"
    )


if __name__ == "__main__":
    main()
