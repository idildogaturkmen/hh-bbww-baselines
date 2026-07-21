#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import argparse
import csv


REQUIRED_COLUMNS = {
    "family",
    "group",
    "events",
    "seed",
    "dataset_split",
    "sampling_mode",
    "forced_decay",
    "generator_enrichment",
    "run_card_profile",
    "dataset_role",
    "overlap_group",
    "normalization_rule",
    "production_status",
}

ALLOWED_MODES = {
    "inclusive_reference",
    "forced_decay_reference",
    "exclusive_hf_stratum",
    "hf_me_enrichment",
}

ALLOWED_ROLES = {
    "physical_reference",
    "physical_stratum",
    "ml_enrichment",
}

PROHIBITED_ENRICHMENT_TOKENS = {
    "m_bb",
    "mbb",
    "m_hh",
    "mhh",
    "r_hh",
    "rhh",
    "classifier",
    "score",
    "signal_region",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("policy")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    policy_path = Path(args.policy)

    if not plan_path.is_file():
        raise SystemExit(
            f"ERROR: missing plan: {plan_path}"
        )

    if not policy_path.is_file():
        raise SystemExit(
            f"ERROR: missing policy: {policy_path}"
        )

    with plan_path.open(newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing = REQUIRED_COLUMNS - fieldnames

    if missing:
        raise SystemExit(
            f"ERROR: missing columns: {sorted(missing)}"
        )

    if len(rows) != 16:
        raise SystemExit(
            f"ERROR: expected 16 rows, found {len(rows)}"
        )

    if sum(int(row["events"]) for row in rows) != 160_000:
        raise SystemExit(
            "ERROR: expected 160000 pilot events"
        )

    seeds = [
        int(row["seed"])
        for row in rows
    ]

    if len(set(seeds)) != 16:
        raise SystemExit(
            "ERROR: seeds are not unique"
        )

    families = [
        row["family"]
        for row in rows
    ]

    if len(set(families)) != 16:
        raise SystemExit(
            "ERROR: family names are not unique"
        )

    split_counts = Counter(
        row["dataset_split"]
        for row in rows
    )

    if split_counts != {
        "train": 12,
        "validation": 4,
    }:
        raise SystemExit(
            f"ERROR: wrong split counts: {split_counts}"
        )

    group_counts = Counter(
        row["group"]
        for row in rows
    )

    expected_groups = {
        "single_higgs": 4,
        "single_top": 5,
        "rare_top": 4,
        "diboson_triboson": 3,
    }

    if dict(group_counts) != expected_groups:
        raise SystemExit(
            f"ERROR: wrong group counts: {group_counts}"
        )

    for row in rows:
        if row["sampling_mode"] not in ALLOWED_MODES:
            raise SystemExit(
                "ERROR: unsupported sampling mode for "
                f"{row['family']}: "
                f"{row['sampling_mode']}"
            )

        if row["dataset_role"] not in ALLOWED_ROLES:
            raise SystemExit(
                "ERROR: unsupported dataset role for "
                f"{row['family']}: "
                f"{row['dataset_role']}"
            )

        if row["run_card_profile"] != "preserve_template":
            raise SystemExit(
                "ERROR: missing-family templates must use "
                f"preserve_template: {row['family']}"
            )

        enrichment = row[
            "generator_enrichment"
        ].lower()

        for token in PROHIBITED_ENRICHMENT_TOKENS:
            if token in enrichment:
                raise SystemExit(
                    "ERROR: prohibited analysis-variable "
                    f"enrichment in {row['family']}: "
                    f"{token}"
                )

        if (
            row["dataset_role"] == "ml_enrichment"
            and "stitch" not in row["normalization_rule"]
        ):
            raise SystemExit(
                "ERROR: ML-enrichment row lacks a "
                f"stitching restriction: {row['family']}"
            )

    policy = policy_path.read_text().lower()

    required_policy_phrases = [
        "branching fraction",
        "overlap",
        "stitch",
        "do not select",
        "total generated events",
    ]

    for phrase in required_policy_phrases:
        if phrase not in policy:
            raise SystemExit(
                "ERROR: policy is missing phrase: "
                f"{phrase}"
            )

    print(f"pilot_rows={len(rows)}")
    print(
        "pilot_events="
        f"{sum(int(row['events']) for row in rows)}"
    )
    print(f"unique_seeds={len(set(seeds))}")
    print(f"split_counts={dict(split_counts)}")
    print(f"group_counts={dict(group_counts)}")
    print(
        "sampling_modes="
        f"{dict(Counter(row['sampling_mode'] for row in rows))}"
    )
    print(
        "dataset_roles="
        f"{dict(Counter(row['dataset_role'] for row in rows))}"
    )
    print("WAVEB_IMPORTANCE_PLAN_VALID")


if __name__ == "__main__":
    main()
