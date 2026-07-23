#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


EXPECTED_BACKGROUND_EVENTS = 5_000_000
EXPECTED_BACKGROUND_MEMBERS = 520

EXPECTED_SPLIT_EVENTS = {
    "train": 3_791_373,
    "validation": 1_024_084,
    "test": 184_543,
}

EXPECTED_COMPONENT_EVENTS = {
    "qcd": 2_510_000,
    "ttbar": 270_000,
    "wavea": 1_300_000,
    "waveb": 170_000,
    "wavec1": 220_000,
    "single_top": 330_000,
    "hbb": 150_000,
    "triboson": 50_000,
}

COMPONENT_ORDER = [
    "qcd",
    "ttbar",
    "wavea",
    "waveb",
    "wavec1",
    "single_top",
    "hbb",
    "triboson",
]

COMPONENT_DESCRIPTIONS = {
    "qcd":
        "Adaptive HardQCD importance-sampled production",
    "ttbar":
        "Canonical subset of legacy inclusive ttbar production",
    "wavea":
        "QCD bbbb, inclusive ttbar and Zbbbb Wave-A production",
    "waveb":
        "ttH(bb), ttZ(bb), tttt and VBF H(bb) Wave-B subset",
    "wavec1":
        "ttW, WH(bb), WW and WZ(bb) Wave-C1 production",
    "single_top":
        "Single-top pilot and scale-out production",
    "hbb":
        "ggH(bb) and bbH(bb) background production",
    "triboson":
        "WWZ, WZZ and ZZZ with direct-Z→bb truth filtering",
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def as_int(value: Any) -> int:
    return int(
        float(
            str(value).strip()
        )
    )


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


def read_tsv(
    path: Path,
) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"missing TSV {path}",
    )

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


def walk(value: Any) -> Iterator[Any]:
    yield value

    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)

    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def markdown_table(
    headers: list[str],
    rows: list[list[Any]],
) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            "---"
            for _ in headers
        ) + " |",
    ]

    for row in rows:
        output.append(
            "| "
            + " | ".join(
                str(value)
                for value in row
            )
            + " |"
        )

    return output


def find_vbf_audit(
    directory: Path,
) -> Path:
    require(
        directory.is_dir(),
        f"missing VBF audit directory {directory}",
    )

    matches: list[Path] = []

    for path in sorted(
        directory.glob("*.json")
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

        integer_values = {
            value
            for value in walk(document)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
            )
        }

        if (
            document.get("status") == "pass"
            and 100_000 in integer_values
        ):
            matches.append(path)

    require(
        matches,
        (
            "could not locate a passing VBF "
            f"100k audit under {directory}"
        ),
    )

    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--background-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--background-tsv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-tsv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vbf-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    background_json_path = (
        arguments.background_json.resolve()
    )
    background_tsv_path = (
        arguments.background_tsv.resolve()
    )
    source_json_path = (
        arguments.source_json.resolve()
    )
    source_tsv_path = (
        arguments.source_tsv.resolve()
    )
    vbf_dir = arguments.vbf_dir.resolve()
    output_path = arguments.output.resolve()

    for path in (
        background_json_path,
        background_tsv_path,
        source_json_path,
        source_tsv_path,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        not output_path.exists(),
        f"refusing to overwrite {output_path}",
    )

    background = json.loads(
        background_json_path.read_text()
    )

    source_audit = json.loads(
        source_json_path.read_text()
    )

    registry = read_tsv(
        background_tsv_path
    )

    source_summary = read_tsv(
        source_tsv_path
    )

    require(
        background.get("status")
        == "pass",
        "unified background audit did not pass",
    )

    require(
        background.get(
            "unified_background_5m_registry_valid"
        )
        is True,
        "unified background registry is invalid",
    )

    require(
        background.get(
            "unified_background_5m_membership_frozen"
        )
        is True,
        "unified background membership is not frozen",
    )

    require(
        background.get(
            "canonical_background_events"
        )
        == EXPECTED_BACKGROUND_EVENTS,
        "canonical background total is not 5M",
    )

    require(
        background.get(
            "canonical_background_members"
        )
        == EXPECTED_BACKGROUND_MEMBERS,
        (
            "canonical background member "
            "count is not 520"
        ),
    )

    for split, expected in (
        EXPECTED_SPLIT_EVENTS.items()
    ):
        require(
            background.get(
                f"canonical_{split}_events"
            )
            == expected,
            (
                f"unexpected canonical {split} "
                "event total"
            ),
        )

    require(
        source_audit.get("status")
        == "pass",
        "classifier-source audit did not pass",
    )

    require(
        source_audit.get(
            "background_5m_classifier_source_manifest_valid"
        )
        is True,
        "classifier-source manifest is invalid",
    )

    require(
        source_audit.get(
            "generated_events"
        )
        == EXPECTED_BACKGROUND_EVENTS,
        "classifier-source total is not 5M",
    )

    require(
        source_audit.get(
            "members"
        )
        == EXPECTED_BACKGROUND_MEMBERS,
        (
            "classifier-source member "
            "count is not 520"
        ),
    )

    require(
        sha256_file(
            background_tsv_path
        )
        == background[
            "registry_sha256"
        ],
        "unified registry SHA-256 mismatch",
    )

    source_manifest_path = Path(
        source_audit[
            "source_manifest"
        ]
    )

    require(
        source_manifest_path.is_file(),
        (
            "classifier-source manifest "
            "is missing"
        ),
    )

    require(
        sha256_file(
            source_manifest_path
        )
        == source_audit[
            "source_manifest_sha256"
        ],
        (
            "classifier-source manifest "
            "SHA-256 mismatch"
        ),
    )

    require(
        len(registry)
        == EXPECTED_BACKGROUND_MEMBERS,
        "unified registry does not have 520 rows",
    )

    component_events = Counter()
    component_members = Counter()
    component_split_events = Counter()

    family_events = Counter()
    family_members = Counter()
    family_split_events = Counter()

    canonical_identities: set[str] = set()

    for row in registry:
        component = row["component"]
        family = row["family"]
        split = row["dataset_split"]
        events = as_int(
            row["events"]
        )
        identity = row[
            "canonical_identity"
        ]

        require(
            component
            in EXPECTED_COMPONENT_EVENTS,
            f"unexpected component {component}",
        )

        require(
            split
            in {
                "train",
                "validation",
                "test",
            },
            f"unexpected split {split}",
        )

        require(
            identity
            not in canonical_identities,
            (
                "duplicate canonical identity "
                f"{identity}"
            ),
        )

        canonical_identities.add(identity)

        component_events[
            component
        ] += events

        component_members[
            component
        ] += 1

        component_split_events[
            (
                component,
                split,
            )
        ] += events

        family_events[
            (
                component,
                family,
            )
        ] += events

        family_members[
            (
                component,
                family,
            )
        ] += 1

        family_split_events[
            (
                component,
                family,
                split,
            )
        ] += events

    require(
        dict(component_events)
        == EXPECTED_COMPONENT_EVENTS,
        (
            "component totals differ from "
            "the canonical definition: "
            f"{dict(component_events)}"
        ),
    )

    require(
        sum(
            component_events.values()
        )
        == EXPECTED_BACKGROUND_EVENTS,
        "registry event sum is not 5M",
    )

    require(
        len(canonical_identities)
        == EXPECTED_BACKGROUND_MEMBERS,
        (
            "canonical identity count "
            "is not 520"
        ),
    )

    availability_by_component: dict[
        str,
        dict[str, int],
    ] = {}

    for row in source_summary:
        component = row["component"]

        availability_by_component[
            component
        ] = {
            "candidate_rows":
                as_int(
                    row["candidate_rows"]
                ),
            "local_members":
                as_int(
                    row["local_members"]
                ),
            "remote_members":
                as_int(
                    row["remote_members"]
                ),
        }

    require(
        set(
            availability_by_component
        )
        == set(COMPONENT_ORDER),
        (
            "classifier-source summary "
            "has unexpected components"
        ),
    )

    local_total = sum(
        row["local_members"]
        for row
        in availability_by_component.values()
    )

    remote_total = sum(
        row["remote_members"]
        for row
        in availability_by_component.values()
    )

    require(
        local_total == 27,
        (
            "local candidate-source "
            f"total is {local_total}, not 27"
        ),
    )

    require(
        remote_total == 493,
        (
            "remote candidate-source "
            f"total is {remote_total}, not 493"
        ),
    )

    vbf_audit_path = find_vbf_audit(
        vbf_dir
    )

    git_head = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    component_rows: list[
        list[Any]
    ] = []

    for component in COMPONENT_ORDER:
        availability = (
            availability_by_component[
                component
            ]
        )

        component_rows.append([
            component,
            COMPONENT_DESCRIPTIONS[
                component
            ],
            f"{component_events[component]:,}",
            component_members[component],
            (
                f"{component_split_events[(component, 'train')]:,}"
            ),
            (
                f"{component_split_events[(component, 'validation')]:,}"
            ),
            (
                f"{component_split_events[(component, 'test')]:,}"
            ),
            (
                f"{availability['candidate_rows']:,}"
            ),
        ])

    family_rows: list[
        list[Any]
    ] = []

    for component in COMPONENT_ORDER:
        families = sorted(
            family
            for (
                current_component,
                family,
            ) in family_events
            if current_component
            == component
        )

        for family in families:
            family_rows.append([
                component,
                family,
                (
                    f"{family_events[(component, family)]:,}"
                ),
                family_members[
                    (
                        component,
                        family,
                    )
                ],
                (
                    f"{family_split_events[(component, family, 'train')]:,}"
                ),
                (
                    f"{family_split_events[(component, family, 'validation')]:,}"
                ),
                (
                    f"{family_split_events[(component, family, 'test')]:,}"
                ),
            ])

    availability_rows: list[
        list[Any]
    ] = []

    for component in COMPONENT_ORDER:
        availability = (
            availability_by_component[
                component
            ]
        )

        availability_rows.append([
            component,
            availability[
                "local_members"
            ],
            availability[
                "remote_members"
            ],
            (
                f"{availability['candidate_rows']:,}"
            ),
        ])

    lines: list[str] = []

    def add(*values: str) -> None:
        lines.extend(values)

    add(
        "# Canonical Run-2 13 TeV Delphes HH→4b Dataset",
        "",
        "## Scope",
        "",
        (
            "This dataset card documents the immutable "
            "Delphes samples used for the HH→4b "
            "machine-learning baseline study."
        ),
        "",
        (
            "The canonical background contains exactly "
            "5,000,000 generated events in 520 frozen "
            "members. Signal samples are counted separately "
            "and are not included in the 5M background."
        ),
        "",
        (
            "The dataset is registry-driven and remains "
            "partitioned by shard. The ROOT files must not "
            "be combined into one giant merged ROOT file."
        ),
        "",
        "## Canonical background status",
        "",
        "- Generated background events: **5,000,000**",
        "- Canonical background members: **520**",
        "- Train events: **3,791,373**",
        "- Validation events: **1,024,084**",
        "- Test events: **184,543**",
        "- Membership: **validated and frozen**",
        "- Classifier-source manifest: **validated**",
        "- Classifier-table materialization: **authorized**",
        "- Physics weights: **not yet frozen**",
        "- Physics-yield normalization: **not yet authorized**",
        "",
        (
            "Generated-event counts and candidate-row counts "
            "are different quantities. The 5,000,000 total "
            "always refers to generated canonical events."
        ),
        "",
        "## Background composition by canonical component",
        "",
    )

    lines.extend(
        markdown_table(
            [
                "Component",
                "Description",
                "Events",
                "Members",
                "Train",
                "Validation",
                "Test",
                "Candidate rows",
            ],
            component_rows,
        )
    )

    add(
        "",
        "## Detailed background composition by process family",
        "",
    )

    lines.extend(
        markdown_table(
            [
                "Component",
                "Process family",
                "Events",
                "Members",
                "Train",
                "Validation",
                "Test",
            ],
            family_rows,
        )
    )

    add(
        "",
        "## Candidate-source availability",
        "",
        (
            "The classifier-source audit found 27 canonical "
            "members with an existing local candidate Parquet "
            "and 493 members whose candidate Parquet must be "
            "extracted from an audited EOS bundle."
        ),
        "",
    )

    lines.extend(
        markdown_table(
            [
                "Component",
                "Local candidate Parquets",
                "EOS bundle extractions",
                "Candidate rows",
            ],
            availability_rows,
        )
    )

    add(
        "",
        "## Signal samples",
        "",
    )

    lines.extend(
        markdown_table(
            [
                "Signal process",
                "Canonical events",
                "Train",
                "Validation",
                "Test",
                "Status",
            ],
            [
                [
                    "VBF HH→bbbb",
                    "100,000",
                    "80,000",
                    "20,000",
                    "0",
                    (
                        "Canonical membership validated and "
                        "frozen; physics-yield normalization "
                        "is not yet authorized."
                    ),
                ],
                [
                    "ggF HH→bbbb",
                    "Not yet frozen",
                    "—",
                    "—",
                    "—",
                    (
                        "Production artifacts exist, but "
                        "canonical ggF membership and duplicate "
                        "handling require a separate audit."
                    ),
                ],
            ],
        )
    )

    add(
        "",
        (
            "The VBF signal count is supported by the passing "
            "canonical 100k registry audit. No publishable ggF "
            "count is assigned until its canonical registry is "
            "reconciled and frozen."
        ),
        "",
        "## Canonical background accounting",
        "",
        "```text",
        (
            "Adaptive HardQCD importance production       "
            "2,510,000"
        ),
        (
            "Legacy inclusive ttbar canonical subset        "
            "270,000"
        ),
        (
            "Wave-A audited production                    "
            "1,300,000"
        ),
        (
            "Wave-B deterministic canonical subset          "
            "170,000"
        ),
        (
            "Wave-C1 audited production                     "
            "220,000"
        ),
        (
            "Single-top pilot and scale-out                  "
            "330,000"
        ),
        (
            "H(bb) backgrounds                              "
            "150,000"
        ),
        (
            "Canonical triboson                              "
            "50,000"
        ),
        (
            "                                              "
            "---------"
        ),
        (
            "Total                                        "
            "5,000,000"
        ),
        "```",
        "",
        "## Required accounting and reuse conventions",
        "",
        (
            "1. QA canaries, smoke tests, malformed jobs and "
            "failed jobs contribute zero canonical events."
        ),
        (
            "2. Frozen train, validation and test assignments "
            "must be preserved. The QCD test split must not be "
            "used for training, hyperparameter selection or "
            "threshold optimization."
        ),
        (
            "3. Candidate rows are selected analysis "
            "candidates and must not be reported as "
            "generated-event counts."
        ),
        (
            "4. `physics_yield_authorized=false` means final "
            "normalization has not yet been frozen; it does "
            "not invalidate the events."
        ),
        (
            "5. Importance-sampled, enriched and truth-filtered "
            "samples require process-specific normalization."
        ),
        (
            "6. Triboson normalization must apply the "
            "direct-Z→bb branching/filter factor exactly once."
        ),
        (
            "7. Final HH signal normalization must use the "
            "separately frozen official SM production cross "
            "sections and branching fractions, not raw "
            "generator estimates."
        ),
        "",
        "## Authoritative artifacts",
        "",
        (
            f"- Repository commit at generation: "
            f"`{git_head}`"
        ),
        (
            f"- Unified registry: "
            f"`{background_tsv_path}`"
        ),
        (
            "- Unified registry SHA-256: "
            f"`{background['registry_sha256']}`"
        ),
        (
            f"- Unified registry audit: "
            f"`{background_json_path}`"
        ),
        (
            "- Unified registry audit SHA-256: "
            f"`{sha256_file(background_json_path)}`"
        ),
        (
            f"- Classifier-source manifest: "
            f"`{source_manifest_path}`"
        ),
        (
            "- Classifier-source manifest SHA-256: "
            f"`{source_audit['source_manifest_sha256']}`"
        ),
        (
            f"- Classifier-source audit: "
            f"`{source_json_path}`"
        ),
        (
            "- Classifier-source audit SHA-256: "
            f"`{sha256_file(source_json_path)}`"
        ),
        (
            f"- VBF canonical audit: "
            f"`{vbf_audit_path}`"
        ),
        (
            "- VBF canonical audit SHA-256: "
            f"`{sha256_file(vbf_audit_path)}`"
        ),
        "",
        "## Reproducibility",
        "",
        (
            "Classifier-table materialization must retain the "
            "canonical component, process family, source "
            "campaign, target tag, shard identity, seed and "
            "frozen dataset split for every member."
        ),
        "",
        (
            "This card documents membership and event "
            "accounting. Final physics weights and yield "
            "normalization constants belong in a separate, "
            "versioned normalization freeze."
        ),
        "",
        (
            "Dataset card generated at: "
            f"`{datetime.now(timezone.utc).isoformat()}`"
        ),
        "",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    require(
        output_path.is_file()
        and output_path.stat().st_size > 0,
        "dataset card was not written",
    )

    print(
        "CANONICAL_BACKGROUND_5M_DATASET_CARD_WRITTEN"
    )
    print(
        "BACKGROUND_EVENTS=5000000"
    )
    print(
        "BACKGROUND_MEMBERS=520"
    )
    print(
        "BACKGROUND_TRAIN_EVENTS=3791373"
    )
    print(
        "BACKGROUND_VALIDATION_EVENTS=1024084"
    )
    print(
        "BACKGROUND_TEST_EVENTS=184543"
    )
    print(
        "SIGNAL_VBF_EVENTS=100000"
    )
    print(
        "SIGNAL_GGF_STATUS=PENDING_CANONICAL_RECONCILIATION"
    )
    print(
        f"dataset_card={output_path}"
    )


if __name__ == "__main__":
    main()
