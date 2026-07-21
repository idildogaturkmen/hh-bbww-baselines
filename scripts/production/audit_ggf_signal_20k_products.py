#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pyarrow.parquet as pq
import ROOT


EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

EXPECTED_V2_SCHEMA_SHA256 = (
    "ae7b765f343c3028db6571f3d53c0d46"
    "d6f4207806064132a07535ce25ed8ef8"
)

EXPECTED_LEGACY_SCHEMA_SHA256 = (
    "c4586f2f00b85912d4883a308572437d"
    "e18bc84de23721623141a1e2eea0f8dd"
)

EXPECTED_EVENT_SCHEMA_SHA256 = (
    "339e16b0411742326749e6e278440961"
    "f6623882e842c99ba66aea39ae2dec4e"
)


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


def schema_sha256(
    parquet: pq.ParquetFile,
) -> str:
    return hashlib.sha256(
        str(
            parquet.schema_arrow
        ).encode()
    ).hexdigest()


def exactly_one(
    paths: list[Path],
    label: str,
) -> Path:
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {label}, found {len(paths)}"
        )

    return paths[0]


def safe_extract(
    archive: Path,
    destination: Path,
) -> None:
    destination = destination.resolve()

    with tarfile.open(
        archive,
        "r:gz",
    ) as handle:
        for member in handle.getmembers():
            target = (
                destination
                / member.name
            ).resolve()

            if (
                target != destination
                and destination
                not in target.parents
            ):
                raise RuntimeError(
                    "unsafe archive member: "
                    f"{member.name}"
                )

        handle.extractall(
            destination
        )


def count_hepmc_events(
    path: Path,
) -> int:
    count = 0

    with path.open(
        "rb",
    ) as handle:
        for line in handle:
            if line.startswith(
                b"E "
            ):
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--temp-root",
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

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    ).resolve()

    temp_root = Path(
        args.temp_root
    ).resolve()

    output_tsv = Path(
        args.output_tsv
    ).resolve()

    output_json = Path(
        args.output_json
    ).resolve()

    with manifest_path.open(
        newline=""
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if len(rows) != 20:
        raise SystemExit(
            f"ERROR: expected 20 rows, found {len(rows)}"
        )

    shutil.rmtree(
        temp_root,
        ignore_errors=True,
    )

    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROOT.gROOT.SetBatch(True)

    records = []
    errors = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        local_shard = int(
            row["local_shard"]
        )

        inner_shard = int(
            row["inner_shard"]
        )

        seed = int(
            row["seed"]
        )

        expected_events = int(
            row["n_events"]
        )

        immutable_split = row[
            "immutable_split"
        ]

        remote_bundle = row[
            "remote_bundle"
        ]

        work_dir = (
            temp_root
            / f"shard_{local_shard:03d}"
        )

        bundle_path = (
            work_dir
            / "bundle.tar.gz"
        )

        outer_dir = (
            work_dir
            / "outer"
        )

        reconstruction_dir = (
            work_dir
            / "reconstruction"
        )

        work_dir.mkdir(
            parents=True
        )

        outer_dir.mkdir()
        reconstruction_dir.mkdir()

        print(
            f"[{index:02d}/20] "
            f"local_shard={local_shard} "
            f"split={immutable_split}"
        )

        try:
            source_url = (
                f"{args.eos_host}/"
                f"{remote_bundle}"
            )

            copy_result = subprocess.run(
                [
                    "xrdcp",
                    "--force",
                    source_url,
                    str(bundle_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )

            if copy_result.returncode != 0:
                raise RuntimeError(
                    "xrdcp failed: "
                    + copy_result.stderr.strip()
                )

            safe_extract(
                bundle_path,
                outer_dir,
            )

            hepmc_path = exactly_one(
                sorted(
                    outer_dir.rglob(
                        "*.hepmc"
                    )
                ),
                "HepMC file",
            )

            provenance_path = exactly_one(
                sorted(
                    outer_dir.rglob(
                        "*_provenance.json"
                    )
                ),
                "provenance JSON",
            )

            nested_archive = exactly_one(
                sorted(
                    outer_dir.rglob(
                        "*_reconstruction.tar.gz"
                    )
                ),
                "nested reconstruction archive",
            )

            provenance = json.loads(
                provenance_path.read_text()
            )

            provenance_checks = {
                "campaign": (
                    provenance.get(
                        "campaign"
                    )
                    == row["campaign"]
                ),
                "local_shard": (
                    int(
                        provenance.get(
                            "local_shard",
                            -1,
                        )
                    )
                    == local_shard
                ),
                "inner_shard": (
                    int(
                        provenance.get(
                            "inner_shard",
                            -1,
                        )
                    )
                    == inner_shard
                ),
                "seed": (
                    int(
                        provenance.get(
                            "seed",
                            -1,
                        )
                    )
                    == seed
                ),
                "n_events": (
                    int(
                        provenance.get(
                            "n_events",
                            -1,
                        )
                    )
                    == expected_events
                ),
                "delphes_card": (
                    provenance.get(
                        "delphes_card_sha256"
                    )
                    == EXPECTED_CARD_SHA256
                ),
                "hepmc_sha256": (
                    sha256_file(
                        hepmc_path
                    )
                    == provenance.get(
                        "hepmc_sha256"
                    )
                ),
                "reconstruction_sha256": (
                    sha256_file(
                        nested_archive
                    )
                    == provenance.get(
                        "reconstruction_archive_sha256"
                    )
                ),
            }

            failed_provenance = [
                name
                for name, passed
                in provenance_checks.items()
                if not passed
            ]

            if failed_provenance:
                raise RuntimeError(
                    "provenance checks failed: "
                    + ", ".join(
                        failed_provenance
                    )
                )

            hepmc_events = count_hepmc_events(
                hepmc_path
            )

            if hepmc_events != expected_events:
                raise RuntimeError(
                    "wrong HepMC count: "
                    f"{hepmc_events} != {expected_events}"
                )

            safe_extract(
                nested_archive,
                reconstruction_dir,
            )

            root_path = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*.root"
                    )
                ),
                "ROOT file",
            )

            event_summary_path = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*_event_summary.parquet"
                    )
                ),
                "event-summary Parquet",
            )

            v2_candidate_path = exactly_one(
                sorted(
                    reconstruction_dir.rglob(
                        "*_hh4b_candidates_v2.parquet"
                    )
                ),
                "v2 candidate Parquet",
            )

            legacy_candidate_paths = [
                path
                for path in sorted(
                    reconstruction_dir.rglob(
                        "*_hh4b_candidates.parquet"
                    )
                )
                if not path.name.endswith(
                    "_v2.parquet"
                )
            ]

            legacy_candidate_path = exactly_one(
                legacy_candidate_paths,
                "legacy candidate Parquet",
            )

            event_summary = pq.ParquetFile(
                event_summary_path
            )

            v2_candidates = pq.ParquetFile(
                v2_candidate_path
            )

            legacy_candidates = pq.ParquetFile(
                legacy_candidate_path
            )

            event_rows = (
                event_summary.metadata.num_rows
            )

            v2_rows = (
                v2_candidates.metadata.num_rows
            )

            legacy_rows = (
                legacy_candidates.metadata.num_rows
            )

            event_schema = schema_sha256(
                event_summary
            )

            v2_schema = schema_sha256(
                v2_candidates
            )

            legacy_schema = schema_sha256(
                legacy_candidates
            )

            if event_rows != expected_events:
                raise RuntimeError(
                    "wrong event-summary rows: "
                    f"{event_rows} != {expected_events}"
                )

            if v2_rows != legacy_rows:
                raise RuntimeError(
                    "candidate row mismatch: "
                    f"{v2_rows} != {legacy_rows}"
                )

            if (
                len(
                    event_summary.schema_arrow.names
                )
                != 13
                or event_schema
                != EXPECTED_EVENT_SCHEMA_SHA256
            ):
                raise RuntimeError(
                    "event-summary schema mismatch"
                )

            if (
                len(
                    v2_candidates.schema_arrow.names
                )
                != 72
                or v2_schema
                != EXPECTED_V2_SCHEMA_SHA256
            ):
                raise RuntimeError(
                    "v2 candidate schema mismatch"
                )

            if (
                len(
                    legacy_candidates.schema_arrow.names
                )
                != 15
                or legacy_schema
                != EXPECTED_LEGACY_SCHEMA_SHA256
            ):
                raise RuntimeError(
                    "legacy candidate schema mismatch"
                )

            root_file = ROOT.TFile.Open(
                str(root_path)
            )

            if (
                root_file is None
                or root_file.IsZombie()
            ):
                raise RuntimeError(
                    "ROOT file is unreadable"
                )

            tree = root_file.Get(
                "Delphes"
            )

            if tree is None:
                root_file.Close()

                raise RuntimeError(
                    "missing Delphes tree"
                )

            root_entries = int(
                tree.GetEntries()
            )

            root_file.Close()

            if root_entries != expected_events:
                raise RuntimeError(
                    "wrong ROOT count: "
                    f"{root_entries} != {expected_events}"
                )

            records.append({
                "campaign": row[
                    "campaign"
                ],
                "local_shard": (
                    local_shard
                ),
                "inner_shard": (
                    inner_shard
                ),
                "seed": seed,
                "immutable_split": (
                    immutable_split
                ),
                "generated_events": (
                    expected_events
                ),
                "hepmc_events": (
                    hepmc_events
                ),
                "root_events": (
                    root_entries
                ),
                "event_summary_rows": (
                    event_rows
                ),
                "candidate_rows": (
                    v2_rows
                ),
                "v2_columns": len(
                    v2_candidates
                    .schema_arrow
                    .names
                ),
                "v2_schema_sha256": (
                    v2_schema
                ),
                "remote_bundle": (
                    remote_bundle
                ),
                "payload_sha256": (
                    provenance.get(
                        "payload_sha256",
                        "",
                    )
                ),
                "delphes_card_sha256": (
                    provenance.get(
                        "delphes_card_sha256",
                        "",
                    )
                ),
                "provenance_split": (
                    provenance.get(
                        "immutable_split",
                        "",
                    )
                ),
                "status": "pass",
            })

            print(
                "  PASS "
                f"events={root_entries} "
                f"candidates={v2_rows}"
            )

        except Exception as exc:
            error = (
                f"local_shard={local_shard}: "
                f"{exc}"
            )

            errors.append(
                error
            )

            print(
                f"  ERROR {error}"
            )

        finally:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

    generated_events = sum(
        record[
            "generated_events"
        ]
        for record in records
    )

    candidate_rows = sum(
        record[
            "candidate_rows"
        ]
        for record in records
    )

    split_events = {}

    split_candidates = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        split_events[split] = sum(
            record[
                "generated_events"
            ]
            for record in records
            if record[
                "immutable_split"
            ]
            == split
        )

        split_candidates[split] = sum(
            record[
                "candidate_rows"
            ]
            for record in records
            if record[
                "immutable_split"
            ]
            == split
        )

    payload_hashes = sorted({
        record[
            "payload_sha256"
        ]
        for record in records
    })

    card_hashes = sorted({
        record[
            "delphes_card_sha256"
        ]
        for record in records
    })

    if len(records) != 20:
        errors.append(
            f"expected 20 passing shards, "
            f"found {len(records)}"
        )

    if generated_events != 20000:
        errors.append(
            "expected 20000 generated events, "
            f"found {generated_events}"
        )

    if split_events != {
        "train": 14000,
        "validation": 3000,
        "test": 3000,
    }:
        errors.append(
            "wrong split event totals: "
            f"{split_events}"
        )

    if len(payload_hashes) != 1:
        errors.append(
            "expected one payload hash, "
            f"found {len(payload_hashes)}"
        )

    if card_hashes != [
        EXPECTED_CARD_SHA256
    ]:
        errors.append(
            "unexpected Delphes-card hashes"
        )

    output_tsv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if records:
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
            writer.writerows(
                records
            )
    else:
        output_tsv.write_text(
            "status\n"
        )

    summary = {
        "schema_version": 1,
        "status": (
            "pass"
            if not errors
            else "fail"
        ),
        "manifest_rows": len(rows),
        "product_valid_shards": len(
            records
        ),
        "validated_generated_events": (
            generated_events
        ),
        "candidate_rows": (
            candidate_rows
        ),
        "candidate_efficiency": (
            candidate_rows
            / generated_events
            if generated_events
            else 0.0
        ),
        "split_events": (
            split_events
        ),
        "split_candidate_rows": (
            split_candidates
        ),
        "training_candidate_rows": (
            split_candidates["train"]
        ),
        "validation_candidate_rows": (
            split_candidates[
                "validation"
            ]
        ),
        "sealed_test_candidate_rows": (
            split_candidates["test"]
        ),
        "test_used_for_training": (
            False
        ),
        "v2_candidate_columns": (
            72
        ),
        "canonical_75_column_conversion_required": (
            True
        ),
        "payload_sha256": (
            payload_hashes
        ),
        "delphes_card_sha256": (
            card_hashes
        ),
        "output_tsv": str(
            output_tsv
        ),
        "errors": errors,
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

    print()
    print(
        f"product_valid_shards={len(records)}"
    )

    print(
        "validated_generated_events="
        f"{generated_events}"
    )

    print(
        f"candidate_rows={candidate_rows}"
    )

    print(
        "split_candidate_rows="
        f"{split_candidates}"
    )

    print(
        "test_used_for_training=False"
    )

    print(
        "canonical_75_column_conversion_required=True"
    )

    print(
        f"output_tsv={output_tsv}"
    )

    print(
        f"output_json={output_json}"
    )

    if errors:
        for error in errors:
            print(
                f"ERROR: {error}"
            )

        raise SystemExit(2)

    print(
        "GGF_SIGNAL_20K_PRODUCTS_VALID"
    )


if __name__ == "__main__":
    main()
