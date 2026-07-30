#!/usr/bin/env python3
"""Freeze the completed train-only hard-QCD generator-weight transport.

This is a text-only/checksum-only checkpointing step. It opens no ROOT, Parquet,
HepMC, LHE, or other event/candidate payload and computes no luminosity weight or
physical yield.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

EXPECTED = {
    "production": 260,
    "train": 186,
    "variable_train": 127,
    "uniform_train": 59,
    "validation": 56,
    "test": 18,
}

TOP_LEVEL_REQUIRED = (
    "fragment_checksum_manifest.tsv",
    "inventory_summary.json",
    "ordered_train_transport_targets.tsv",
    "PATCH_PN_C4I_ZERO_CANDIDATE_ROWS_V3.txt",
    "PN_C4I_QCD_TRAIN_GENERATOR_WEIGHT_TRANSPORT_SCALEOUT_PASS",
    "process_qcd_train_transport.py",
    "process_qcd_train_transport_v1.py",
    "qcd_train_generator_weight_sidecars.tsv",
    "qcd_train_generator_weight_transport_registry.tsv",
    "qcd_train_uniform_generator_weight_registry.tsv",
    "qcd_transport_target_inventory.tsv",
    "report.txt",
    "RUN_CONTRACT.txt",
    "source_canary_report.txt",
    "source_canary_SHA256SUMS",
    "source_canary_summary.json",
    "summary.json",
)

COPIED_PRODUCTS = (
    "fragment_checksum_manifest.tsv",
    "inventory_summary.json",
    "ordered_train_transport_targets.tsv",
    "PATCH_PN_C4I_ZERO_CANDIDATE_ROWS_V3.txt",
    "qcd_train_generator_weight_sidecars.tsv",
    "qcd_train_generator_weight_transport_registry.tsv",
    "qcd_train_uniform_generator_weight_registry.tsv",
    "qcd_transport_target_inventory.tsv",
    "RUN_CONTRACT.txt",
    "source_canary_report.txt",
    "source_canary_SHA256SUMS",
    "source_canary_summary.json",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise RuntimeError(f"invalid boolean value: {value!r}")


def verify_checksum_manifest(directory: Path, manifest_name: str = "SHA256SUMS") -> None:
    manifest = directory / manifest_name
    if not manifest.is_file():
        raise RuntimeError(f"checksum manifest absent: {manifest}")
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            raise RuntimeError(f"malformed checksum line {manifest}:{line_number}")
        expected, rel_text = pieces
        rel_text = rel_text.lstrip("*")
        rel = PurePosixPath(rel_text)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe checksum path in {manifest}: {rel_text}")
        normalized = rel_text.removeprefix("./")
        if normalized in seen:
            raise RuntimeError(f"duplicate checksum path in {manifest}: {normalized}")
        seen.add(normalized)
        target = directory / normalized
        if not target.is_file():
            raise RuntimeError(f"checksummed file absent: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for {target}: actual={actual}, expected={expected}"
            )


def write_checksum_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    products = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name != manifest.name
    )
    manifest.write_text(
        "".join(f"{sha256_file(path)}  ./{path.name}\n" for path in products),
        encoding="utf-8",
    )


def require_zero_controls(controls: dict[str, object], context: str) -> None:
    protected = (
        "train_candidate_nonidentity_columns_opened",
        "validation_payload_files_opened",
        "validation_candidate_files_opened",
        "evaluation_payload_files_opened",
        "evaluation_candidate_files_opened",
        "physical_luminosity_weights_calculated",
        "physical_yields_calculated",
        "models_trained",
        "thresholds_selected",
    )
    for field in protected:
        if field in controls and int(controls[field]) != 0:
            raise RuntimeError(f"protected control changed in {context}: {field}={controls[field]}")


def validate_scaleout(source: Path) -> dict[str, object]:
    for name in TOP_LEVEL_REQUIRED:
        if not (source / name).is_file():
            raise RuntimeError(f"required PN-c4i product absent: {source / name}")
    verify_checksum_manifest(source)

    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "hh4b_qcd_train_generator_weight_transport_scaleout_pass":
        raise RuntimeError("PN-c4i source summary is not pass")
    transport = summary.get("transport", {})
    expected_transport = {
        "train_shards": 186,
        "variable_train_shards": 127,
        "uniform_train_shards": 59,
        "complete_fragments": 186,
        "first_transport_bundle_checksums_frozen": 3,
    }
    for field, expected in expected_transport.items():
        if int(transport.get(field, -1)) != expected:
            raise RuntimeError(
                f"PN-c4i transport mismatch: {field}={transport.get(field)!r}, expected={expected}"
            )
    controls = summary.get("controls", {})
    require_zero_controls(controls, "PN-c4i top-level summary")
    authorization = summary.get("authorization", {})
    if authorization.get("qcd_train_generator_weight_transport_complete") is not True:
        raise RuntimeError("QCD train transport is not complete")
    if authorization.get("qcd_train_physical_weight_application_authorized") is not False:
        raise RuntimeError("physical QCD weight application was unexpectedly authorized")
    if authorization.get("validation_generator_weight_transport_authorized") is not False:
        raise RuntimeError("validation transport was unexpectedly authorized")
    if authorization.get("evaluation_generator_weight_transport_authorized") is not False:
        raise RuntimeError("evaluation transport was unexpectedly authorized")

    inventory = read_tsv(source / "qcd_transport_target_inventory.tsv")
    targets = read_tsv(source / "ordered_train_transport_targets.tsv")
    registry = read_tsv(source / "qcd_train_generator_weight_transport_registry.tsv")
    sidecars = read_tsv(source / "qcd_train_generator_weight_sidecars.tsv")
    uniform = read_tsv(source / "qcd_train_uniform_generator_weight_registry.tsv")
    fragment_manifest = read_tsv(source / "fragment_checksum_manifest.tsv")

    if len(inventory) != EXPECTED["production"]:
        raise RuntimeError(f"transport inventory rows={len(inventory)}, expected=260")
    split_counts = Counter(row["final_split"] for row in inventory)
    if split_counts != {"train": 186, "validation": 56, "test": 18}:
        raise RuntimeError(f"transport split counts changed: {dict(split_counts)}")
    if len(targets) != EXPECTED["train"]:
        raise RuntimeError(f"ordered train targets={len(targets)}, expected=186")
    if len(registry) != EXPECTED["train"]:
        raise RuntimeError(f"transport registry rows={len(registry)}, expected=186")
    if len(uniform) != EXPECTED["uniform_train"]:
        raise RuntimeError(f"uniform registry rows={len(uniform)}, expected=59")
    if len(fragment_manifest) != EXPECTED["train"]:
        raise RuntimeError(f"fragment manifest rows={len(fragment_manifest)}, expected=186")
    if len(sidecars) != int(transport.get("variable_candidate_sidecar_rows", -1)):
        raise RuntimeError("consolidated sidecar row count differs from source summary")
    if sum(int(row["candidate_rows"]) for row in uniform) != int(
        transport.get("uniform_candidate_rows_covered_by_constant_registry", -1)
    ):
        raise RuntimeError("uniform candidate coverage differs from source summary")

    target_tags = [row["source_tag"] for row in targets]
    if len(target_tags) != len(set(target_tags)):
        raise RuntimeError("ordered train target source tags are not unique")
    registry_by_tag = {row["source_tag"]: row for row in registry}
    manifest_by_tag = {row["source_tag"]: row for row in fragment_manifest}
    if set(registry_by_tag) != set(target_tags):
        raise RuntimeError("transport registry target set differs from ordered targets")
    if set(manifest_by_tag) != set(target_tags):
        raise RuntimeError("fragment manifest target set differs from ordered targets")

    transport_counts = Counter(row["transport_class"] for row in registry)
    if transport_counts != {"variable": 127, "uniform": 59}:
        raise RuntimeError(f"transport-class counts changed: {dict(transport_counts)}")
    for row in registry:
        if row["final_split"] != "train":
            raise RuntimeError(f"non-train row in transport registry: {row['source_tag']}")
        if not parse_bool(row["transport_complete"]):
            raise RuntimeError(f"incomplete transport registry row: {row['source_tag']}")
        if parse_bool(row["physical_weight_application_authorized"]):
            raise RuntimeError(f"physical weight unexpectedly authorized: {row['source_tag']}")

    closure_rows: list[dict[str, object]] = []
    first_checksum_freezes = 0
    zero_candidate_variable = 0
    variable_sidecar_total = 0
    for expected_index, target in enumerate(targets, start=1):
        if int(target["target_index"]) != expected_index:
            raise RuntimeError("ordered train target indices changed")
        tag = target["source_tag"]
        fragment = source / "fragments" / tag
        if fragment.resolve() != Path(registry_by_tag[tag]["fragment_path"]).resolve():
            raise RuntimeError(f"fragment path mismatch for {tag}")
        if not fragment.is_dir() or not (fragment / "COMPLETE").is_file():
            raise RuntimeError(f"fragment incomplete: {fragment}")
        verify_checksum_manifest(fragment)
        if sha256_file(fragment / "SHA256SUMS") != registry_by_tag[tag]["fragment_sha256s_sha256"]:
            raise RuntimeError(f"fragment checksum-manifest digest mismatch: {tag}")
        if sha256_file(fragment / "SHA256SUMS") != manifest_by_tag[tag]["fragment_sha256s_sha256"]:
            raise RuntimeError(f"fragment top-level manifest mismatch: {tag}")

        fragment_summary = json.loads((fragment / "summary.json").read_text(encoding="utf-8"))
        if fragment_summary.get("complete") is not True:
            raise RuntimeError(f"fragment summary incomplete: {tag}")
        if fragment_summary.get("source_tag") != tag:
            raise RuntimeError(f"fragment source-tag mismatch: {tag}")
        if int(fragment_summary.get("target_index", -1)) != expected_index:
            raise RuntimeError(f"fragment target-index mismatch: {tag}")
        transport_class = registry_by_tag[tag]["transport_class"]
        if fragment_summary.get("transport_class") != transport_class:
            raise RuntimeError(f"fragment transport-class mismatch: {tag}")
        controls = fragment_summary.get("controls", {})
        require_zero_controls(controls, f"fragment {tag}")

        base = {
            "target_index": expected_index,
            "source_tag": tag,
            "campaign": registry_by_tag[tag]["campaign"],
            "transport_class": transport_class,
            "generated_events": int(registry_by_tag[tag]["generated_events"]),
            "candidate_rows": int(registry_by_tag[tag]["candidate_rows"]),
            "candidate_sha256": registry_by_tag[tag]["candidate_sha256"],
            "source_bundle_sha256": registry_by_tag[tag]["source_bundle_sha256"],
            "source_bundle_sha256_origin": registry_by_tag[tag]["source_bundle_sha256_origin"],
            "fragment_sha256s_sha256": registry_by_tag[tag]["fragment_sha256s_sha256"],
            "fragment_complete_marker_sha256": sha256_file(fragment / "COMPLETE"),
            "root_tree": "",
            "root_weight_branch": "",
            "root_weight_storage_dtype": "",
            "n_events_closure": "not_applicable_uniform",
            "sum_event_weights_closure": "not_applicable_uniform",
            "sum_squared_event_weights_closure": "not_applicable_uniform",
            "min_event_weight_closure": "not_applicable_uniform",
            "max_event_weight_closure": "not_applicable_uniform",
            "candidate_event_join_column": "not_applicable_uniform",
            "candidate_event_join_pass": "not_applicable_uniform",
            "sidecar_rows": 0,
            "source_bundle_sha256_frozen_during_transport": False,
            "validation_payload_files_opened": 0,
            "evaluation_payload_files_opened": 0,
            "physical_luminosity_weights_calculated": 0,
            "physical_yields_calculated": 0,
            "transport_complete": True,
        }

        closure = fragment_summary.get("closure", {})
        if transport_class == "variable":
            expected_status = "hh4b_qcd_train_variable_weight_sidecar_fragment_pass"
            if fragment_summary.get("status") != expected_status:
                raise RuntimeError(f"variable fragment status changed: {tag}")
            required_closure = (
                "n_events",
                "sum_event_weights",
                "sum_squared_event_weights",
                "min_event_weight",
                "max_event_weight",
            )
            if any(closure.get(field) is not True for field in required_closure):
                raise RuntimeError(f"variable fragment closure failed: {tag}")
            if fragment_summary.get("candidate_event_join_pass") is not True:
                raise RuntimeError(f"candidate event join failed: {tag}")
            sidecar_rows = int(fragment_summary.get("sidecar_rows", -1))
            if sidecar_rows != int(registry_by_tag[tag]["candidate_rows"]):
                raise RuntimeError(f"variable sidecar row mismatch: {tag}")
            variable_sidecar_total += sidecar_rows
            if int(registry_by_tag[tag]["candidate_rows"]) == 0:
                zero_candidate_variable += 1
            if fragment_summary.get("source_bundle_sha256_frozen_now") is True:
                first_checksum_freezes += 1
            moments = read_tsv(fragment / "root_event_weight_moments.tsv")
            joins = read_tsv(fragment / "train_candidate_event_join_audit.tsv")
            if len(moments) != 1 or len(joins) != 1:
                raise RuntimeError(f"variable audit row count changed: {tag}")
            moment = moments[0]
            join = joins[0]
            if not parse_bool(moment["all_moment_closures_pass"]):
                raise RuntimeError(f"moment audit not pass: {tag}")
            if not parse_bool(join["join_pass"]):
                raise RuntimeError(f"join audit not pass: {tag}")
            if int(join["candidate_nonidentity_columns_opened"]) != 0:
                raise RuntimeError(f"nonidentity candidate column opened: {tag}")
            base.update(
                {
                    "root_tree": fragment_summary["root_tree"],
                    "root_weight_branch": fragment_summary["root_weight_branch"],
                    "root_weight_storage_dtype": fragment_summary["root_weight_storage_dtype"],
                    "n_events_closure": True,
                    "sum_event_weights_closure": True,
                    "sum_squared_event_weights_closure": True,
                    "min_event_weight_closure": True,
                    "max_event_weight_closure": True,
                    "candidate_event_join_column": join["event_join_column"],
                    "candidate_event_join_pass": True,
                    "sidecar_rows": sidecar_rows,
                    "source_bundle_sha256_frozen_during_transport": bool(
                        fragment_summary.get("source_bundle_sha256_frozen_now")
                    ),
                }
            )
        elif transport_class == "uniform":
            expected_status = "hh4b_qcd_train_uniform_weight_registry_fragment_pass"
            if fragment_summary.get("status") != expected_status:
                raise RuntimeError(f"uniform fragment status changed: {tag}")
            if any(value is not True for value in closure.values()):
                raise RuntimeError(f"uniform metadata closure failed: {tag}")
            if int(controls.get("train_root_files_opened", -1)) != 0:
                raise RuntimeError(f"uniform ROOT payload unexpectedly opened: {tag}")
            if int(controls.get("train_candidate_parquet_content_opened", -1)) != 0:
                raise RuntimeError(f"uniform candidate content unexpectedly opened: {tag}")
        else:
            raise RuntimeError(f"unknown transport class for {tag}: {transport_class}")
        closure_rows.append(base)

    if variable_sidecar_total != len(sidecars):
        raise RuntimeError("fragment sidecar total differs from consolidated sidecars")
    if first_checksum_freezes != 3:
        raise RuntimeError(f"first-transport checksum freezes={first_checksum_freezes}, expected=3")
    if zero_candidate_variable != int(transport.get("zero_candidate_variable_train_shards", -1)):
        raise RuntimeError("zero-candidate variable shard count differs from source summary")

    return {
        "source_summary": summary,
        "closure_rows": closure_rows,
        "sidecar_rows": len(sidecars),
        "uniform_rows": len(uniform),
        "zero_candidate_variable": zero_candidate_variable,
        "first_checksum_freezes": first_checksum_freezes,
    }


def write_readme(path: Path, source: Path, source_commit: str, facts: dict[str, object]) -> None:
    source_summary = facts["source_summary"]
    transport = source_summary["transport"]
    path.write_text(
        "# HH4b hard-QCD train generator-weight transport freeze\n\n"
        "This checkpoint freezes the completed PN-c4i train-only transport of nominal "
        "Pythia generator weights into immutable candidate-level sidecars for variable-weight "
        "hard-QCD shards and a compact constant-weight registry for uniform-positive shards.\n\n"
        f"- Source commit: `{source_commit}`\n"
        f"- Source scale-out: `{source}`\n"
        "- Train shards: 186 (127 variable, 59 uniform)\n"
        f"- Variable candidate sidecar rows: {transport['variable_candidate_sidecar_rows']}\n"
        f"- Uniform candidate rows covered by constants: "
        f"{transport['uniform_candidate_rows_covered_by_constant_registry']}\n"
        f"- Zero-candidate variable shards: {transport['zero_candidate_variable_train_shards']}\n"
        "- Validation payloads opened: 0\n"
        "- Final-evaluation payloads opened: 0\n"
        "- Physical luminosity weights or yields calculated: 0\n\n"
        "The sidecar quantity `generator_cross_section_contribution_pb` is the already-frozen "
        "direct hard-QCD stitched projection coefficient multiplied by the nominal generator "
        "weight. Direct hard-QCD remains secondary closure/projection only; the intended "
        "primary multijet treatment remains the lower-b-tag control-region transfer.\n\n"
        "## Next gate\n\n"
        "Build the process reference-cross-section and normalization provenance registry for "
        "the HH signals and non-QCD backgrounds. No luminosity-normalized candidate weight is "
        "authorized until generator definitions, reference cross sections, decay/filter "
        "conventions, signed generator-weight denominators, and overlap policies are frozen.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaleout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source = args.scaleout_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if not source.is_dir():
        raise RuntimeError(f"scale-out source absent: {source}")

    facts = validate_scaleout(source)
    output.mkdir(parents=True)
    try:
        for name in COPIED_PRODUCTS:
            shutil.copy2(source / name, output / name)
        shutil.copy2(source / "summary.json", output / "source_scaleout_summary.json")
        shutil.copy2(source / "report.txt", output / "source_scaleout_report.txt")
        shutil.copy2(source / "SHA256SUMS", output / "source_scaleout_SHA256SUMS")
        shutil.copy2(
            source / "PN_C4I_QCD_TRAIN_GENERATOR_WEIGHT_TRANSPORT_SCALEOUT_PASS",
            output / "source_PN_C4I_QCD_TRAIN_GENERATOR_WEIGHT_TRANSPORT_SCALEOUT_PASS",
        )

        closure_rows = facts["closure_rows"]
        write_tsv(
            output / "qcd_train_transport_fragment_closure_registry.tsv",
            closure_rows,
            list(closure_rows[0]),
        )

        source_summary = facts["source_summary"]
        source_transport = source_summary["transport"]
        summary = {
            "schema_version": 1,
            "status": "hh4b_qcd_train_generator_weight_transport_freeze_pass",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "source_scaleout_dir": str(source),
            "source_scaleout_summary_sha256": sha256_file(source / "summary.json"),
            "inventory": {
                "production_qcd_shards": 260,
                "train_qcd_shards": 186,
                "variable_train_shards": 127,
                "uniform_train_shards": 59,
                "validation_qcd_shards_sealed": 56,
                "evaluation_qcd_shards_sealed": 18,
                "variable_candidate_sidecar_rows": facts["sidecar_rows"],
                "uniform_registry_rows": facts["uniform_rows"],
                "uniform_candidate_rows_covered": int(
                    source_transport["uniform_candidate_rows_covered_by_constant_registry"]
                ),
                "zero_candidate_variable_train_shards": facts["zero_candidate_variable"],
                "first_transport_bundle_checksums_frozen": facts["first_checksum_freezes"],
                "fragment_closure_rows": len(closure_rows),
            },
            "readiness": {
                "all_186_train_qcd_transports_frozen": True,
                "all_127_variable_root_moment_closures_frozen": True,
                "all_127_variable_candidate_joins_frozen": True,
                "all_59_uniform_constant_weight_records_frozen": True,
                "qcd_train_generator_weight_transport_complete": True,
                "qcd_train_physical_weight_application_authorized": False,
                "validation_generator_weight_transport_authorized": False,
                "evaluation_generator_weight_transport_authorized": False,
            },
            "controls": {
                "payload_files_opened_in_freeze_step": 0,
                "candidate_rows_read_in_freeze_step": 0,
                "validation_payload_files_opened": 0,
                "evaluation_payload_files_opened": 0,
                "physical_luminosity_weights_calculated": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": "build_process_reference_cross_section_registry",
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_readme(output / "README.md", source, args.source_commit, facts)
        report = [
            "PN-c4j QCD train generator-weight transport freeze",
            "====================================================",
            "",
            "Train QCD shards frozen: 186",
            "Variable train sidecars frozen: 127",
            "Uniform train registry rows frozen: 59",
            f"Variable candidate sidecar rows: {facts['sidecar_rows']}",
            f"Zero-candidate variable shards: {facts['zero_candidate_variable']}",
            "Variable ROOT moment closures frozen: 127/127",
            "Variable candidate joins frozen: 127/127",
            "Validation payload files opened: 0",
            "Final-evaluation payload files opened: 0",
            "Physical luminosity weights calculated: 0",
            "Physical yields calculated: 0",
            "",
            "NEXT_GATE: build_process_reference_cross_section_registry",
            "",
            "PN_C4J_QCD_TRAIN_GENERATOR_WEIGHT_TRANSPORT_FREEZE_PASS",
        ]
        (output / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        (output / "PN_C4J_QCD_TRAIN_GENERATOR_WEIGHT_TRANSPORT_FREEZE_PASS").write_text(
            "", encoding="utf-8"
        )
        write_checksum_manifest(output)
        verify_checksum_manifest(output)
        print("\n".join(report))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
