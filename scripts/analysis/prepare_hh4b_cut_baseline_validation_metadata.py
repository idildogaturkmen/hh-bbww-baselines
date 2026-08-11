#!/usr/bin/env python3
"""Prepare the sealed HH->4b cut-baseline validation source contract.

This program is deliberately metadata-only.  It reads frozen TSV registries and
uses ``stat`` for the five local legacy ROOT files; it never opens a validation
ROOT, Parquet, archive, or event payload.  Its outputs are suitable for binding
into the train-only master freeze and the later explicit validation
authorization checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


XROOTD_PREFIX = "root://cmseos.fnal.gov/"
AUXILIARY_PROCESSES = {"qcd_bbbb_general", "qcd_bbbb_iht400to600"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_uri(lfn: str) -> str:
    require(lfn.startswith("/store/"), f"not an EOS LFN: {lfn}")
    return XROOTD_PREFIX + lfn


def source_uid(row: pd.Series) -> str:
    return "::".join(
        [
            clean(row["group_id"]),
            clean(row["sample_class"]),
            str(int(row["source_index"])),
            clean(row["member_id"]),
        ]
    )


def bundle_lfn(row: pd.Series) -> str:
    candidates = [clean(row["source_locator"]), clean(row["member_id"])]
    for candidate in candidates:
        if candidate.startswith("bundle:"):
            candidate = candidate.removeprefix("bundle:")
        if candidate.startswith("/store/") and candidate.endswith(".tar.gz"):
            return candidate
    return ""


def select_root_map_row(
    validation_row: pd.Series,
    root_map: pd.DataFrame,
) -> pd.Series | None:
    candidate_path = clean(validation_row["candidate_path"])
    matches = root_map.loc[
        root_map["candidate_path"].astype(str).map(clean) == candidate_path
    ]
    if len(matches) == 1:
        return matches.iloc[0]
    require(len(matches) == 0, f"ambiguous candidate-path ROOT map: {candidate_path}")

    lfn = bundle_lfn(validation_row)
    if lfn:
        matches = root_map.loc[
            root_map["remote_bundle_path"].astype(str).map(clean) == lfn
        ]
        if len(matches) == 1:
            return matches.iloc[0]
        require(len(matches) == 0, f"ambiguous bundle ROOT map: {lfn}")
    return None


def legacy_ttbar_root_path(member_id: str, legacy_root: Path) -> Path:
    member = clean(member_id).removeprefix("event_parquet:")
    token = Path(member).name.removesuffix("_event_summary.parquet")
    require(token.startswith("ttbar_") and "_shard" in token, f"bad ttbar token: {token}")

    if token in {"ttbar_100k_shard004", "ttbar_100k_shard005"}:
        return (
            legacy_root
            / "condor_return/hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1"
            / "members"
            / token
            / "root"
            / f"{token}_pythia8_delphes.root"
        )
    return legacy_root / "root" / f"{token}_pythia8_delphes.root"


def hard_qcd_source_tag(lfn: str) -> str:
    name = Path(lfn).name
    require(name.endswith("_bundle.tar.gz"), f"bad hard-QCD bundle: {name}")
    return name.removesuffix("_bundle.tar.gz")


def build_metadata(
    development: pd.DataFrame,
    root_map: pd.DataFrame,
    ordinary: pd.DataFrame,
    hard_qcd: pd.DataFrame,
    ttbar_registry: pd.DataFrame,
    legacy_root: Path,
    remote_checksums: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation = development.loc[
        development["final_split"].astype(str).map(clean) == "validation"
    ].copy()
    validation = validation.sort_values(
        ["sample_class", "source_index", "group_id"], kind="mergesort"
    ).reset_index(drop=True)

    require(len(validation) == 121, f"validation sources={len(validation)}, expected 121")
    require(validation["group_id"].astype(str).nunique() == 121, "validation group IDs not unique")
    require(
        int(pd.to_numeric(validation["generated_events"], errors="raise").sum())
        == 1_002_584,
        "validation generated-event total changed",
    )
    for column in [
        "candidate_content_opened_in_this_step",
        "candidate_content_access_authorized_now",
        "physical_normalization_authorized",
    ]:
        require(not validation[column].map(truthy).any(), f"sealed flag changed: {column}")

    access_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for production_row_index, row in validation.iterrows():
        process = clean(row["process_or_mode"])
        sample_class = clean(row["sample_class"])
        auxiliary = process in AUXILIARY_PROCESSES
        uid = source_uid(row)
        lfn = bundle_lfn(row)
        mapped = select_root_map_row(row, root_map)

        if mapped is not None:
            require(clean(mapped["resolution_status"]) == "resolved", f"unresolved ROOT map: {uid}")
            remote_lfn = clean(mapped["remote_bundle_path"])
            require(remote_lfn == lfn, f"bundle locator mismatch: {uid}")
            access_mode = "remote_bundle_extraction_required"
            source_locator = remote_uri(remote_lfn)
            archive_locator = source_locator
            root_locator = ""
            root_archive_member = clean(mapped["root_archive_member"])
            root_basename = clean(mapped["root_basename"])
            layout_rule = clean(mapped["layout_rule"])
            checksum_kind = "adler32" if clean(mapped["bundle_adler32_expected"]) else ""
            checksum = clean(mapped["bundle_adler32_expected"])
            size_bytes = int(mapped["bundle_size_bytes"])
            access_evidence = "frozen_member_level_root_source_map"
        else:
            require(process == "ttbar_inclusive", f"missing ROOT map for non-ttbar source: {uid}")
            root_path = legacy_ttbar_root_path(clean(row["member_id"]), legacy_root)
            require(root_path.is_file(), f"missing legacy ttbar ROOT metadata target: {root_path}")
            token = root_path.name.removesuffix("_pythia8_delphes.root")
            matches = ttbar_registry.loc[
                ttbar_registry["member_id"].astype(str).map(clean) == token
            ]
            require(len(matches) == 1, f"legacy ttbar registry rows={len(matches)} for {token}")
            frozen = matches.iloc[0]
            require(clean(frozen["final_split"]) == "validation", f"ttbar split changed: {token}")
            require(int(frozen["generated_events_frozen"]) == int(row["generated_events"]), f"ttbar event count changed: {token}")
            access_mode = "local_direct_root"
            source_locator = str(root_path)
            archive_locator = ""
            root_locator = str(root_path)
            root_archive_member = ""
            root_basename = root_path.name
            layout_rule = "authoritative_local_legacy_root"
            checksum_kind = "root_sha256"
            checksum = clean(frozen["root_sha256"])
            size_bytes = root_path.stat().st_size
            access_evidence = "frozen_legacy_ttbar_registry_plus_stat_only"

        if auxiliary:
            coefficient = None
            coefficient_source = "excluded_auxiliary_qcd"
            coefficient_campaign = ""
            physical_eligible = False
        elif process == "qcd_hardqcd":
            tag = hard_qcd_source_tag(lfn)
            matches = hard_qcd.loc[
                hard_qcd["source_tag"].astype(str).map(clean) == tag
            ]
            require(len(matches) == 1, f"hard-QCD coefficient rows={len(matches)} for {tag}")
            frozen = matches.iloc[0]
            coefficient = float(frozen["run2_yield_coefficient_per_generator_weight"])
            coefficient_source = "frozen_hard_qcd_shard_registry"
            coefficient_campaign = clean(frozen["campaign"])
            physical_eligible = True
        else:
            matches = ordinary.loc[
                (ordinary["sample_class"].astype(str).map(clean) == sample_class)
                & (ordinary["process_or_mode"].astype(str).map(clean) == process)
            ]
            require(len(matches) == 1, f"ordinary coefficient rows={len(matches)} for {sample_class}/{process}")
            frozen = matches.iloc[0]
            coefficient = float(frozen["run2_yield_coefficient_per_generator_weight"])
            coefficient_source = "frozen_ordinary_process_registry"
            coefficient_campaign = "process_global"
            physical_eligible = True

        require(coefficient is None or coefficient > 0.0, f"invalid coefficient: {uid}")
        workflow_population = (
            "auxiliary_qcd_classification_only"
            if auxiliary
            else "primary_physical_validation"
        )
        transport_id = (
            f"bundle:{lfn}" if lfn else f"root:{root_locator}"
        )

        access_rows.append(
            {
                "production_row_index": production_row_index,
                "source_uid": uid,
                "group_id": clean(row["group_id"]),
                "source_index": int(row["source_index"]),
                "sample_class": sample_class,
                "process_or_mode": process,
                "split": "validation",
                "workflow_population": workflow_population,
                "generated_events": int(row["generated_events"]),
                "candidate_rows_metadata": int(row["candidate_rows_metadata"]),
                "member_id": clean(row["member_id"]),
                "transport_id": transport_id,
                "access_mode": access_mode,
                "source_locator": source_locator,
                "archive_locator": archive_locator,
                "root_locator": root_locator,
                "root_archive_member": root_archive_member,
                "root_basename": root_basename,
                "layout_rule": layout_rule,
                "source_checksum_kind": checksum_kind,
                "source_checksum": checksum,
                "source_size_bytes": size_bytes,
                "access_evidence": access_evidence,
                "auxiliary_qcd": auxiliary,
                "physical_evaluation_eligible": physical_eligible,
                "validation_content_opened": False,
                "validation_access_authorized": False,
                "test_content_opened": False,
            }
        )
        coefficient_rows.append(
            {
                "production_row_index": production_row_index,
                "source_uid": uid,
                "group_id": clean(row["group_id"]),
                "sample_class": sample_class,
                "process_or_mode": process,
                "auxiliary_qcd": auxiliary,
                "physical_evaluation_eligible": physical_eligible,
                "coefficient_campaign": coefficient_campaign,
                "coefficient_source": coefficient_source,
                "run2_yield_coefficient_per_generator_weight": coefficient,
                "luminosity_pb_inverse": 138000.0,
                "coefficient_construction_authorized": True,
                "validation_weight_application_authorized": False,
                "validation_content_opened": False,
                "test_content_opened": False,
            }
        )

    access = pd.DataFrame(access_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    remote_mask = access["access_mode"].eq("remote_bundle_extraction_required")
    if remote_checksums is not None:
        required_checksum_columns = {
            "source_uid",
            "archive_locator",
            "source_size_bytes",
            "checksum_kind",
            "checksum",
            "independent_query_passes",
            "independent_queries_match",
            "validation_event_payload_opened",
            "test_event_payload_opened",
            "status",
        }
        require(
            required_checksum_columns.issubset(remote_checksums.columns),
            "remote checksum registry schema changed",
        )
        require(len(remote_checksums) == int(remote_mask.sum()) == 116, "remote checksum row closure changed")
        require(remote_checksums["source_uid"].astype(str).is_unique, "remote checksum source UID is not unique")
        checksum_by_uid = remote_checksums.set_index("source_uid", drop=False)
        for index in access.index[remote_mask]:
            uid = access.at[index, "source_uid"]
            require(uid in checksum_by_uid.index, f"remote checksum missing: {uid}")
            frozen = checksum_by_uid.loc[uid]
            require(clean(frozen["archive_locator"]) == access.at[index, "archive_locator"], f"remote checksum locator changed: {uid}")
            require(int(frozen["source_size_bytes"]) == int(access.at[index, "source_size_bytes"]), f"remote checksum size changed: {uid}")
            require(clean(frozen["checksum_kind"]).lower() == "adler32", f"remote checksum kind changed: {uid}")
            require(int(frozen["independent_query_passes"]) == 2, f"remote checksum query count changed: {uid}")
            require(truthy(frozen["independent_queries_match"]), f"remote checksum rerun mismatch: {uid}")
            require(not truthy(frozen["validation_event_payload_opened"]), f"checksum query opened validation: {uid}")
            require(not truthy(frozen["test_event_payload_opened"]), f"checksum query opened test: {uid}")
            require(clean(frozen["status"]) == "pass_metadata_only_remote_checksum_closure", f"remote checksum status changed: {uid}")
            checksum = clean(frozen["checksum"]).lower()
            require(len(checksum) == 8, f"bad remote Adler-32: {uid}")
            preexisting = clean(access.at[index, "source_checksum"]).lower()
            require(not preexisting or preexisting == checksum, f"preexisting checksum mismatch: {uid}")
            access.at[index, "source_checksum_kind"] = "adler32"
            access.at[index, "source_checksum"] = checksum

    require(
        access.loc[remote_mask, "source_checksum_kind"].eq("adler32").all()
        and access.loc[remote_mask, "source_checksum"].astype(str).str.fullmatch(r"[0-9a-f]{8}").all(),
        "remote validation archive checksum closure is incomplete",
    )
    require(access["source_uid"].is_unique, "validation source UID is not unique")
    require(coefficients["source_uid"].is_unique, "validation coefficient UID is not unique")
    require(access["source_uid"].tolist() == coefficients["source_uid"].tolist(), "registry order mismatch")
    require(int(access["physical_evaluation_eligible"].sum()) == 116, "physical validation source count changed")
    require(int(access["auxiliary_qcd"].sum()) == 5, "auxiliary validation source count changed")
    require(not access["validation_content_opened"].any(), "validation content unexpectedly opened")
    require(not access["test_content_opened"].any(), "test content unexpectedly opened")

    summary = {
        "schema_version": 1,
        "status": "pass_metadata_only",
        "validation_sources": len(access),
        "validation_generated_events": int(access["generated_events"].sum()),
        "validation_background_sources": int(access["sample_class"].eq("background").sum()),
        "validation_signal_sources": int(access["sample_class"].eq("signal").sum()),
        "physical_evaluation_sources": int(access["physical_evaluation_eligible"].sum()),
        "auxiliary_qcd_sources": int(access["auxiliary_qcd"].sum()),
        "remote_bundle_sources": int(access["access_mode"].eq("remote_bundle_extraction_required").sum()),
        "remote_bundle_checksum_closure": "pass_116_of_116",
        "remote_bundle_checksum_independent_queries_per_source": 2,
        "local_direct_root_sources": int(access["access_mode"].eq("local_direct_root").sum()),
        "luminosity_pb_inverse": 138000.0,
        "exposure_label": "Run-2 13 TeV, 138 fb^-1 expected-yield projection",
        "event_payload_files_opened": 0,
        "validation_payloads_opened": 0,
        "validation_access_authorized": False,
        "test_payloads_opened": 0,
        "nominal_cut_changed": False,
    }
    return access, coefficients, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--root-source-map", type=Path, required=True)
    parser.add_argument("--ordinary-coefficients", type=Path, required=True)
    parser.add_argument("--hard-qcd-coefficients", type=Path, required=True)
    parser.add_argument("--ttbar-registry", type=Path, required=True)
    parser.add_argument("--remote-checksum-registry", type=Path, required=True)
    parser.add_argument(
        "--remote-checksum-registry-label",
        help="Stable repository-relative provenance label when the registry is staged atomically.",
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_paths = [
        args.development_manifest,
        args.root_source_map,
        args.ordinary_coefficients,
        args.hard_qcd_coefficients,
        args.ttbar_registry,
        args.remote_checksum_registry,
    ]
    for path in input_paths:
        require(path.is_file(), f"missing frozen metadata input: {path}")

    access, coefficients, summary = build_metadata(
        pd.read_csv(args.development_manifest, sep="\t", keep_default_na=False),
        pd.read_csv(args.root_source_map, sep="\t", keep_default_na=False),
        pd.read_csv(args.ordinary_coefficients, sep="\t", keep_default_na=False),
        pd.read_csv(args.hard_qcd_coefficients, sep="\t", keep_default_na=False),
        pd.read_csv(args.ttbar_registry, sep="\t", keep_default_na=False),
        args.legacy_root,
        pd.read_csv(args.remote_checksum_registry, sep="\t", keep_default_na=False),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    access_path = args.output_dir / "validation_121_source_access_manifest.tsv"
    coefficient_path = args.output_dir / "validation_121_physical_coefficient_registry.tsv"
    summary_path = args.output_dir / "validation_metadata_preparation_summary.json"
    inputs_path = args.output_dir / "frozen_metadata_input_manifest.tsv"

    access.to_csv(access_path, sep="\t", index=False, lineterminator="\n")
    coefficients.to_csv(coefficient_path, sep="\t", index=False, lineterminator="\n")
    input_labels = [str(path) for path in input_paths]
    if args.remote_checksum_registry_label:
        input_labels[-1] = args.remote_checksum_registry_label
    summary["input_sha256"] = {
        label: sha256(path) for label, path in zip(input_labels, input_paths)
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"path": label, "sha256": sha256(path), "bytes": path.stat().st_size}
            for label, path in zip(input_labels, input_paths)
        ]
    ).to_csv(inputs_path, sep="\t", index=False, lineterminator="\n")

    outputs = [access_path, coefficient_path, inputs_path, summary_path]
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(outputs)),
        encoding="utf-8",
    )

    print("VALIDATION_METADATA_PREPARATION=PASS")
    print("VALIDATION_SOURCES=121")
    print("VALIDATION_GENERATED_EVENTS=1002584")
    print("PHYSICAL_EVALUATION_SOURCES=116")
    print("AUXILIARY_QCD_SOURCES=5")
    print("EVENT_PAYLOAD_FILES_OPENED=0")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("VALIDATION_ACCESS_AUTHORIZED=FALSE")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
