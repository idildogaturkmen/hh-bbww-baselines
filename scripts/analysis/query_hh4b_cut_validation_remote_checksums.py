#!/usr/bin/env python3
"""Close remote validation bundle identities using metadata-only XRootD queries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlparse

import pandas as pd


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


def lfn_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    require(parsed.scheme == "root" and parsed.netloc == "cmseos.fnal.gov", f"bad XRootD URI: {uri}")
    lfn = "/" + parsed.path.lstrip("/")
    require(lfn.startswith("/store/"), f"bad EOS LFN in URI: {uri}")
    return lfn


def parse_checksum(output: str) -> tuple[str, str]:
    fields = output.strip().split()
    require(len(fields) == 2, f"unexpected xrdfs checksum output: {output!r}")
    kind = fields[0].lower()
    value = fields[1].lower().removeprefix("0x")
    require(kind == "adler32", f"unexpected checksum kind: {kind}")
    require(len(value) == 8 and all(char in "0123456789abcdef" for char in value), f"bad Adler-32: {value}")
    return kind, value


def query_one(row: dict[str, Any]) -> dict[str, Any]:
    uri = clean(row["archive_locator"])
    lfn = lfn_from_uri(uri)
    observations = []
    for _ in range(2):
        result = subprocess.run(
            ["xrdfs", "root://cmseos.fnal.gov", "query", "checksum", lfn],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        require(result.returncode == 0, f"xrdfs checksum failed for {lfn}: {result.stderr}")
        observations.append(parse_checksum(result.stdout))
    require(observations[0] == observations[1], f"checksum rerun mismatch: {lfn}")
    kind, checksum = observations[0]
    frozen = clean(row["source_checksum"])
    frozen_kind = clean(row["source_checksum_kind"])
    if frozen:
        require(frozen_kind.lower() == kind and frozen.lower() == checksum, f"frozen checksum mismatch: {lfn}")
    return {
        "production_row_index": int(row["production_row_index"]),
        "source_uid": clean(row["source_uid"]),
        "archive_locator": uri,
        "lfn": lfn,
        "source_size_bytes": int(row["source_size_bytes"]),
        "checksum_kind": kind,
        "checksum": checksum,
        "independent_query_passes": 2,
        "independent_queries_match": True,
        "frozen_checksum_preexisted": bool(frozen),
        "validation_event_payload_opened": False,
        "test_event_payload_opened": False,
        "status": "pass_metadata_only_remote_checksum_closure",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-source-access-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.draft_source_access_manifest.is_file(), "draft access manifest missing")
    require(1 <= args.workers <= 16, "workers must be in [1,16]")
    access = pd.read_csv(
        args.draft_source_access_manifest, sep="\t", keep_default_na=False
    )
    require(len(access) == 121 and access["source_uid"].nunique() == 121, "draft source closure changed")
    require(not access["validation_content_opened"].map(truthy).any(), "validation content already opened")
    require(not access["test_content_opened"].map(truthy).any(), "test content already opened")
    remote = access.loc[
        access["access_mode"].astype(str).eq("remote_bundle_extraction_required")
    ].copy()
    require(len(remote) == 116, f"remote validation bundles={len(remote)}, expected 116")
    require(remote["archive_locator"].astype(str).is_unique, "remote archive locator is not unique")

    rows = remote.sort_values("production_row_index").to_dict("records")
    output_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(query_one, row): row for row in rows}
        for future in as_completed(futures):
            output_rows.append(future.result())
            if len(output_rows) % 20 == 0:
                print(f"REMOTE_CHECKSUMS_CLOSED={len(output_rows)}", flush=True)
    output_rows.sort(key=lambda row: int(row["production_row_index"]))
    require(len(output_rows) == 116, "remote checksum output count changed")

    require(not args.output_dir.exists(), f"output already exists: {args.output_dir}")
    args.output_dir.mkdir()
    registry_path = args.output_dir / "validation_116_remote_bundle_checksum_registry.tsv"
    pd.DataFrame(output_rows).to_csv(
        registry_path, sep="\t", index=False, lineterminator="\n"
    )
    summary = {
        "schema_version": 1,
        "status": "pass_metadata_only_validation_remote_bundle_checksum_closure",
        "remote_bundles": 116,
        "independent_checksum_queries_per_bundle": 2,
        "all_independent_queries_match": True,
        "preexisting_frozen_checksums_verified": sum(
            int(row["frozen_checksum_preexisted"]) for row in output_rows
        ),
        "new_metadata_checksum_closures": sum(
            int(not row["frozen_checksum_preexisted"]) for row in output_rows
        ),
        "draft_source_access_manifest_sha256": sha256(
            args.draft_source_access_manifest
        ),
        "event_payload_files_opened": 0,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    summary_path = args.output_dir / "validation_remote_checksum_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SHA256SUMS").write_text(
        f"{sha256(registry_path)}  {registry_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print("VALIDATION_REMOTE_BUNDLE_CHECKSUM_CLOSURE=PASS")
    print("REMOTE_BUNDLES=116")
    print("INDEPENDENT_CHECKSUM_QUERIES_PER_BUNDLE=2")
    print("EVENT_PAYLOAD_FILES_OPENED=0")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
