'''
Products are staged to EOS in two steps:
1. Verify the local file exists and matches the canonical checksum.
2. Copy the file to EOS and verify the remote checksum matches the canonical checksum.
The transfer manifest is updated after each file is staged, so the process can be resumed if interrupted

'''
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import uuid
import zlib


def command(args, *, check=True):
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def checksums(path):
    sha256 = hashlib.sha256()
    adler32 = 1
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size_bytes += len(chunk)
            sha256.update(chunk)
            adler32 = zlib.adler32(chunk, adler32)
    return size_bytes, sha256.hexdigest(), f"{adler32 & 0xffffffff:08x}"


def remote_checksum(host, lfn):
    result = command(
        ["xrdfs", host, "query", "checksum", lfn],
        check=False,
    )
    if result.returncode:
        return None
    fields = result.stdout.strip().split()
    return fields[-1].lower() if fields else None


def remote_exists(host, lfn):
    return command(
        ["xrdfs", host, "stat", lfn],
        check=False,
    ).returncode == 0


def record(manifest, payload):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def stage_one(host, eos_dir, row, transfer_manifest):
    local = Path(row["path"]).resolve()
    remote = str(PurePosixPath(eos_dir) / row["filename"])

    if not local.is_file() or local.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty local file: {local}")

    size_bytes, sha256, adler32 = checksums(local)
    expected_sha256 = row["sha256"].lower()
    if sha256 != expected_sha256:
        raise RuntimeError(
            f"Lineage SHA-256 mismatch for {local}\n"
            f"expected: {expected_sha256}\nobserved: {sha256}"
        )

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "process": row["process"],
        "analysis_role": row["analysis_role"],
        "immutable_split": row["immutable_split"],
        "n_events": int(row["n_events"]),
        "local_path": str(local),
        "remote_lfn": remote,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "adler32": adler32,
    }

    if remote_exists(host, remote):
        observed = remote_checksum(host, remote)
        if observed != adler32:
            raise RuntimeError(
                "Destination exists with a different checksum; refusing "
                f"to overwrite: {remote}\nlocal={adler32} remote={observed}"
            )
        payload.update(
            status="already_verified",
            remote_adler32=observed,
        )
        record(transfer_manifest, payload)
        return payload

    parent = str(PurePosixPath(remote).parent)
    command(["xrdfs", host, "mkdir", "-p", parent])

    partial = remote + f".partial.{uuid.uuid4().hex}"
    remote_url = host.rstrip("/") + "/" + partial

    copied = command(
        [
            "xrdcp",
            "--nopbar",
            "--cksum",
            "adler32:print",
            str(local),
            remote_url,
        ],
        check=False,
    )
    if copied.returncode:
        command(["xrdfs", host, "rm", partial], check=False)
        raise RuntimeError(
            f"xrdcp failed for {local}\n{copied.stdout}{copied.stderr}"
        )

    observed = remote_checksum(host, partial)
    if observed != adler32:
        command(["xrdfs", host, "rm", partial], check=False)
        raise RuntimeError(
            f"Temporary EOS checksum mismatch for {local}: "
            f"local={adler32} remote={observed}"
        )

    command(["xrdfs", host, "mv", partial, remote])
    observed = remote_checksum(host, remote)
    if observed != adler32:
        raise RuntimeError(
            f"Final EOS checksum mismatch for {local}: "
            f"local={adler32} remote={observed}"
        )

    payload.update(
        status="copied_and_verified",
        remote_adler32=observed,
    )
    record(transfer_manifest, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--transfer-manifest", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    host = os.environ.get("HH4B_EOS_HOST", "root://cmseos.fnal.gov")
    base = os.environ.get("HH4B_EOS_PRODUCTION")
    if not base:
        raise SystemExit("Set HH4B_EOS_PRODUCTION.")

    eos_dir = str(PurePosixPath(base) / "hepmc")
    lineage_manifest = Path(args.lineage_manifest)
    transfer_manifest = Path(args.transfer_manifest)

    with lineage_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No canonical rows found.")

    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['filename']}", flush=True)
        payload = stage_one(host, eos_dir, row, transfer_manifest)
        print(
            f"  {payload['status']}: {payload['remote_lfn']} "
            f"({payload['adler32']})",
            flush=True,
        )

    print(f"Verified files: {len(rows)}")
    print(f"Transfer manifest: {transfer_manifest}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)