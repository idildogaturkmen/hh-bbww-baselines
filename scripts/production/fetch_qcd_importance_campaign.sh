#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 RECEIPT_DIR OUTPUT_DIR X509_PROXY" >&2
  exit 2
fi

RECEIPT_DIR="$1"
OUTPUT_DIR="$2"
X509_PROXY="$3"
EOS_HOST="${HH4B_EOS_HOST:-root://cmseos.fnal.gov}"

test -d "$RECEIPT_DIR" || {
  echo "ERROR: missing receipt directory: $RECEIPT_DIR" >&2
  exit 3
}

test -r "$X509_PROXY" || {
  echo "ERROR: missing X.509 proxy: $X509_PROXY" >&2
  exit 3
}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to reuse output path: $OUTPUT_DIR" >&2
  exit 4
fi

for COMMAND in openssl python3 xrdcp xrdfs; do
  command -v "$COMMAND" >/dev/null || {
    echo "ERROR: missing command: $COMMAND" >&2
    exit 3
  }
done

openssl x509 -in "$X509_PROXY" -noout -checkend 3600 >/dev/null || {
  echo "ERROR: X.509 proxy certificate has less than one hour remaining" >&2
  exit 3
}

shopt -s nullglob
RECEIPTS=("$RECEIPT_DIR"/*_receipt.json)
shopt -u nullglob

if [[ "${#RECEIPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no receipts found under $RECEIPT_DIR" >&2
  exit 3
fi

mapfile -t PLAN_LINES < <(
  python3 - "${RECEIPTS[@]}" <<'PY_PLAN'
from pathlib import Path
import json
import re
import sys

expected_bins = set(range(8))
seen_bins = set()
rows = []

for receipt_name in sys.argv[1:]:
    if "\t" in receipt_name or "\n" in receipt_name:
        raise SystemExit("receipt path contains a forbidden control character")
    receipt = json.loads(Path(receipt_name).read_text())
    bin_id = receipt.get("bin_id")
    if type(bin_id) is not int or bin_id not in expected_bins:
        raise SystemExit(f"invalid bin_id in {receipt_name}")
    if bin_id in seen_bins:
        raise SystemExit(f"duplicate receipt for bin {bin_id}")
    if receipt.get("exit_status") != 0:
        raise SystemExit(f"nonzero exit_status in {receipt_name}")
    if receipt.get("stage") != "complete_copied_and_verified":
        raise SystemExit(f"incomplete stage in {receipt_name}")
    remote = receipt.get("remote_bundle")
    adler = receipt.get("adler32")
    if not isinstance(remote, str) or not re.fullmatch(
        r"/store/user/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+_bundle\.tar\.gz", remote
    ):
        raise SystemExit(f"invalid remote_bundle in {receipt_name}")
    if not isinstance(adler, str) or not re.fullmatch(r"[0-9a-fA-F]{8}", adler):
        raise SystemExit(f"invalid Adler-32 in {receipt_name}")
    seen_bins.add(bin_id)
    rows.append((bin_id, remote, adler.lower(), receipt_name))

if seen_bins != expected_bins:
    raise SystemExit(f"receipt bins are {sorted(seen_bins)}, expected 0--7")

for row in sorted(rows):
    print("\t".join(map(str, row)))
PY_PLAN
)

if [[ "${#PLAN_LINES[@]}" -ne 8 ]]; then
  echo "ERROR: receipt preflight did not produce exactly bins 0--7" >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR/bundles" "$OUTPUT_DIR/extracted"
CHECKSUM_MANIFEST="$OUTPUT_DIR/eos_bundle_checksums.csv"
EXTRACTED_MEMBERS_MANIFEST="$OUTPUT_DIR/extracted_file_members.txt"
: > "$EXTRACTED_MEMBERS_MANIFEST"
python3 - "$CHECKSUM_MANIFEST" <<'PY_HEADER'
from pathlib import Path
import csv
import sys

with Path(sys.argv[1]).open("w", newline="") as handle:
    csv.writer(handle).writerow([
        "bin_id", "receipt", "remote_bundle", "local_bundle",
        "receipt_adler32", "eos_adler32", "local_adler32",
        "bundle_bytes", "status",
    ])
PY_HEADER

export X509_USER_PROXY="$X509_PROXY"

for PLAN_LINE in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r BIN_ID REMOTE_BUNDLE RECEIPT_ADLER32 RECEIPT <<< "$PLAN_LINE"
  LOCAL_BUNDLE="$OUTPUT_DIR/bundles/${REMOTE_BUNDLE##*/}"

  if [[ -e "$LOCAL_BUNDLE" ]]; then
    echo "ERROR: duplicate bundle destination: $LOCAL_BUNDLE" >&2
    exit 5
  fi

  EOS_ADLER32=$(
    xrdfs "$EOS_HOST" query checksum "$REMOTE_BUNDLE" |
      awk '{print tolower($NF)}'
  )

  if [[ "$EOS_ADLER32" != "$RECEIPT_ADLER32" ]]; then
    echo "ERROR: current EOS checksum disagrees with receipt: $REMOTE_BUNDLE" >&2
    exit 6
  fi

  xrdcp --nopbar --cksum adler32:print \
    "${EOS_HOST}/${REMOTE_BUNDLE}" \
    "$LOCAL_BUNDLE"

  LOCAL_ADLER32=$(
    python3 - "$LOCAL_BUNDLE" <<'PY_ADLER'
import sys
import zlib

checksum = 1
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        checksum = zlib.adler32(chunk, checksum)
print(f"{checksum & 0xffffffff:08x}")
PY_ADLER
  )

  if [[ "$LOCAL_ADLER32" != "$EOS_ADLER32" ]]; then
    echo "ERROR: downloaded bundle checksum mismatch: $LOCAL_BUNDLE" >&2
    exit 6
  fi

  python3 - \
    "$LOCAL_BUNDLE" \
    "$OUTPUT_DIR/extracted" \
    "$EXTRACTED_MEMBERS_MANIFEST" <<'PY_EXTRACT'
from pathlib import Path, PurePosixPath
import sys
import tarfile

bundle = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3])
existing_files = set(manifest_path.read_text().splitlines())

with tarfile.open(bundle, "r:gz") as archive:
    normalized = set()
    bundle_files = set()
    for member in archive.getmembers():
        name = member.name
        while name.startswith("./"):
            name = name[2:]
        if not name and member.isdir():
            continue
        path = PurePosixPath(name)
        canonical = str(path)
        if canonical == "." and member.isdir():
            continue
        if not name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe tar member: {member.name}")
        if not path.parts:
            raise SystemExit(f"empty tar member path: {member.name}")
        if canonical in normalized:
            raise SystemExit(f"duplicate tar member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported tar member type: {member.name}")
        if path.parts[0] not in {"root", "hepmc", "parquet", "logs", "metadata"}:
            raise SystemExit(f"unexpected tar top-level path: {member.name}")
        normalized.add(canonical)
        if member.isfile():
            bundle_files.add(canonical)
    collisions = bundle_files & existing_files
    if collisions:
        raise SystemExit(f"cross-bundle file collision: {sorted(collisions)}")
    archive.extractall(destination)
    with manifest_path.open("a") as handle:
        for name in sorted(bundle_files):
            handle.write(name + "\n")
PY_EXTRACT
  BUNDLE_BYTES=$(stat -c '%s' "$LOCAL_BUNDLE")

  python3 - "$CHECKSUM_MANIFEST" \
    "$BIN_ID" \
    "$RECEIPT" \
    "$REMOTE_BUNDLE" \
    "$LOCAL_BUNDLE" \
    "$RECEIPT_ADLER32" \
    "$EOS_ADLER32" \
    "$LOCAL_ADLER32" \
    "$BUNDLE_BYTES" <<'PY_ROW'
from pathlib import Path
import csv
import sys

with Path(sys.argv[1]).open("a", newline="") as handle:
    csv.writer(handle).writerow([*sys.argv[2:], "pass"])
PY_ROW
done

echo "Downloaded and extracted ${#PLAN_LINES[@]} verified bundles"
echo "Checksum manifest: $CHECKSUM_MANIFEST"
echo "Extracted campaign: $OUTPUT_DIR/extracted"
