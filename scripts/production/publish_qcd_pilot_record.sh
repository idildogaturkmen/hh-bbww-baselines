#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 LOCAL_RECORD_DIR EOS_RECORD_DIR X509_PROXY [--verify-existing]" >&2
  exit 2
fi

LOCAL_RECORD_DIR="$1"
EOS_RECORD_DIR="$2"
X509_PROXY="$3"
VERIFY_EXISTING=0
if [[ "$#" -eq 4 ]]; then
  [[ "$4" == "--verify-existing" ]] || {
    echo "ERROR: the only supported optional argument is --verify-existing" >&2
    exit 2
  }
  VERIFY_EXISTING=1
fi
EOS_HOST="${HH4B_EOS_HOST:-root://cmseos.fnal.gov}"
CHECKSUM_MANIFEST="$LOCAL_RECORD_DIR/checksums.csv"
VERIFICATION_MANIFEST="$LOCAL_RECORD_DIR/eos_verification.csv"

test -d "$LOCAL_RECORD_DIR" || {
  echo "ERROR: missing local record directory: $LOCAL_RECORD_DIR" >&2
  exit 3
}

test -s "$CHECKSUM_MANIFEST" || {
  echo "ERROR: missing checksum manifest: $CHECKSUM_MANIFEST" >&2
  exit 3
}

test -r "$X509_PROXY" || {
  echo "ERROR: missing X.509 proxy: $X509_PROXY" >&2
  exit 3
}

[[ "$EOS_RECORD_DIR" =~ ^/store/user/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+$ ]] || {
  echo "ERROR: invalid EOS record directory: $EOS_RECORD_DIR" >&2
  exit 3
}

for COMMAND in cmp mktemp openssl python3 sha256sum stat xrdcp xrdfs; do
  command -v "$COMMAND" >/dev/null || {
    echo "ERROR: missing command: $COMMAND" >&2
    exit 3
  }
done

openssl x509 -in "$X509_PROXY" -noout -checkend 3600 >/dev/null || {
  echo "ERROR: X.509 proxy certificate has less than one hour remaining" >&2
  exit 3
}

mapfile -t PLAN_LINES < <(
  python3 - "$LOCAL_RECORD_DIR" "$CHECKSUM_MANIFEST" <<'PY_PLAN'
from pathlib import Path, PurePosixPath
import csv
import hashlib
import sys
import zlib

root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
rows = list(csv.DictReader(manifest.open(newline="")))
expected_header = ["relative_path", "bytes", "sha256", "adler32"]
if (list(rows[0]) if rows else []) != expected_header:
    raise SystemExit("invalid or empty checksum manifest")

seen = set()
for row in rows:
    relative = PurePosixPath(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SystemExit(f"unsafe manifest path: {relative}")
    canonical = relative.as_posix()
    if canonical in seen or canonical in {"checksums.csv", "eos_verification.csv"}:
        raise SystemExit(f"duplicate or reserved manifest path: {canonical}")
    seen.add(canonical)
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or unsupported record file: {canonical}")
    sha = hashlib.sha256()
    adler = 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            adler = zlib.adler32(chunk, adler)
    observed = (
        str(path.stat().st_size),
        sha.hexdigest(),
        f"{adler & 0xffffffff:08x}",
    )
    expected = (row["bytes"], row["sha256"].lower(), row["adler32"].lower())
    if observed != expected:
        raise SystemExit(f"checksum manifest mismatch: {canonical}")
    if "\t" in canonical or "\n" in canonical:
        raise SystemExit(f"control character in record path: {canonical}")
    print("\t".join((canonical, *observed)))

actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name not in {"checksums.csv", "eos_verification.csv"}
}
if actual != seen:
    raise SystemExit(
        f"record directory differs from manifest: missing={sorted(seen-actual)}, "
        f"extra={sorted(actual-seen)}"
    )
PY_PLAN
)

if [[ "${#PLAN_LINES[@]}" -eq 0 ]]; then
  echo "ERROR: no compact record files passed checksum validation" >&2
  exit 4
fi

export X509_USER_PROXY="$X509_PROXY"

if [[ "$VERIFY_EXISTING" -eq 0 ]]; then
  if xrdfs "$EOS_HOST" stat "$EOS_RECORD_DIR" >/dev/null 2>&1; then
    echo "ERROR: refusing to reuse EOS record directory: $EOS_RECORD_DIR" >&2
    exit 5
  fi
  xrdfs "$EOS_HOST" mkdir -p "$EOS_RECORD_DIR"
else
  xrdfs "$EOS_HOST" stat "$EOS_RECORD_DIR" >/dev/null || {
    echo "ERROR: missing EOS record directory: $EOS_RECORD_DIR" >&2
    exit 5
  }
fi

VERIFY_TMP=$(mktemp -d)
cleanup() {
  rm -rf "$VERIFY_TMP"
}
trap cleanup EXIT

printf '%s\n' \
  'relative_path,bytes,sha256,local_adler32,eos_adler32,status' \
  > "$VERIFICATION_MANIFEST"

for PLAN_LINE in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r RELATIVE BYTES SHA256 LOCAL_ADLER32 <<< "$PLAN_LINE"
  LOCAL_FILE="$LOCAL_RECORD_DIR/$RELATIVE"
  REMOTE_FILE="$EOS_RECORD_DIR/$RELATIVE"
  REMOTE_PARENT=${REMOTE_FILE%/*}

  if [[ "$VERIFY_EXISTING" -eq 0 ]]; then
    xrdfs "$EOS_HOST" mkdir -p "$REMOTE_PARENT"
    xrdcp --nopbar --cksum adler32:print \
      "$LOCAL_FILE" \
      "${EOS_HOST}/${REMOTE_FILE}"
  fi
  EOS_ADLER32=$(
    xrdfs "$EOS_HOST" query checksum "$REMOTE_FILE" |
      awk '{print tolower($NF)}'
  )
  if [[ "$EOS_ADLER32" != "$LOCAL_ADLER32" ]]; then
    echo "ERROR: EOS checksum mismatch: $RELATIVE" >&2
    exit 6
  fi
  if [[ "$VERIFY_EXISTING" -eq 1 ]]; then
    VERIFY_FILE="$VERIFY_TMP/$RELATIVE"
    mkdir -p "${VERIFY_FILE%/*}"
    xrdcp --nopbar --cksum adler32:print \
      "${EOS_HOST}/${REMOTE_FILE}" \
      "$VERIFY_FILE"
    VERIFY_BYTES=$(stat -c '%s' "$VERIFY_FILE")
    VERIFY_SHA256=$(sha256sum "$VERIFY_FILE" | awk '{print $1}')
    if [[ "$VERIFY_BYTES" != "$BYTES" || "$VERIFY_SHA256" != "$SHA256" ]]; then
      echo "ERROR: downloaded EOS content mismatch: $RELATIVE" >&2
      exit 6
    fi
  fi
  printf '%s,%s,%s,%s,%s,pass\n' \
    "$RELATIVE" "$BYTES" "$SHA256" "$LOCAL_ADLER32" "$EOS_ADLER32" \
    >> "$VERIFICATION_MANIFEST"
done

MANIFEST_ADLER32=$(
  python3 - "$CHECKSUM_MANIFEST" <<'PY_ADLER'
import sys
import zlib

checksum = 1
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        checksum = zlib.adler32(chunk, checksum)
print(f"{checksum & 0xffffffff:08x}")
PY_ADLER
)

if [[ "$VERIFY_EXISTING" -eq 0 ]]; then
  xrdcp --nopbar --cksum adler32:print \
    "$CHECKSUM_MANIFEST" \
    "${EOS_HOST}/${EOS_RECORD_DIR}/checksums.csv"
fi
EOS_MANIFEST_ADLER32=$(
  xrdfs "$EOS_HOST" query checksum "$EOS_RECORD_DIR/checksums.csv" |
    awk '{print tolower($NF)}'
)
if [[ "$EOS_MANIFEST_ADLER32" != "$MANIFEST_ADLER32" ]]; then
  echo "ERROR: EOS checksum mismatch: checksums.csv" >&2
  exit 6
fi
if [[ "$VERIFY_EXISTING" -eq 1 ]]; then
  VERIFY_CHECKSUM_MANIFEST="$VERIFY_TMP/checksums.csv"
  xrdcp --nopbar --cksum adler32:print \
    "${EOS_HOST}/${EOS_RECORD_DIR}/checksums.csv" \
    "$VERIFY_CHECKSUM_MANIFEST"
  if ! cmp -s "$CHECKSUM_MANIFEST" "$VERIFY_CHECKSUM_MANIFEST"; then
    echo "ERROR: downloaded EOS content mismatch: checksums.csv" >&2
    exit 6
  fi
fi

printf '%s,%s,%s,%s,%s,pass\n' \
  "checksums.csv" \
  "$(stat -c '%s' "$CHECKSUM_MANIFEST")" \
  "$(sha256sum "$CHECKSUM_MANIFEST" | awk '{print $1}')" \
  "$MANIFEST_ADLER32" \
  "$EOS_MANIFEST_ADLER32" \
  >> "$VERIFICATION_MANIFEST"

REMOTE_FILE_COUNT=$(
  xrdfs "$EOS_HOST" ls -l -R "$EOS_RECORD_DIR" |
    awk '$1 ~ /^-/ {count += 1} END {print count + 0}'
)
EXPECTED_FILE_COUNT=$((${#PLAN_LINES[@]} + 1))
if [[ "$REMOTE_FILE_COUNT" -ne "$EXPECTED_FILE_COUNT" ]]; then
  echo "ERROR: remote file count is $REMOTE_FILE_COUNT, expected $EXPECTED_FILE_COUNT" >&2
  exit 7
fi

if [[ "$VERIFY_EXISTING" -eq 1 ]]; then
  echo "Re-downloaded and verified $EXPECTED_FILE_COUNT compact pilot-record files"
else
  echo "Published and verified $EXPECTED_FILE_COUNT compact pilot-record files"
fi
echo "EOS record: $EOS_RECORD_DIR"
echo "Verification manifest: $VERIFICATION_MANIFEST"
