from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from query_hh4b_cut_validation_remote_checksums import (  # noqa: E402
    lfn_from_uri,
    parse_checksum,
)


def test_lfn_normalization() -> None:
    assert lfn_from_uri(
        "root://cmseos.fnal.gov//store/user/example/bundle.tar.gz"
    ) == "/store/user/example/bundle.tar.gz"


def test_adler32_parse() -> None:
    assert parse_checksum("adler32 5A5DCABB\n") == ("adler32", "5a5dcabb")
    assert parse_checksum("adler32 0x00000001") == ("adler32", "00000001")


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
