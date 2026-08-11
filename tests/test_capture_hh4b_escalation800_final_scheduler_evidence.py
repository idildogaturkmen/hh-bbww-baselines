from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from capture_hh4b_escalation800_final_scheduler_evidence import (  # noqa: E402
    CaptureError,
    PACKAGE_ROOT,
    RECOVERY_CMD,
    RUNTIME,
    validate_history_text,
)


def history_payload(rows: int = 8000) -> bytes:
    transfer = f"{PACKAGE_ROOT / 'submission/run_escalation800_category_fold_job.sh'},{RUNTIME}"
    return "".join(
        f"30020809 {proc} 4 0 false 1 {RECOVERY_CMD} {transfer} "
        f"{PACKAGE_ROOT / f'returns/job_{proc:04d}'}\n"
        for proc in range(rows)
    ).encode("utf-8")


def test_exact_history_identity_closure_passes() -> None:
    validate_history_text(history_payload())


def test_missing_history_row_fails_closed() -> None:
    try:
        validate_history_text(history_payload(7999))
    except CaptureError:
        pass
    else:
        raise AssertionError("incomplete scheduler history passed")


def test_nonclean_or_restarted_history_fails_closed() -> None:
    clean = history_payload()
    first = clean.splitlines()[0]
    for bad in (
        first.replace(b" 4 0 false 1 ", b" 3 0 false 1 "),
        first.replace(b" 4 0 false 1 ", b" 4 1 false 1 "),
        first.replace(b" 4 0 false 1 ", b" 4 0 true 1 "),
        first.replace(b" 4 0 false 1 ", b" 4 0 false 2 "),
    ):
        payload = bad + b"\n" + b"\n".join(clean.splitlines()[1:]) + b"\n"
        try:
            validate_history_text(payload)
        except CaptureError:
            pass
        else:
            raise AssertionError("nonclean/restarted scheduler history passed")


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
