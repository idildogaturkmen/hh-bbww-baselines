#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from hh4b_sophon_canary_config import (
    CONFIG_SCHEMA_VERSION,
    RELEASE_MODEL_COUNT,
    RemoteCanaryConfig,
    build_execution_plan,
    load_canary_config,
    validate_model_files,
)


PathLike = Union[str, Path]

RUNNER_SCHEMA_VERSION = 1
DRY_RUN_STATUS = "dry_run_validated"


def sha256_file(
    path: PathLike,
) -> str:
    """Return the SHA-256 digest of one regular file."""

    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            str(
                file_path
            )
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_json(
    value: Mapping[str, Any],
) -> str:
    """Return deterministic human-readable JSON."""

    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
    )


def validate_dry_run_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Validate the safety-critical dry-run receipt fields."""

    required_keys = {
        "runner_schema_version",
        "config_schema_version",
        "status",
        "config_path",
        "config_sha256",
        "execution_plan",
        "model_receipts",
        "operations",
    }

    actual_keys = set(
        receipt
    )

    missing = sorted(
        required_keys
        - actual_keys
    )

    extra = sorted(
        actual_keys
        - required_keys
    )

    if missing:
        raise KeyError(
            "dry-run receipt is missing keys: {}".format(
                missing
            )
        )

    if extra:
        raise KeyError(
            "dry-run receipt has unknown keys: {}".format(
                extra
            )
        )

    if receipt[
        "runner_schema_version"
    ] != RUNNER_SCHEMA_VERSION:
        raise ValueError(
            "runner schema-version mismatch"
        )

    if receipt[
        "config_schema_version"
    ] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "config schema-version mismatch"
        )

    if receipt[
        "status"
    ] != DRY_RUN_STATUS:
        raise ValueError(
            "dry-run status mismatch"
        )

    execution_plan = receipt[
        "execution_plan"
    ]

    if not isinstance(
        execution_plan,
        Mapping,
    ):
        raise TypeError(
            "execution_plan must be an object"
        )

    if execution_plan.get(
        "execution_mode"
    ) != "dry_run":
        raise ValueError(
            "receipt execution mode must be dry_run"
        )

    if execution_plan.get(
        "execution_permitted"
    ) is not False:
        raise ValueError(
            "execution must not be permitted in dry-run mode"
        )

    model_receipts = receipt[
        "model_receipts"
    ]

    if (
        not isinstance(
            model_receipts,
            list,
        )
        or len(
            model_receipts
        ) != RELEASE_MODEL_COUNT
    ):
        raise ValueError(
            "receipt must contain exactly {} model receipts".format(
                RELEASE_MODEL_COUNT
            )
        )

    expected_indices = list(
        range(
            RELEASE_MODEL_COUNT
        )
    )

    actual_indices = [
        model_receipt.get(
            "model_index"
        )
        for model_receipt in model_receipts
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "model receipt indices are not ordered as {}".format(
                expected_indices
            )
        )

    for model_receipt in model_receipts:
        if model_receipt.get(
            "size_bytes",
            0,
        ) <= 0:
            raise ValueError(
                "model receipt has a non-positive file size"
            )

        sha256 = model_receipt.get(
            "sha256"
        )

        if (
            not isinstance(
                sha256,
                str,
            )
            or len(
                sha256
            ) != 64
        ):
            raise ValueError(
                "model receipt has an invalid SHA-256 digest"
            )

    operations = receipt[
        "operations"
    ]

    expected_operations = {
        "configuration_loaded": True,
        "configuration_validated": True,
        "model_files_hashed": True,
        "network_accessed": False,
        "remote_root_opened": False,
        "onnxruntime_imported": False,
        "onnx_inference_run": False,
        "events_tree_written": False,
    }

    if operations != expected_operations:
        raise ValueError(
            "dry-run operation flags do not match the safety contract"
        )


def build_dry_run_receipt(
    config_path: PathLike,
) -> Dict[str, Any]:
    """Validate a canary configuration without network or inference."""

    resolved_config_path = Path(
        config_path
    ).resolve(
        strict=True
    )

    config = load_canary_config(
        resolved_config_path
    )

    if config.execution_mode != "dry_run":
        raise PermissionError(
            "this runner currently supports dry_run only; "
            "remote ROOT access and ONNX inference remain disabled"
        )

    execution_plan = build_execution_plan(
        config
    )

    if execution_plan[
        "execution_permitted"
    ]:
        raise RuntimeError(
            "dry-run execution plan unexpectedly permits execution"
        )

    model_receipts = list(
        validate_model_files(
            config
        )
    )

    receipt: Dict[
        str,
        Any,
    ] = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "status": DRY_RUN_STATUS,
        "config_path": str(
            resolved_config_path
        ),
        "config_sha256": sha256_file(
            resolved_config_path
        ),
        "execution_plan": execution_plan,
        "model_receipts": model_receipts,
        "operations": {
            "configuration_loaded": True,
            "configuration_validated": True,
            "model_files_hashed": True,
            "network_accessed": False,
            "remote_root_opened": False,
            "onnxruntime_imported": False,
            "onnx_inference_run": False,
            "events_tree_written": False,
        },
    }

    validate_dry_run_receipt(
        receipt
    )

    return receipt


def _validated_receipt_path(
    receipt_path: PathLike,
    *,
    temporary_root: PathLike,
) -> Path:
    """Require an absolute JSON receipt path below temporary_root."""

    candidate = Path(
        receipt_path
    )

    if not candidate.is_absolute():
        raise ValueError(
            "receipt path must be absolute"
        )

    root = Path(
        temporary_root
    ).resolve(
        strict=False
    )

    resolved_candidate = candidate.resolve(
        strict=False
    )

    try:
        resolved_candidate.relative_to(
            root
        )
    except ValueError as error:
        raise ValueError(
            "receipt path must remain under temporary_root"
        ) from error

    if resolved_candidate == root:
        raise ValueError(
            "receipt path must identify a JSON file below temporary_root"
        )

    if resolved_candidate.suffix.lower() != ".json":
        raise ValueError(
            "receipt path must end in .json"
        )

    return resolved_candidate


def write_receipt_exclusive(
    receipt: Mapping[str, Any],
    receipt_path: PathLike,
    *,
    temporary_root: PathLike,
) -> Path:
    """Write a receipt atomically without replacing an existing file."""

    validate_dry_run_receipt(
        receipt
    )

    output_path = _validated_receipt_path(
        receipt_path,
        temporary_root=temporary_root,
    )

    root = Path(
        temporary_root
    ).resolve(
        strict=False
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not root.is_dir():
        raise NotADirectoryError(
            str(
                root
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_parent = output_path.parent.resolve(
        strict=True
    )

    try:
        resolved_parent.relative_to(
            root
        )
    except ValueError as error:
        raise ValueError(
            "receipt parent escaped temporary_root"
        ) from error

    if output_path.exists():
        raise FileExistsError(
            str(
                output_path
            )
        )

    payload = (
        canonical_json(
            receipt
        )
        + "\n"
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sophon_canary_receipt_",
        suffix=".tmp",
        dir=str(
            resolved_parent
        ),
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                payload
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.link(
            temporary_path,
            output_path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


def run_dry_run(
    config_path: PathLike,
    *,
    receipt_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Run configuration/model validation and optionally write a receipt."""

    receipt = build_dry_run_receipt(
        config_path
    )

    if receipt_path is not None:
        temporary_root = receipt[
            "execution_plan"
        ][
            "temporary_root"
        ]

        write_receipt_exclusive(
            receipt,
            receipt_path,
            temporary_root=temporary_root,
        )

    return receipt


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a SophonHH remote-canary configuration "
            "without network access or ONNX inference."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the remote-canary JSON configuration.",
    )

    parser.add_argument(
        "--receipt",
        help=(
            "Optional absolute JSON output path below the "
            "configured temporary_root."
        ),
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    try:
        receipt = run_dry_run(
            arguments.config,
            receipt_path=arguments.receipt,
        )
    except Exception as error:
        print(
            "ERROR: {}: {}".format(
                type(error).__name__,
                error,
            ),
            file=sys.stderr,
        )

        return 2

    print(
        canonical_json(
            receipt
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
