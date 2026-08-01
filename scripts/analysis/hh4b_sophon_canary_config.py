#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Tuple, Union
from urllib.parse import urlsplit


PathLike = Union[str, Path]

CONFIG_SCHEMA_VERSION = 1
RELEASE_MODEL_COUNT = 3
MAX_REQUESTED_EVENTS = 10000

ALLOWED_SAMPLE_ROLES = (
    "control",
    "independent_hh",
)

ALLOWED_TREE_POLICIES = (
    "highest_cycle",
)

ALLOWED_EXECUTION_MODES = (
    "dry_run",
    "execute",
)

CONFIG_KEYS = (
    "schema_version",
    "sample_name",
    "sample_role",
    "endpoint",
    "remote_path",
    "tree_policy",
    "requested_events",
    "model_paths",
    "temporary_root",
    "execution_mode",
    "allow_independent_hh_execution",
    "write_events_tree",
)


@dataclass(
    frozen=True
)
class RemoteCanaryConfig:
    schema_version: int
    sample_name: str
    sample_role: str
    endpoint: str
    remote_path: str
    tree_policy: str
    requested_events: int
    model_paths: Tuple[str, str, str]
    temporary_root: str
    execution_mode: str
    allow_independent_hh_execution: bool
    write_events_tree: bool

    @property
    def xrootd_url(
        self,
    ) -> str:
        return (
            self.endpoint
            + self.remote_path
        )

    @property
    def execution_permitted(
        self,
    ) -> bool:
        if self.execution_mode != "execute":
            return False

        if self.sample_role == "independent_hh":
            return self.allow_independent_hh_execution

        return True


def _require_nonempty_string(
    value: Any,
    *,
    name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "{} must be a string".format(
                name
            )
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "{} must not be empty".format(
                name
            )
        )

    if normalized != value:
        raise ValueError(
            "{} must not contain leading or trailing whitespace".format(
                name
            )
        )

    return normalized


def _require_boolean(
    value: Any,
    *,
    name: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            "{} must be a JSON boolean".format(
                name
            )
        )

    return value


def _require_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            "{} must be an integer".format(
                name
            )
        )

    return value


def _normalize_endpoint(
    value: Any,
) -> str:
    endpoint = _require_nonempty_string(
        value,
        name="endpoint",
    )

    parsed = urlsplit(
        endpoint
    )

    if parsed.scheme != "root":
        raise ValueError(
            "endpoint must use the root:// scheme"
        )

    if not parsed.netloc:
        raise ValueError(
            "endpoint must include a host"
        )

    if parsed.path not in (
        "",
        "/",
    ):
        raise ValueError(
            "endpoint must not contain a dataset path"
        )

    if (
        parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint must not contain a query or fragment"
        )

    return "root://{}".format(
        parsed.netloc
    )


def _normalize_remote_path(
    value: Any,
) -> str:
    remote_path = _require_nonempty_string(
        value,
        name="remote_path",
    )

    if not remote_path.startswith(
        "/"
    ):
        raise ValueError(
            "remote_path must be absolute"
        )

    path = PurePosixPath(
        remote_path
    )

    if ".." in path.parts:
        raise ValueError(
            "remote_path must not contain '..'"
        )

    normalized = str(
        path
    )

    if normalized == "/":
        raise ValueError(
            "remote_path must identify a ROOT file"
        )

    if not normalized.lower().endswith(
        ".root"
    ):
        raise ValueError(
            "remote_path must end in .root"
        )

    return normalized


def _normalize_temporary_root(
    value: Any,
) -> str:
    text = _require_nonempty_string(
        value,
        name="temporary_root",
    )

    path = Path(
        text
    )

    if not path.is_absolute():
        raise ValueError(
            "temporary_root must be absolute"
        )

    resolved = path.resolve(
        strict=False
    )

    system_tmp = Path(
        "/tmp"
    ).resolve()

    try:
        resolved.relative_to(
            system_tmp
        )
    except ValueError as error:
        raise ValueError(
            "temporary_root must be under /tmp"
        ) from error

    if resolved == system_tmp:
        raise ValueError(
            "temporary_root must be a dedicated path below /tmp"
        )

    return str(
        resolved
    )


def _normalize_model_paths(
    value: Any,
) -> Tuple[str, str, str]:
    if (
        not isinstance(
            value,
            list,
        )
        and not isinstance(
            value,
            tuple,
        )
    ):
        raise TypeError(
            "model_paths must be a JSON array"
        )

    if len(value) != RELEASE_MODEL_COUNT:
        raise ValueError(
            "model_paths must contain exactly {} released models".format(
                RELEASE_MODEL_COUNT
            )
        )

    normalized = []

    for model_index, raw_path in enumerate(
        value
    ):
        text = _require_nonempty_string(
            raw_path,
            name="model_paths[{}]".format(
                model_index
            ),
        )

        path = Path(
            text
        )

        if not path.is_absolute():
            raise ValueError(
                "model_paths[{}] must be absolute".format(
                    model_index
                )
            )

        if path.suffix.lower() != ".onnx":
            raise ValueError(
                "model_paths[{}] must end in .onnx".format(
                    model_index
                )
            )

        normalized.append(
            str(
                path.resolve(
                    strict=False
                )
            )
        )

    if len(
        set(
            normalized
        )
    ) != RELEASE_MODEL_COUNT:
        raise ValueError(
            "model_paths must identify three distinct files"
        )

    return (
        normalized[0],
        normalized[1],
        normalized[2],
    )


def parse_canary_config(
    raw: Mapping[str, Any],
) -> RemoteCanaryConfig:
    if not isinstance(
        raw,
        Mapping,
    ):
        raise TypeError(
            "configuration must be a JSON object"
        )

    expected = set(
        CONFIG_KEYS
    )

    actual = set(
        raw
    )

    missing = sorted(
        expected
        - actual
    )

    extra = sorted(
        actual
        - expected
    )

    if missing:
        raise KeyError(
            "configuration is missing keys: {}".format(
                missing
            )
        )

    if extra:
        raise KeyError(
            "configuration has unknown keys: {}".format(
                extra
            )
        )

    schema_version = _require_integer(
        raw["schema_version"],
        name="schema_version",
    )

    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be {}, got {}".format(
                CONFIG_SCHEMA_VERSION,
                schema_version,
            )
        )

    sample_name = _require_nonempty_string(
        raw["sample_name"],
        name="sample_name",
    )

    sample_role = _require_nonempty_string(
        raw["sample_role"],
        name="sample_role",
    )

    if sample_role not in ALLOWED_SAMPLE_ROLES:
        raise ValueError(
            "sample_role must be one of {}".format(
                ALLOWED_SAMPLE_ROLES
            )
        )

    endpoint = _normalize_endpoint(
        raw["endpoint"]
    )

    remote_path = _normalize_remote_path(
        raw["remote_path"]
    )

    tree_policy = _require_nonempty_string(
        raw["tree_policy"],
        name="tree_policy",
    )

    if tree_policy not in ALLOWED_TREE_POLICIES:
        raise ValueError(
            "tree_policy must be one of {}".format(
                ALLOWED_TREE_POLICIES
            )
        )

    requested_events = _require_integer(
        raw["requested_events"],
        name="requested_events",
    )

    if (
        requested_events <= 0
        or requested_events > MAX_REQUESTED_EVENTS
    ):
        raise ValueError(
            "requested_events must be in [1, {}]".format(
                MAX_REQUESTED_EVENTS
            )
        )

    model_paths = _normalize_model_paths(
        raw["model_paths"]
    )

    temporary_root = _normalize_temporary_root(
        raw["temporary_root"]
    )

    execution_mode = _require_nonempty_string(
        raw["execution_mode"],
        name="execution_mode",
    )

    if execution_mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(
            "execution_mode must be one of {}".format(
                ALLOWED_EXECUTION_MODES
            )
        )

    allow_independent = _require_boolean(
        raw[
            "allow_independent_hh_execution"
        ],
        name="allow_independent_hh_execution",
    )

    write_events_tree = _require_boolean(
        raw["write_events_tree"],
        name="write_events_tree",
    )

    if (
        sample_role == "control"
        and allow_independent
    ):
        raise ValueError(
            "allow_independent_hh_execution must be false "
            "for a control sample"
        )

    if (
        sample_role == "independent_hh"
        and execution_mode == "execute"
        and not allow_independent
    ):
        raise PermissionError(
            "independent-HH execution is disabled; an authoritative "
            "sample path must be supplied and explicitly enabled"
        )

    return RemoteCanaryConfig(
        schema_version=schema_version,
        sample_name=sample_name,
        sample_role=sample_role,
        endpoint=endpoint,
        remote_path=remote_path,
        tree_policy=tree_policy,
        requested_events=requested_events,
        model_paths=model_paths,
        temporary_root=temporary_root,
        execution_mode=execution_mode,
        allow_independent_hh_execution=allow_independent,
        write_events_tree=write_events_tree,
    )


def load_canary_config(
    path: PathLike,
) -> RemoteCanaryConfig:
    config_path = Path(
        path
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            str(
                config_path
            )
        )

    try:
        raw = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "invalid JSON in {}".format(
                config_path
            )
        ) from error

    return parse_canary_config(
        raw
    )


def build_execution_plan(
    config: RemoteCanaryConfig,
) -> Dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "sample_name": config.sample_name,
        "sample_role": config.sample_role,
        "xrootd_url": config.xrootd_url,
        "tree_policy": config.tree_policy,
        "requested_events": config.requested_events,
        "model_paths": list(
            config.model_paths
        ),
        "released_model_count": RELEASE_MODEL_COUNT,
        "temporary_root": config.temporary_root,
        "execution_mode": config.execution_mode,
        "execution_permitted": config.execution_permitted,
        "write_events_tree": config.write_events_tree,
    }


def validate_model_files(
    config: RemoteCanaryConfig,
) -> Tuple[Dict[str, Any], ...]:
    receipts = []

    for model_index, path_text in enumerate(
        config.model_paths
    ):
        path = Path(
            path_text
        )

        if not path.is_file():
            raise FileNotFoundError(
                "released model {} is missing: {}".format(
                    model_index,
                    path
                )
            )

        size_bytes = int(
            path.stat().st_size
        )

        if size_bytes <= 0:
            raise ValueError(
                "released model {} is empty: {}".format(
                    model_index,
                    path
                )
            )

        digest = hashlib.sha256()

        with path.open(
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

        receipts.append(
            {
                "model_index": model_index,
                "path": str(
                    path
                ),
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
            }
        )

    return tuple(
        receipts
    )
