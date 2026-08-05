#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import zlib

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot


MAX_EVENTS_PER_SOURCE = 512
JET_PT_MIN_GEV = 30.0
JET_ABS_ETA_MAX = 2.5
TRUTH_MATCH_DR_MAX = 0.4

OUTPUT_COLUMNS = [
    "source_uid",
    "group_id",
    "source_entry",
    "event_uid",
    "sample_class",
    "process_or_mode",
    "workflow_population",
    "class_label",
    "auxiliary_qcd",
    "physical_evaluation_eligible",
    "raw_event_weight_available",
    "raw_event_weight",
    "n_selected_jets",
    "n_selected_bjets",
    "broad_event_eligible",
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_btag",
    "jet_mask",
    "jet_truth_higgs_slot",
    "assignment_matchable",
    "truth_direct_higgs_b_count",
    "truth_higgs_mother_count",
]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value) -> str:
    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def truthy(value) -> bool:
    return clean(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def member_name(value) -> str:
    text = clean(value).replace(
        "\\",
        "/",
    )

    while text.startswith("./"):
        text = text[2:]

    text = text.lstrip("/")

    if (
        text in {"", "."}
        or text.endswith("/")
    ):
        return ""

    require(
        (
            not text.startswith("../")
            and "/../" not in text
        ),
        f"unsafe archive member: {value}",
    )

    return text


def normalize_checksum(value) -> str:
    text = clean(value).lower()

    for prefix in (
        "sha256:",
        "sha-256:",
        "adler32:",
        "adler:",
        "md5:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    if "=" in text:
        left, right = text.split(
            "=",
            1,
        )

        if any(
            token in left
            for token in (
                "sha",
                "adler",
                "md5",
            )
        ):
            text = right

    if text.startswith("0x"):
        text = text[2:]

    return "".join(
        character
        for character in text
        if character
        in "0123456789abcdef"
    )


def file_digest(
    path: Path,
    kind: str,
) -> tuple[str, str]:
    lowered = clean(kind).lower()

    if "adler" in lowered:
        value = 1

        with path.open("rb") as handle:
            for block in iter(
                lambda: handle.read(
                    8 * 1024 * 1024
                ),
                b"",
            ):
                value = zlib.adler32(
                    block,
                    value,
                )

        return (
            "adler32",
            f"{value & 0xFFFFFFFF:08x}",
        )

    digest = (
        hashlib.md5()
        if "md5" in lowered
        else hashlib.sha256()
    )

    name = (
        "md5"
        if "md5" in lowered
        else "sha256"
    )

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return name, digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def copy_remote(
    uri: str,
    destination: Path,
) -> None:
    completed = subprocess.run(
        [
            "xrdcp",
            "-f",
            uri,
            str(destination),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
        check=False,
    )

    require(
        completed.returncode == 0,
        (
            f"xrdcp failed for {uri}: "
            f"{completed.stderr[-4000:]}"
        ),
    )

    require(
        (
            destination.is_file()
            and destination.stat().st_size
            > 0
        ),
        (
            "xrdcp produced no usable "
            f"file for {uri}"
        ),
    )


def extract_entry(
    archive: tarfile.TarFile,
    info: tarfile.TarInfo,
    destination: Path,
) -> None:
    source = archive.extractfile(info)

    require(
        source is not None,
        f"cannot extract {info.name}",
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        source,
        destination.open("wb")
        as output,
    ):
        shutil.copyfileobj(
            source,
            output,
        )

    require(
        destination.stat().st_size > 0,
        f"empty archive member: {info.name}",
    )


def resolve_root(
    archive_path: Path,
    work: Path,
    frozen_member: str,
    frozen_basename: str,
    chain: str,
    depth: int = 0,
) -> tuple[Path, str, str]:
    require(
        depth <= 4,
        "nested archive depth exceeded",
    )

    require(
        tarfile.is_tarfile(
            archive_path
        ),
        f"not a tar archive: {archive_path}",
    )

    nested_suffixes = (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tar.xz",
        ".tar.zst",
    )

    roots: list[
        tuple[
            Path,
            str,
            str,
            bool,
            bool,
        ]
    ] = []

    nested_files: list[
        tuple[Path, str]
    ] = []

    with tarfile.open(
        archive_path,
        "r:*",
    ) as archive:
        files: list[
            tuple[
                tarfile.TarInfo,
                str,
            ]
        ] = []

        for info in archive.getmembers():
            if not info.isfile():
                continue

            name = member_name(
                info.name
            )

            if name:
                files.append(
                    (
                        info,
                        name,
                    )
                )

        direct = []
        nested = []

        for info, name in files:
            basename = Path(
                name
            ).name

            exact_member = bool(
                frozen_member
                and name == frozen_member
            )

            exact_basename = bool(
                frozen_basename
                and basename
                == frozen_basename
            )

            if (
                exact_member
                or exact_basename
                or name.lower().endswith(
                    ".root"
                )
            ):
                direct.append(
                    (
                        info,
                        name,
                        exact_member,
                        exact_basename,
                    )
                )

            if name.lower().endswith(
                nested_suffixes
            ):
                nested.append(
                    (
                        info,
                        name,
                    )
                )

        direct.sort(
            key=lambda item: (
                0
                if item[2]
                else (
                    1
                    if item[3]
                    else 2
                ),
                item[1],
            )
        )

        for (
            index,
            (
                info,
                name,
                exact_member,
                exact_basename,
            ),
        ) in enumerate(direct):
            destination = (
                work
                / (
                    f"d{depth}_root_"
                    f"{index:03d}.bin"
                )
            )

            extract_entry(
                archive,
                info,
                destination,
            )

            with destination.open(
                "rb"
            ) as handle:
                magic = handle.read(4)

            if magic == b"root":
                roots.append(
                    (
                        destination,
                        f"{chain}::{name}",
                        "tar_member",
                        exact_member,
                        exact_basename,
                    )
                )

        exact_matches = [
            item
            for item in roots
            if item[3]
        ]

        basename_matches = [
            item
            for item in roots
            if item[4]
        ]

        if len(exact_matches) == 1:
            return exact_matches[0][:3]

        if len(basename_matches) == 1:
            return basename_matches[0][:3]

        if len(roots) == 1:
            return roots[0][:3]

        for (
            index,
            (
                info,
                name,
            ),
        ) in enumerate(
            sorted(
                nested,
                key=lambda item: item[1],
            )
        ):
            destination = (
                work
                / (
                    f"d{depth}_nested_"
                    f"{index:03d}.bin"
                )
            )

            extract_entry(
                archive,
                info,
                destination,
            )

            if tarfile.is_tarfile(
                destination
            ):
                nested_files.append(
                    (
                        destination,
                        f"{chain}::{name}",
                    )
                )

    found = []

    for (
        nested_path,
        nested_chain,
    ) in nested_files:
        nested_work = (
            work
            / hashlib.sha256(
                nested_chain.encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
        )

        nested_work.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            (
                resolved_path,
                resolved_chain,
                mode,
            ) = resolve_root(
                nested_path,
                nested_work,
                frozen_member,
                frozen_basename,
                nested_chain,
                depth + 1,
            )

            found.append(
                (
                    resolved_path,
                    resolved_chain,
                    f"nested_{mode}",
                )
            )

        except RuntimeError:
            continue

    frozen_matches = [
        item
        for item in found
        if (
            frozen_basename
            and Path(
                item[1].split("::")[-1]
            ).name
            == frozen_basename
        )
    ]

    if len(frozen_matches) == 1:
        return frozen_matches[0]

    if len(found) == 1:
        return found[0]

    raise RuntimeError(
        (
            "ROOT resolution ambiguous for "
            f"{chain}: "
            f"direct={len(roots)}, "
            f"nested={len(found)}"
        )
    )


def stage_source(
    row: pd.Series,
    work: Path,
) -> tuple[Path, str, str]:
    access_mode = clean(
        row["access_mode"]
    )

    if access_mode == "local_direct_root":
        path = Path(
            clean(
                row["root_locator"]
            )
            or clean(
                row["source_locator"]
            )
        )

        require(
            path.is_file(),
            f"missing direct ROOT: {path}",
        )

        expected_size = int(
            row["source_size_bytes"]
        )

        require(
            (
                expected_size <= 0
                or path.stat().st_size
                == expected_size
            ),
            (
                "direct ROOT size mismatch "
                f"for {row['source_uid']}"
            ),
        )

        checksum_kind = clean(
            row["source_checksum_kind"]
        )

        expected_checksum = (
            normalize_checksum(
                row["source_checksum"]
            )
        )

        if (
            checksum_kind
            and expected_checksum
        ):
            _, observed = file_digest(
                path,
                checksum_kind,
            )

            require(
                (
                    normalize_checksum(
                        observed
                    )
                    == expected_checksum
                ),
                (
                    "direct ROOT checksum "
                    f"mismatch for "
                    f"{row['source_uid']}"
                ),
            )

        return (
            path,
            "direct_root",
            str(path),
        )

    require(
        (
            access_mode
            == "remote_bundle_"
            "extraction_required"
        ),
        (
            "unsupported access mode: "
            f"{access_mode}"
        ),
    )

    uri = clean(
        row["archive_locator"]
    )

    require(
        bool(uri),
        (
            "empty archive locator for "
            f"{row['source_uid']}"
        ),
    )

    outer = (
        work
        / (
            Path(
                uri.split(
                    "?",
                    1,
                )[0]
            ).name
            or "bundle.tar"
        )
    )

    copy_remote(
        uri,
        outer,
    )

    expected_size = int(
        row["source_size_bytes"]
    )

    require(
        (
            expected_size <= 0
            or outer.stat().st_size
            == expected_size
        ),
        (
            "bundle size mismatch for "
            f"{row['source_uid']}"
        ),
    )

    checksum_kind = clean(
        row["source_checksum_kind"]
    )

    expected_checksum = (
        normalize_checksum(
            row["source_checksum"]
        )
    )

    if (
        checksum_kind
        and expected_checksum
    ):
        _, observed = file_digest(
            outer,
            checksum_kind,
        )

        require(
            (
                normalize_checksum(
                    observed
                )
                == expected_checksum
            ),
            (
                "bundle checksum mismatch "
                f"for {row['source_uid']}"
            ),
        )

    frozen_member = (
        member_name(
            row["root_archive_member"]
        )
        if clean(
            row["root_archive_member"]
        )
        else ""
    )

    frozen_basename = Path(
        clean(
            row["root_basename"]
        )
    ).name

    return resolve_root(
        outer,
        work,
        frozen_member,
        frozen_basename,
        outer.name,
    )


def resolve_branch(
    branches: set[str],
    aliases: list[str],
) -> str:
    for alias in aliases:
        if alias in branches:
            return alias

    for existing in sorted(branches):
        for alias in aliases:
            if existing.endswith(
                "/" + alias
            ):
                return existing

    return ""


def scalarize_weight(
    array,
    length: int,
) -> np.ndarray:
    try:
        values = np.asarray(
            ak.to_numpy(array),
            dtype=np.float64,
        )

        if (
            values.ndim == 1
            and len(values) == length
        ):
            return values

    except Exception:
        pass

    values = ak.to_numpy(
        ak.fill_none(
            ak.firsts(array),
            np.nan,
        )
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    require(
        len(values) == length,
        "raw event-weight length mismatch",
    )

    return values


def delta_phi(
    phi_a: float,
    phi_b: float,
) -> float:
    return math.atan2(
        math.sin(phi_a - phi_b),
        math.cos(phi_a - phi_b),
    )


def exact_four_to_jets(
    cost: np.ndarray,
) -> list[int] | None:
    require(
        cost.shape[0] == 4,
        (
            "truth assignment expects "
            "exactly four b quarks"
        ),
    )

    n_jets = cost.shape[1]

    if n_jets < 4:
        return None

    dp: dict[
        int,
        tuple[
            float,
            list[int],
        ],
    ] = {
        0: (
            0.0,
            [-1, -1, -1, -1],
        )
    }

    for jet_index in range(n_jets):
        updated = dict(dp)

        for (
            mask,
            (
                current_cost,
                assignment,
            ),
        ) in dp.items():
            for b_index in range(4):
                if mask & (
                    1 << b_index
                ):
                    continue

                next_mask = (
                    mask
                    | (1 << b_index)
                )

                next_cost = (
                    current_cost
                    + float(
                        cost[
                            b_index,
                            jet_index,
                        ]
                    )
                )

                previous = updated.get(
                    next_mask
                )

                if (
                    previous is None
                    or next_cost
                    < previous[0]
                ):
                    next_assignment = (
                        assignment.copy()
                    )

                    next_assignment[
                        b_index
                    ] = jet_index

                    updated[
                        next_mask
                    ] = (
                        next_cost,
                        next_assignment,
                    )

        dp = updated

    result = dp.get(15)

    return (
        None
        if result is None
        else result[1]
    )


def truth_labels_for_event(
    pid: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    particle_eta: np.ndarray,
    particle_phi: np.ndarray,
    jet_eta: list[float],
    jet_phi: list[float],
) -> tuple[
    list[int],
    bool,
    int,
    int,
]:
    labels = [
        -1
    ] * len(jet_eta)

    direct_b: list[
        tuple[int, int]
    ] = []

    for (
        particle_index,
        particle_pid,
    ) in enumerate(pid):
        if abs(
            int(particle_pid)
        ) != 5:
            continue

        mother_candidates = []

        for mother_value in (
            m1[particle_index],
            m2[particle_index],
        ):
            mother_index = int(
                mother_value
            )

            if (
                0
                <= mother_index
                < len(pid)
                and abs(
                    int(
                        pid[mother_index]
                    )
                )
                == 25
            ):
                mother_candidates.append(
                    mother_index
                )

        if mother_candidates:
            direct_b.append(
                (
                    particle_index,
                    min(
                        mother_candidates
                    ),
                )
            )

    unique_higgs_mothers = sorted({
        mother
        for _, mother
        in direct_b
    })

    if (
        len(direct_b) != 4
        or len(
            unique_higgs_mothers
        )
        != 2
        or any(
            sum(
                mother == target
                for _, mother
                in direct_b
            )
            != 2
            for target
            in unique_higgs_mothers
        )
        or len(jet_eta) < 4
    ):
        return (
            labels,
            False,
            len(direct_b),
            len(
                unique_higgs_mothers
            ),
        )

    cost = np.empty(
        (
            4,
            len(jet_eta),
        ),
        dtype=np.float64,
    )

    for (
        b_slot,
        (
            particle_index,
            _,
        ),
    ) in enumerate(direct_b):
        for jet_index in range(
            len(jet_eta)
        ):
            deta = (
                float(
                    particle_eta[
                        particle_index
                    ]
                )
                - float(
                    jet_eta[
                        jet_index
                    ]
                )
            )

            dphi = delta_phi(
                float(
                    particle_phi[
                        particle_index
                    ]
                ),
                float(
                    jet_phi[
                        jet_index
                    ]
                ),
            )

            cost[
                b_slot,
                jet_index,
            ] = math.sqrt(
                deta * deta
                + dphi * dphi
            )

    assignment = exact_four_to_jets(
        cost
    )

    if assignment is None:
        return (
            labels,
            False,
            len(direct_b),
            len(
                unique_higgs_mothers
            ),
        )

    if any(
        cost[
            b_slot,
            jet_index,
        ]
        >= TRUTH_MATCH_DR_MAX
        for (
            b_slot,
            jet_index,
        ) in enumerate(
            assignment
        )
    ):
        return (
            labels,
            False,
            len(direct_b),
            len(
                unique_higgs_mothers
            ),
        )

    mother_to_slot = {
        mother_index: slot
        for (
            slot,
            mother_index,
        ) in enumerate(
            unique_higgs_mothers
        )
    }

    for (
        b_slot,
        jet_index,
    ) in enumerate(assignment):
        labels[
            jet_index
        ] = mother_to_slot[
            direct_b[b_slot][1]
        ]

    matchable = (
        labels.count(0) == 2
        and labels.count(1) == 2
    )

    return (
        labels,
        matchable,
        len(direct_b),
        len(
            unique_higgs_mothers
        ),
    )


def event_uid(
    source_uid: str,
    source_entry: int,
) -> str:
    return hashlib.sha256(
        (
            f"{source_uid}::entry::"
            f"{source_entry}"
        ).encode("utf-8")
    ).hexdigest()


def build_table(
    row: pd.Series,
    root_path: Path,
) -> tuple[
    pa.Table,
    dict,
]:
    source_uid = clean(
        row["source_uid"]
    )

    is_signal = (
        clean(
            row["sample_class"]
        )
        == "signal"
    )

    auxiliary_qcd = (
        clean(
            row[
                "workflow_population"
            ]
        )
        == (
            "auxiliary_qcd_"
            "classification_only"
        )
    )

    physical_eval = truthy(
        row[
            "physical_evaluation_eligible"
        ]
    )

    with uproot.open(
        root_path
    ) as root_file:
        tree_keys = [
            key
            for (
                key,
                classname,
            ) in root_file.classnames().items()
            if "TTree" in str(classname)
        ]

        require(
            bool(tree_keys),
            f"no TTree for {source_uid}",
        )

        tree_key = next(
            (
                key
                for key
                in tree_keys
                if str(key).split(";")[0]
                == "Delphes"
            ),
            tree_keys[0],
        )

        tree = root_file[
            tree_key
        ]

        expected_entries = int(
            row["generated_events"]
        )

        require(
            (
                int(
                    tree.num_entries
                )
                == expected_entries
            ),
            (
                "entry mismatch for "
                f"{source_uid}: "
                f"{tree.num_entries} "
                f"!= {expected_entries}"
            ),
        )

        branches = {
            str(
                existing
            ).split(";")[0]
            for existing
            in tree.keys(
                recursive=True
            )
        }

        jet_branches = {
            "pt": resolve_branch(
                branches,
                [
                    "Jet.PT",
                    "Jet.Pt",
                ],
            ),
            "eta": resolve_branch(
                branches,
                ["Jet.Eta"],
            ),
            "phi": resolve_branch(
                branches,
                ["Jet.Phi"],
            ),
            "mass": resolve_branch(
                branches,
                ["Jet.Mass"],
            ),
            "btag": resolve_branch(
                branches,
                [
                    "Jet.BTag",
                    "Jet.BTagPhys",
                ],
            ),
        }

        require(
            all(
                jet_branches.values()
            ),
            (
                "missing jet branches: "
                f"{jet_branches}"
            ),
        )

        truth_branches = {
            "pid": resolve_branch(
                branches,
                ["Particle.PID"],
            ),
            "status": resolve_branch(
                branches,
                ["Particle.Status"],
            ),
            "m1": resolve_branch(
                branches,
                ["Particle.M1"],
            ),
            "m2": resolve_branch(
                branches,
                ["Particle.M2"],
            ),
            "pt": resolve_branch(
                branches,
                [
                    "Particle.PT",
                    "Particle.Pt",
                ],
            ),
            "eta": resolve_branch(
                branches,
                ["Particle.Eta"],
            ),
            "phi": resolve_branch(
                branches,
                ["Particle.Phi"],
            ),
            "mass": resolve_branch(
                branches,
                ["Particle.Mass"],
            ),
        }

        if is_signal:
            require(
                all(
                    truth_branches.values()
                ),
                (
                    "missing signal truth "
                    f"branches: "
                    f"{truth_branches}"
                ),
            )

        weight_branch = resolve_branch(
            branches,
            [
                "Event.Weight",
                "Weight.Weight",
                "LHEFEvent.Weight",
                "HepMCEvent.Weight",
            ],
        )

        rows_to_read = min(
            MAX_EVENTS_PER_SOURCE,
            expected_entries,
        )

        expressions = list(
            jet_branches.values()
        )

        if is_signal:
            expressions.extend(
                truth_branches.values()
            )

        if weight_branch:
            expressions.append(
                weight_branch
            )

        expressions = list(
            dict.fromkeys(
                expressions
            )
        )

        arrays = tree.arrays(
            expressions=expressions,
            entry_start=0,
            entry_stop=rows_to_read,
            library="ak",
            how=dict,
        )

    selected = (
        (
            arrays[
                jet_branches["pt"]
            ]
            > JET_PT_MIN_GEV
        )
        & (
            abs(
                arrays[
                    jet_branches[
                        "eta"
                    ]
                ]
            )
            < JET_ABS_ETA_MAX
        )
    )

    jet_pt = arrays[
        jet_branches["pt"]
    ][selected]

    jet_eta = arrays[
        jet_branches["eta"]
    ][selected]

    jet_phi = arrays[
        jet_branches["phi"]
    ][selected]

    jet_mass = arrays[
        jet_branches["mass"]
    ][selected]

    jet_btag = arrays[
        jet_branches["btag"]
    ][selected]

    order = ak.argsort(
        jet_pt,
        axis=1,
        ascending=False,
    )

    jet_pt = jet_pt[order]
    jet_eta = jet_eta[order]
    jet_phi = jet_phi[order]
    jet_mass = jet_mass[order]
    jet_btag = jet_btag[order]

    pt_lists = [
        [
            float(value)
            for value in values
        ]
        for values
        in ak.to_list(jet_pt)
    ]

    eta_lists = [
        [
            float(value)
            for value in values
        ]
        for values
        in ak.to_list(jet_eta)
    ]

    phi_lists = [
        [
            float(value)
            for value in values
        ]
        for values
        in ak.to_list(jet_phi)
    ]

    mass_lists = [
        [
            float(value)
            for value in values
        ]
        for values
        in ak.to_list(jet_mass)
    ]

    btag_lists = [
        [
            float(value)
            for value in values
        ]
        for values
        in ak.to_list(jet_btag)
    ]

    n_selected_jets = np.asarray(
        [
            len(values)
            for values
            in pt_lists
        ],
        dtype=np.int32,
    )

    n_selected_bjets = np.asarray(
        [
            sum(
                value > 0
                for value in values
            )
            for values
            in btag_lists
        ],
        dtype=np.int32,
    )

    broad_eligible = (
        n_selected_jets >= 4
    )

    mask_lists = [
        [True] * len(values)
        for values in pt_lists
    ]

    if weight_branch:
        raw_weights = scalarize_weight(
            arrays[weight_branch],
            rows_to_read,
        )

        raw_weight_available = (
            np.ones(
                rows_to_read,
                dtype=np.bool_,
            )
        )

    else:
        raw_weights = np.full(
            rows_to_read,
            np.nan,
            dtype=np.float64,
        )

        raw_weight_available = (
            np.zeros(
                rows_to_read,
                dtype=np.bool_,
            )
        )

    truth_label_lists: list[
        list[int]
    ] = []

    assignment_matchable = (
        np.zeros(
            rows_to_read,
            dtype=np.bool_,
        )
    )

    direct_b_counts = np.zeros(
        rows_to_read,
        dtype=np.int16,
    )

    higgs_mother_counts = (
        np.zeros(
            rows_to_read,
            dtype=np.int16,
        )
    )

    if is_signal:
        pid_events = ak.to_list(
            arrays[
                truth_branches["pid"]
            ]
        )

        m1_events = ak.to_list(
            arrays[
                truth_branches["m1"]
            ]
        )

        m2_events = ak.to_list(
            arrays[
                truth_branches["m2"]
            ]
        )

        particle_eta_events = (
            ak.to_list(
                arrays[
                    truth_branches[
                        "eta"
                    ]
                ]
            )
        )

        particle_phi_events = (
            ak.to_list(
                arrays[
                    truth_branches[
                        "phi"
                    ]
                ]
            )
        )

        for index in range(
            rows_to_read
        ):
            (
                labels,
                matchable,
                direct_count,
                mother_count,
            ) = truth_labels_for_event(
                np.asarray(
                    pid_events[index]
                ),
                np.asarray(
                    m1_events[index]
                ),
                np.asarray(
                    m2_events[index]
                ),
                np.asarray(
                    particle_eta_events[
                        index
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    particle_phi_events[
                        index
                    ],
                    dtype=np.float64,
                ),
                eta_lists[index],
                phi_lists[index],
            )

            truth_label_lists.append(
                labels
            )

            assignment_matchable[
                index
            ] = matchable

            direct_b_counts[
                index
            ] = direct_count

            higgs_mother_counts[
                index
            ] = mother_count

    else:
        truth_label_lists = [
            [-1] * len(values)
            for values in pt_lists
        ]

    source_entries = np.arange(
        rows_to_read,
        dtype=np.int64,
    )

    event_uids = [
        event_uid(
            source_uid,
            int(entry),
        )
        for entry
        in source_entries
    ]

    data = {
        "source_uid": pa.array(
            [source_uid]
            * rows_to_read,
            type=pa.string(),
        ),
        "group_id": pa.array(
            [
                clean(
                    row["group_id"]
                )
            ]
            * rows_to_read,
            type=pa.string(),
        ),
        "source_entry": pa.array(
            source_entries,
            type=pa.int64(),
        ),
        "event_uid": pa.array(
            event_uids,
            type=pa.string(),
        ),
        "sample_class": pa.array(
            [
                clean(
                    row["sample_class"]
                )
            ]
            * rows_to_read,
            type=pa.string(),
        ),
        "process_or_mode": pa.array(
            [
                clean(
                    row[
                        "process_or_mode"
                    ]
                )
            ]
            * rows_to_read,
            type=pa.string(),
        ),
        "workflow_population": (
            pa.array(
                [
                    clean(
                        row[
                            "workflow_population"
                        ]
                    )
                ]
                * rows_to_read,
                type=pa.string(),
            )
        ),
        "class_label": pa.array(
            np.full(
                rows_to_read,
                (
                    1
                    if is_signal
                    else 0
                ),
                dtype=np.int8,
            ),
            type=pa.int8(),
        ),
        "auxiliary_qcd": pa.array(
            np.full(
                rows_to_read,
                auxiliary_qcd,
                dtype=np.bool_,
            ),
            type=pa.bool_(),
        ),
        "physical_evaluation_eligible": (
            pa.array(
                np.full(
                    rows_to_read,
                    physical_eval,
                    dtype=np.bool_,
                ),
                type=pa.bool_(),
            )
        ),
        "raw_event_weight_available": (
            pa.array(
                raw_weight_available,
                type=pa.bool_(),
            )
        ),
        "raw_event_weight": pa.array(
            raw_weights,
            type=pa.float64(),
            from_pandas=True,
        ),
        "n_selected_jets": pa.array(
            n_selected_jets,
            type=pa.int32(),
        ),
        "n_selected_bjets": pa.array(
            n_selected_bjets,
            type=pa.int32(),
        ),
        "broad_event_eligible": (
            pa.array(
                broad_eligible,
                type=pa.bool_(),
            )
        ),
        "jet_pt": pa.array(
            pt_lists,
            type=pa.list_(
                pa.float32()
            ),
        ),
        "jet_eta": pa.array(
            eta_lists,
            type=pa.list_(
                pa.float32()
            ),
        ),
        "jet_phi": pa.array(
            phi_lists,
            type=pa.list_(
                pa.float32()
            ),
        ),
        "jet_mass": pa.array(
            mass_lists,
            type=pa.list_(
                pa.float32()
            ),
        ),
        "jet_btag": pa.array(
            btag_lists,
            type=pa.list_(
                pa.float32()
            ),
        ),
        "jet_mask": pa.array(
            mask_lists,
            type=pa.list_(
                pa.bool_()
            ),
        ),
        "jet_truth_higgs_slot": (
            pa.array(
                truth_label_lists,
                type=pa.list_(
                    pa.int8()
                ),
            )
        ),
        "assignment_matchable": (
            pa.array(
                assignment_matchable,
                type=pa.bool_(),
            )
        ),
        "truth_direct_higgs_b_count": (
            pa.array(
                direct_b_counts,
                type=pa.int16(),
            )
        ),
        "truth_higgs_mother_count": (
            pa.array(
                higgs_mother_counts,
                type=pa.int16(),
            )
        ),
    }

    table = pa.Table.from_pydict(
        data
    )

    table = table.select(
        OUTPUT_COLUMNS
    )

    metadata = {
        "source_uid": source_uid,
        "source_entries_total": (
            expected_entries
        ),
        "rows_materialized": (
            rows_to_read
        ),
        "weight_branch": (
            weight_branch
        ),
        "raw_weight_available_rows": int(
            raw_weight_available.sum()
        ),
        "broad_eligible_rows": int(
            broad_eligible.sum()
        ),
        "assignment_matchable_rows": int(
            assignment_matchable.sum()
        ),
        "max_selected_jets": (
            int(
                n_selected_jets.max()
            )
            if rows_to_read
            else 0
        ),
        "jet_order": (
            "selected jets sorted by "
            "descending Jet.PT"
        ),
        "jet_storage": (
            "all selected jets; no "
            "extraction-time truncation"
        ),
        "truth_match_rule": (
            "exactly four direct b "
            "daughters of exactly two "
            "Higgs mothers; minimum-cost "
            "one-to-one jet matching with "
            "deltaR < 0.4"
        ),
    }

    return table, metadata


def write_table(
    table: pa.Table,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )


def validate_readback(
    path: Path,
    row: pd.Series,
    expected_rows: int,
    require_signal_matchable: bool,
) -> dict:
    table = pq.read_table(path)

    require(
        table.column_names
        == OUTPUT_COLUMNS,
        (
            "Parquet column order/"
            "schema changed"
        ),
    )

    require(
        table.num_rows
        == expected_rows,
        (
            "Parquet row mismatch: "
            f"{table.num_rows} "
            f"!= {expected_rows}"
        ),
    )

    data = table.to_pydict()

    source_uid = clean(
        row["source_uid"]
    )

    require(
        set(
            data["source_uid"]
        )
        == {source_uid},
        "source_uid readback mismatch",
    )

    require(
        data["source_entry"]
        == list(
            range(expected_rows)
        ),
        (
            "source_entry is not "
            "contiguous"
        ),
    )

    require(
        len(
            set(
                data["event_uid"]
            )
        )
        == expected_rows,
        "event_uid is not unique",
    )

    for (
        entry,
        uid_value,
    ) in zip(
        data["source_entry"],
        data["event_uid"],
    ):
        require(
            uid_value
            == event_uid(
                source_uid,
                int(entry),
            ),
            "event_uid contract mismatch",
        )

    matchable_count = 0

    for index in range(
        expected_rows
    ):
        njet = int(
            data[
                "n_selected_jets"
            ][index]
        )

        nbjet = int(
            data[
                "n_selected_bjets"
            ][index]
        )

        lists = [
            data["jet_pt"][index],
            data["jet_eta"][index],
            data["jet_phi"][index],
            data["jet_mass"][index],
            data["jet_btag"][index],
            data["jet_mask"][index],
            data[
                "jet_truth_higgs_slot"
            ][index],
        ]

        require(
            all(
                len(values) == njet
                for values in lists
            ),
            (
                "per-jet list length "
                "mismatch"
            ),
        )

        require(
            all(
                math.isfinite(
                    float(value)
                )
                for column_name in (
                    "jet_pt",
                    "jet_eta",
                    "jet_phi",
                    "jet_mass",
                    "jet_btag",
                )
                for value in data[
                    column_name
                ][index]
            ),
            (
                "non-finite per-jet "
                "value found"
            ),
        )

        require(
            all(
                data[
                    "jet_mask"
                ][index]
            ),
            (
                "jet_mask contains false "
                "for unpadded storage"
            ),
        )

        require(
            all(
                data["jet_pt"][index][
                    position
                ]
                >= data["jet_pt"][index][
                    position + 1
                ]
                for position in range(
                    max(
                        0,
                        njet - 1,
                    )
                )
            ),
            (
                "jets are not sorted by "
                "descending pT"
            ),
        )

        require(
            nbjet
            == sum(
                value > 0
                for value in data[
                    "jet_btag"
                ][index]
            ),
            "n_selected_bjets mismatch",
        )

        require(
            (
                bool(
                    data[
                        "broad_event_eligible"
                    ][index]
                )
                == (njet >= 4)
            ),
            (
                "broad_event_eligible "
                "mismatch"
            ),
        )

        labels = data[
            "jet_truth_higgs_slot"
        ][index]

        is_signal = (
            clean(
                row[
                    "sample_class"
                ]
            )
            == "signal"
        )

        if not is_signal:
            require(
                set(labels).issubset({
                    -1
                }),
                (
                    "background row contains "
                    "signal assignment label"
                ),
            )

            require(
                not data[
                    "assignment_matchable"
                ][index],
                (
                    "background marked "
                    "assignment-matchable"
                ),
            )

        elif data[
            "assignment_matchable"
        ][index]:
            matchable_count += 1

            require(
                (
                    labels.count(0) == 2
                    and labels.count(1)
                    == 2
                ),
                (
                    "matchable signal lacks "
                    "2+2 labels"
                ),
            )

            require(
                njet >= 4,
                (
                    "matchable signal is "
                    "not broad-event eligible"
                ),
            )

    if require_signal_matchable:
        require(
            matchable_count > 0,
            (
                "no assignment-matchable "
                f"events in {source_uid}"
            ),
        )

    weight_available = np.asarray(
        data[
            "raw_event_weight_available"
        ],
        dtype=bool,
    )

    weight_values = np.asarray(
        [
            (
                np.nan
                if value is None
                else float(value)
            )
            for value in data[
                "raw_event_weight"
            ]
        ],
        dtype=np.float64,
    )

    if weight_available.any():
        require(
            np.isfinite(
                weight_values[
                    weight_available
                ]
            ).all(),
            (
                "non-finite available "
                "raw weights"
            ),
        )

    return {
        "rows": expected_rows,
        "event_uid_unique": (
            expected_rows
        ),
        "assignment_matchable_rows": (
            matchable_count
        ),
        "broad_eligible_rows": int(
            sum(
                bool(value)
                for value in data[
                    "broad_event_eligible"
                ]
            )
        ),
        "raw_weight_available_rows": int(
            weight_available.sum()
        ),
        "max_selected_jets": (
            max(
                data[
                    "n_selected_jets"
                ]
            )
            if expected_rows
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--current-head",
        required=True,
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest,
        sep="\t",
    )

    plan = pd.read_csv(
        args.plan,
        sep="\t",
    )

    require(
        len(manifest) == 464,
        (
            "manifest rows changed: "
            f"{len(manifest)}"
        ),
    )

    require(
        (
            manifest["source_uid"]
            .astype(str)
            .nunique()
            == 464
        ),
        (
            "manifest source_uid "
            "not unique"
        ),
    )

    require(
        (
            int(
                pd.to_numeric(
                    manifest[
                        "generated_events"
                    ]
                ).sum()
            )
            == 3_799_873
        ),
        "manifest event total changed",
    )

    require(
        (
            manifest["split"]
            .astype(str)
            .eq("train")
            .all()
        ),
        "non-train row found",
    )

    require(
        (
            manifest[
                "source_locator_assigned"
            ]
            .map(truthy)
            .all()
        ),
        "unassigned source found",
    )

    require(
        len(plan) == 7,
        (
            "plan rows changed: "
            f"{len(plan)}"
        ),
    )

    require(
        (
            plan["source_uid"]
            .astype(str)
            .nunique()
            == 7
        ),
        "plan source_uid not unique",
    )

    manifest_by_uid = (
        manifest.set_index(
            "source_uid",
            drop=False,
        )
    )

    plan_uids = [
        clean(value)
        for value in plan[
            "source_uid"
        ]
    ]

    require(
        all(
            uid
            in manifest_by_uid.index
            for uid in plan_uids
        ),
        (
            "plan source absent "
            "from manifest"
        ),
    )

    output_a = (
        args.output_dir
        / "run_a"
    )

    output_b = (
        args.output_dir
        / "run_b"
    )

    output_a.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_b.mkdir(
        parents=True,
        exist_ok=False,
    )

    schema_contract = {
        "schema_version": 1,
        "columns": OUTPUT_COLUMNS,
        "max_events_per_source": (
            MAX_EVENTS_PER_SOURCE
        ),
        "selected_jet": (
            f"Jet.PT > "
            f"{JET_PT_MIN_GEV} GeV "
            f"and abs(Jet.Eta) < "
            f"{JET_ABS_ETA_MAX}"
        ),
        "jet_order": (
            "descending Jet.PT"
        ),
        "jet_storage": (
            "all selected jets; "
            "variable-length lists; "
            "no extraction-time truncation"
        ),
        "broad_event_eligibility": (
            "n_selected_jets >= 4"
        ),
        "pre_model_btag_requirement": (
            "none"
        ),
        "pre_model_higgs_mass_requirement": (
            "none"
        ),
        "event_uid": (
            "sha256(source_uid::entry::"
            "<source_entry>)"
        ),
        "truth_assignment": (
            "signal only; exactly four "
            "direct b daughters of exactly "
            "two Higgs mothers; minimum-cost "
            "one-to-one jet matching with "
            "deltaR < 0.4"
        ),
        "raw_event_weight": (
            "raw ROOT event weight when a "
            "recognized branch exists; no "
            "physical normalization or "
            "sidecar transport join is "
            "performed in this canary"
        ),
        "physical_weight_application_authorized": (
            False
        ),
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }

    (
        args.output_dir
        / "feature_schema_v1.json"
    ).write_text(
        json.dumps(
            schema_contract,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipts = []

    total_rows = 0
    total_matchable = 0
    total_broad = 0

    for plan_row in plan.itertuples(
        index=False
    ):
        canary_class = clean(
            plan_row.canary_class
        )

        source_uid = clean(
            plan_row.source_uid
        )

        row = manifest_by_uid.loc[
            source_uid
        ]

        source_work = (
            args.work_dir
            / hashlib.sha256(
                source_uid.encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
        )

        shutil.rmtree(
            source_work,
            ignore_errors=True,
        )

        source_work.mkdir(
            parents=True,
            exist_ok=True,
        )

        status = "fail"
        error = ""

        receipt: dict = {
            "canary_class": (
                canary_class
            ),
            "source_uid": (
                source_uid
            ),
            "sample_class": clean(
                row["sample_class"]
            ),
            "process_or_mode": clean(
                row[
                    "process_or_mode"
                ]
            ),
        }

        try:
            (
                root_path,
                resolution_mode,
                resolution_chain,
            ) = stage_source(
                row,
                source_work,
            )

            (
                table,
                metadata,
            ) = build_table(
                row,
                root_path,
            )

            safe_name = (
                canary_class.replace(
                    "/",
                    "_",
                )
            )

            path_a = (
                output_a
                / (
                    f"{safe_name}."
                    "parquet"
                )
            )

            path_b = (
                output_b
                / (
                    f"{safe_name}."
                    "parquet"
                )
            )

            write_table(
                table,
                path_a,
            )

            write_table(
                table,
                path_b,
            )

            checksum_a = sha256(
                path_a
            )

            checksum_b = sha256(
                path_b
            )

            require(
                checksum_a
                == checksum_b,
                (
                    "byte-level idempotency "
                    f"failed for "
                    f"{canary_class}"
                ),
            )

            readback = (
                validate_readback(
                    path_a,
                    row,
                    table.num_rows,
                    require_signal_matchable=(
                        clean(
                            row[
                                "sample_class"
                            ]
                        )
                        == "signal"
                    ),
                )
            )

            status = "pass"

            receipt.update({
                "root_resolution_mode": (
                    resolution_mode
                ),
                "root_resolution_chain": (
                    resolution_chain
                ),
                "parquet_a": str(
                    path_a
                ),
                "parquet_b": str(
                    path_b
                ),
                "parquet_sha256": (
                    checksum_a
                ),
                "parquet_byte_identical": (
                    True
                ),
                **metadata,
                **{
                    (
                        f"readback_"
                        f"{key}"
                    ): value
                    for (
                        key,
                        value,
                    ) in readback.items()
                },
            })

            total_rows += int(
                table.num_rows
            )

            total_matchable += int(
                readback[
                    "assignment_matchable_rows"
                ]
            )

            total_broad += int(
                readback[
                    "broad_eligible_rows"
                ]
            )

        except Exception as exc:
            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        receipt["status"] = status
        receipt["error"] = error

        receipts.append(
            receipt
        )

        print(
            "MATERIALIZATION_SOURCE_RESULT="
            f"class={canary_class};"
            f"status={status};"
            f"source_uid={source_uid};"
            f"error={error}"
        )

        shutil.rmtree(
            source_work,
            ignore_errors=True,
        )

    receipt_frame = pd.DataFrame(
        receipts
    )

    receipt_frame.to_csv(
        (
            args.output_dir
            / (
                "seven_class_feature_"
                "materialization_receipts.tsv"
            )
        ),
        sep="\t",
        index=False,
    )

    def numeric_column(
        frame: pd.DataFrame,
        name: str,
        default: float,
    ) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(
                [default] * len(frame),
                index=frame.index,
                dtype=float,
            )

        return pd.to_numeric(
            frame[name],
            errors="coerce",
        ).fillna(default)

    def boolean_column(
        frame: pd.DataFrame,
        name: str,
    ) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(
                [False] * len(frame),
                index=frame.index,
                dtype=bool,
            )

        return (
            frame[name]
            .fillna(False)
            .astype(bool)
        )

    passed = int(
        receipt_frame["status"]
        .astype(str)
        .eq("pass")
        .sum()
    )

    byte_identical = int(
        boolean_column(
            receipt_frame,
            "parquet_byte_identical",
        ).sum()
    )

    signal_receipts = (
        receipt_frame.loc[
            receipt_frame[
                "sample_class"
            ]
            .astype(str)
            .eq("signal")
        ]
    )

    signal_matchable = int(
        (
            numeric_column(
                signal_receipts,
                (
                    "readback_assignment_"
                    "matchable_rows"
                ),
                0,
            )
            > 0
        ).sum()
    )

    event_uid_pass = int(
        (
            numeric_column(
                receipt_frame,
                (
                    "readback_event_uid_"
                    "unique"
                ),
                -1,
            )
            == numeric_column(
                receipt_frame,
                "readback_rows",
                -2,
            )
        ).sum()
    )

    all_pass = (
        passed == 7
        and byte_identical == 7
        and signal_matchable == 2
        and event_uid_pass == 7
        and total_rows
        == (
            7
            * MAX_EVENTS_PER_SOURCE
        )
    )

    summary = {
        "status": (
            "pass"
            if all_pass
            else "needs_review"
        ),
        "current_head": (
            args.current_head
        ),
        "canary_sources": 7,
        "canary_sources_passed": (
            passed
        ),
        "parquet_byte_identical_sources": (
            byte_identical
        ),
        "event_uid_contract_sources_passed": (
            event_uid_pass
        ),
        "signal_sources_with_matchable_assignments": (
            signal_matchable
        ),
        "materialized_rows": (
            total_rows
        ),
        "broad_eligible_rows": (
            total_broad
        ),
        "assignment_matchable_rows": (
            total_matchable
        ),
        "output_schema": str(
            args.output_dir
            / "feature_schema_v1.json"
        ),
        "raw_generator_weight_transport_joined": (
            False
        ),
        "physical_weights_calculated": 0,
        "physical_yields_calculated": 0,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "models_trained": 0,
        "repository_modified": False,
        "result": (
            "SEVEN_CLASS_FEATURE_"
            "MATERIALIZATION_CANARY_"
            "V1_PASS"
            if all_pass
            else
            "SEVEN_CLASS_FEATURE_"
            "MATERIALIZATION_CANARY_"
            "V1_NEEDS_REVIEW"
        ),
        "next": (
            "RUN_TRAIN_GENERATOR_WEIGHT_"
            "JOIN_CANARY_AND_FREEZE_"
            "EXTRACTOR"
            if all_pass
            else
            "REVIEW_FAILED_FEATURE_"
            "MATERIALIZATION_RECEIPTS"
        ),
    }

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "SEVEN_CLASS_FEATURE_SOURCES=7/7"
    )

    print(
        "SEVEN_CLASS_FEATURE_SOURCES_PASSED="
        f"{passed}/7"
    )

    print(
        "PARQUET_BYTE_IDENTICAL_SOURCES="
        f"{byte_identical}/7"
    )

    print(
        "EVENT_UID_CONTRACT_SOURCES_PASSED="
        f"{event_uid_pass}/7"
    )

    print(
        "SIGNAL_SOURCES_WITH_MATCHABLE_ASSIGNMENTS="
        f"{signal_matchable}/2"
    )

    print(
        "MATERIALIZED_ROWS="
        f"{total_rows}/"
        f"{7 * MAX_EVENTS_PER_SOURCE}"
    )

    print(
        f"BROAD_ELIGIBLE_ROWS="
        f"{total_broad}"
    )

    print(
        "ASSIGNMENT_MATCHABLE_ROWS="
        f"{total_matchable}"
    )

    print(
        "RAW_GENERATOR_WEIGHT_TRANSPORT_JOINED=NO"
    )

    print(
        "PHYSICAL_WEIGHTS_CALCULATED=0"
    )

    print(
        "PHYSICAL_YIELDS_CALCULATED=0"
    )

    print(
        "VALIDATION_PAYLOADS_OPENED=0"
    )

    print(
        "TEST_PAYLOADS_OPENED=0"
    )

    print(
        "MODELS_TRAINED=0"
    )

    print(
        "REPOSITORY_MODIFIED=NO"
    )

    print(
        f"RESULT={summary['result']}"
    )

    print(
        f"NEXT={summary['next']}"
    )

    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
