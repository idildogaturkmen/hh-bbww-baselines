#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_OLD_WORKER_SHA256 = (
    "cf5cf37500bc24e46dda4c7d484d1a3c"
    "20f3b2f01fc2cf0ea12a0980dd3ae40e"
)

EXPECTED_CONVERTER_SHA256 = (
    "fdfbd860e5c8e1097d1085e0d669d8e9"
    "ea6e25bd747098b95ad4e23cc70e9000"
)

EXPECTED_DELPHES_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

EXPECTED_EVENT_SUMMARY_SHA256 = (
    "a17ec47b553558e42d13066cda7719bae"
    "130eb63fe4fc95b90e1e17756956b6e"
)

EXPECTED_CANDIDATE_SHA256 = (
    "4d7eb8e3400ec50c1c254744e6f3a2b"
    "11bd033a4525c13ec22e85247ecf59c57"
)

EXPECTED_PARQUET_WRITER_SHA256 = (
    "abef7e4f5d1b82fe72837834b0b0794b"
    "59bb31ff16032b1b5a7f1519a6edbcfb"
)

FAMILIES = (
    "wwz_zbb",
    "wzz_zbb",
    "zzz_zbb",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    require(
        count == 1,
        (
            f"{label}: expected exactly one match, "
            f"found {count}"
        ),
    )

    return text.replace(
        old,
        new,
        1,
    )


def regex_once(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )

    require(
        count == 1,
        (
            f"{label}: expected exactly one regex match, "
            f"found {count}"
        ),
    )

    return updated


def safe_extract(
    archive: tarfile.TarFile,
    destination: Path,
) -> None:
    resolved_destination = destination.resolve()

    for member in archive.getmembers():
        resolved_target = (
            destination
            / member.name
        ).resolve()

        require(
            resolved_target == resolved_destination
            or resolved_destination
            in resolved_target.parents,
            f"unsafe archive member {member.name}",
        )

    archive.extractall(destination)


def update_manifest(
    manifest: Path,
    updates: dict[str, str],
    remove_keys: set[str],
) -> None:
    original_lines = manifest.read_text(
        errors="replace",
    ).splitlines()

    output_lines: list[str] = []
    used: set[str] = set()

    for line in original_lines:
        if "=" not in line:
            output_lines.append(line)
            continue

        key, _ = line.split(
            "=",
            1,
        )

        key = key.strip()

        if key in remove_keys:
            continue

        if key in updates:
            output_lines.append(
                f"{key}={updates[key]}"
            )
            used.add(key)
        else:
            output_lines.append(line)

    for key, value in updates.items():
        if key not in used:
            output_lines.append(
                f"{key}={value}"
            )

    manifest.write_text(
        "\n".join(output_lines)
        + "\n"
    )


def build_worker(
    old_worker: Path,
    new_worker: Path,
) -> str:
    require(
        sha256_file(old_worker)
        == EXPECTED_OLD_WORKER_SHA256,
        "partial worker differs from inspected source",
    )

    text = old_worker.read_text(
        errors="strict",
    )

    text = replace_once(
        text,
        'if [[ "$#" -ne 14 ]]; then',
        'if [[ "$#" -ne 15 ]]; then',
        "argument count",
    )

    text = replace_once(
        text,
        (
            'echo "  $0 CAMPAIGN FAMILY TARGET_TAG SHARD_ID '
            'N_EVENTS SEED DATASET_SPLIT SPLIT_SALT '
            'RUN_CARD_PROFILE DATASET_ROLE INPUT_TARBALL '
            'EOS_DIR EXPECTED_PAYLOAD_SHA256 CLUSTER_ID" >&2'
        ),
        (
            'echo "  $0 CAMPAIGN FAMILY TARGET_TAG SHARD_ID '
            'N_TARGET_ACCEPTED N_INPUT_EVENTS SEED DATASET_SPLIT '
            'SPLIT_SALT RUN_CARD_PROFILE DATASET_ROLE INPUT_TARBALL '
            'EOS_DIR EXPECTED_PAYLOAD_SHA256 CLUSTER_ID" >&2'
        ),
        "usage declaration",
    )

    old_arguments = """N_EVENTS="$5"
SEED="$6"
DATASET_SPLIT="$7"
SPLIT_SALT="$8"
RUN_CARD_PROFILE="$9"
DATASET_ROLE="${10}"
INPUT_TARBALL="${11}"
EOS_DIR="${12}"
EXPECTED_PAYLOAD_SHA256="${13}"
CLUSTER_ID="${14}"
"""

    new_arguments = """N_EVENTS="$5"
N_INPUT_EVENTS="$6"
SEED="$7"
DATASET_SPLIT="$8"
SPLIT_SALT="$9"
RUN_CARD_PROFILE="${10}"
DATASET_ROLE="${11}"
INPUT_TARBALL="${12}"
EOS_DIR="${13}"
EXPECTED_PAYLOAD_SHA256="${14}"
CLUSTER_ID="${15}"
"""

    text = replace_once(
        text,
        old_arguments,
        new_arguments,
        "positional argument block",
    )

    text = replace_once(
        text,
        '[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]]\n',
        (
            '[[ "$N_EVENTS" =~ ^[1-9][0-9]*$ ]]\n'
            '[[ "$N_INPUT_EVENTS" =~ ^[1-9][0-9]*$ ]]\n'
            '(( N_INPUT_EVENTS >= N_EVENTS ))\n'
        ),
        "input-count validation",
    )

    text = replace_once(
        text,
        '  "n_events": $N_EVENTS,\n  "seed": $SEED,',
        (
            '  "n_events": $N_EVENTS,\n'
            '  "input_lhe_events_requested": $N_INPUT_EVENTS,\n'
            '  "seed": $SEED,'
        ),
        "receipt input target",
    )

    text = replace_once(
        text,
        "N_LHE=0\nN_HEPMC=0\n",
        (
            "N_LHE=0\n"
            "N_PROCESSED_INPUT=0\n"
            "N_REJECTED_INPUT=0\n"
            "N_PYTHIA_FAILURES=0\n"
            "MAXIMUM_ZBB_MULTIPLICITY=0\n"
            "FILTER_ACCOUNTING_SHA256=\"\"\n"
            "N_HEPMC=0\n"
        ),
        "filter counters initialization",
    )

    text = replace_once(
        text,
        (
            '  "candidate_sha256": "$CANDIDATE_SHA256",\n'
            '  "bundle_sha256": "$BUNDLE_SHA256",'
        ),
        (
            '  "candidate_sha256": "$CANDIDATE_SHA256",\n'
            '  "filter_accounting_sha256": '
            '"$FILTER_ACCOUNTING_SHA256",\n'
            '  "bundle_sha256": "$BUNDLE_SHA256",'
        ),
        "receipt accounting hash",
    )

    text = replace_once(
        text,
        (
            '  "generator_xsec_pb": $GENERATOR_XSEC_PB,\n'
            '  "lhe_events": $N_LHE,\n'
            '  "hepmc_events": $N_HEPMC,'
        ),
        (
            '  "generator_xsec_pb": $GENERATOR_XSEC_PB,\n'
            '  "generator_xsec_semantics": '
            '"inclusive_before_zbb_filter",\n'
            '  "filter_requirement": '
            '"at_least_one_direct_z_to_bb",\n'
            '  "filter_normalization_status": '
            '"inclusive_xsec_times_separately_frozen_zbb_probability_'
            'not_yet_authorized",\n'
            '  "lhe_events": $N_LHE,\n'
            '  "processed_input_events": $N_PROCESSED_INPUT,\n'
            '  "rejected_input_events": $N_REJECTED_INPUT,\n'
            '  "pythia_failures": $N_PYTHIA_FAILURES,\n'
            '  "accepted_events": $N_HEPMC,\n'
            '  "maximum_zbb_multiplicity": '
            '$MAXIMUM_ZBB_MULTIPLICITY,\n'
            '  "hepmc_events": $N_HEPMC,'
        ),
        "receipt filter accounting",
    )

    text = replace_once(
        text,
        (
            "EXPECTED_CONVERTER_SHA256=\"$(\n"
            "  awk -F= '$1 == \"seeded_converter_sha256\" "
            "{print $2}' \"$MANIFEST\"\n"
            ")\""
        ),
        (
            "EXPECTED_CONVERTER_SHA256=\"$(\n"
            "  awk -F= '$1 == \"exact_target_converter_sha256\" "
            "{print $2}' \"$MANIFEST\"\n"
            ")\""
        ),
        "converter manifest key",
    )

    text = replace_once(
        text,
        (
            'CONVERTER_SOURCE="$REPO/scripts/production/'
            'lhe_to_hepmc3_seeded.cc"'
        ),
        (
            'CONVERTER_SOURCE="$REPO/scripts/production/'
            'lhe_to_hepmc3_triboson_zbb_exact_target.cc"'
        ),
        "converter source path",
    )

    text = replace_once(
        text,
        'CONVERTER="$SCRATCH/lhe_to_hepmc3_seeded"',
        (
            'CONVERTER="$SCRATCH/'
            'lhe_to_hepmc3_triboson_zbb_exact_target"'
        ),
        "converter binary name",
    )

    text = replace_once(
        text,
        (
            '  "$WORK/Cards/run_card.dat" \\\n'
            '  "$N_EVENTS" \\\n'
            '  "$SEED"'
        ),
        (
            '  "$WORK/Cards/run_card.dat" \\\n'
            '  "$N_INPUT_EVENTS" \\\n'
            '  "$SEED"'
        ),
        "MG5 input-event request",
    )

    text = replace_once(
        text,
        (
            'if [[ "$N_LHE" -ne "$N_EVENTS" ]]; then\n'
            '  echo "ERROR: LHE event-count mismatch: '
            '$N_LHE != $N_EVENTS" >&2'
        ),
        (
            'if [[ "$N_LHE" -ne "$N_INPUT_EVENTS" ]]; then\n'
            '  echo "ERROR: LHE event-count mismatch: '
            '$N_LHE != $N_INPUT_EVENTS" >&2'
        ),
        "LHE input-event count",
    )

    old_converter_call = """STAGE="pythia_hepmc"

"$CONVERTER" \\
  "$LHE" \\
  "$HEPMC" \\
  "$N_EVENTS" \\
  "$PYTHIA_SEED" \\
  2>&1 |
tee "$OUT/logs/${TARGET_TAG}_pythia_hepmc.log"

N_HEPMC="$(
"""

    new_converter_call = """STAGE="pythia_hepmc"

CONVERTER_LOG="$OUT/logs/${TARGET_TAG}_pythia_hepmc.log"

"$CONVERTER" \\
  "$LHE" \\
  "$HEPMC" \\
  "$N_INPUT_EVENTS" \\
  "$N_EVENTS" \\
  "$PYTHIA_SEED" \\
  --require-at-least-one-zbb \\
  2>&1 |
tee "$CONVERTER_LOG"

read_converter_value() {
  local key="$1"

  awk -F': ' -v key="$key" '
    $1 == key {
      value = $2
    }
    END {
      print value
    }
  ' "$CONVERTER_LOG"
}

REPORTED_MAX_INPUT="$(
  read_converter_value "Maximum input events"
)"

REPORTED_TARGET_ACCEPTED="$(
  read_converter_value "Target accepted events"
)"

N_PROCESSED_INPUT="$(
  read_converter_value "Processed input events"
)"

N_PYTHIA_FAILURES="$(
  read_converter_value "Pythia failures"
)"

N_REJECTED_INPUT="$(
  read_converter_value "Rejected events"
)"

REPORTED_ACCEPTED="$(
  read_converter_value "Accepted events"
)"

MAXIMUM_ZBB_MULTIPLICITY="$(
  read_converter_value "Maximum Zbb multiplicity"
)"

for value in \\
  "$REPORTED_MAX_INPUT" \\
  "$REPORTED_TARGET_ACCEPTED" \\
  "$N_PROCESSED_INPUT" \\
  "$N_PYTHIA_FAILURES" \\
  "$N_REJECTED_INPUT" \\
  "$REPORTED_ACCEPTED" \\
  "$MAXIMUM_ZBB_MULTIPLICITY"
do
  [[ "$value" =~ ^[0-9]+$ ]]
done

[[ "$REPORTED_MAX_INPUT" -eq "$N_INPUT_EVENTS" ]]
[[ "$REPORTED_TARGET_ACCEPTED" -eq "$N_EVENTS" ]]
[[ "$REPORTED_ACCEPTED" -eq "$N_EVENTS" ]]
[[ "$N_PYTHIA_FAILURES" -eq 0 ]]
[[ "$N_PROCESSED_INPUT" -le "$N_INPUT_EVENTS" ]]
[[ "$N_PROCESSED_INPUT" -eq \\
   $((N_REJECTED_INPUT + REPORTED_ACCEPTED)) ]]

N_HEPMC="$(
"""

    text = replace_once(
        text,
        old_converter_call,
        new_converter_call,
        "exact-target converter invocation",
    )

    accounting_block = r'''
FILTER_ACCOUNTING="$OUT/metadata/${TARGET_TAG}_filter_accounting.json"

python3 - \
  "$FILTER_ACCOUNTING" \
  "$CAMPAIGN" \
  "$FAMILY" \
  "$TARGET_TAG" \
  "$N_INPUT_EVENTS" \
  "$N_LHE" \
  "$N_PROCESSED_INPUT" \
  "$N_REJECTED_INPUT" \
  "$N_PYTHIA_FAILURES" \
  "$N_HEPMC" \
  "$MAXIMUM_ZBB_MULTIPLICITY" \
  "$PYTHIA_SEED" \
  "$GENERATOR_XSEC_PB" <<'PY'
from pathlib import Path
import json
import sys

(
    output_path,
    campaign,
    family,
    target_tag,
    requested_input_events,
    lhe_input_events,
    processed_input_events,
    rejected_input_events,
    pythia_failures,
    accepted_events,
    maximum_zbb_multiplicity,
    pythia_seed,
    generator_xsec_pb,
) = sys.argv[1:]

record = {
    "schema_version": 1,
    "campaign": campaign,
    "family": family,
    "target_tag": target_tag,
    "requested_input_events": int(requested_input_events),
    "lhe_input_events": int(lhe_input_events),
    "processed_input_events": int(processed_input_events),
    "rejected_input_events": int(rejected_input_events),
    "pythia_failures": int(pythia_failures),
    "accepted_events": int(accepted_events),
    "maximum_zbb_multiplicity": int(maximum_zbb_multiplicity),
    "pythia_seed": int(pythia_seed),
    "filter_requirement": "at_least_one_direct_z_to_bb",
    "all_z_forced_to_bb": False,
    "inclusive_pythia_z_decays": True,
    "generator_xsec_pb": float(generator_xsec_pb),
    "generator_xsec_semantics": "inclusive_before_zbb_filter",
    "normalization_status": (
        "inclusive_xsec_times_separately_frozen_zbb_probability_"
        "not_yet_authorized"
    ),
    "physics_yield_authorized": False,
}

Path(output_path).write_text(
    json.dumps(
        record,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

FILTER_ACCOUNTING_SHA256="$(
  sha256sum "$FILTER_ACCOUNTING" |
  awk '{print $1}'
)"

'''

    text = replace_once(
        text,
        (
            'PROVENANCE="$OUT/metadata/'
            '${TARGET_TAG}_provenance.json"\n'
        ),
        (
            accounting_block
            + 'PROVENANCE="$OUT/metadata/'
            '${TARGET_TAG}_provenance.json"\n'
        ),
        "filter-accounting metadata",
    )

    require(
        "--require-at-least-one-zbb"
        in text,
        "patched worker lacks exact Zbb requirement",
    )

    require(
        '"$N_INPUT_EVENTS" \\\n  "$N_EVENTS" \\\n'
        in text,
        "patched converter argument order is wrong",
    )

    require(
        "--force-hbb" not in text,
        "patched worker unexpectedly contains force-Hbb logic",
    )

    require(
        "exact_target_converter_sha256"
        in text,
        "patched worker lacks exact converter manifest key",
    )

    new_worker.write_text(text)
    new_worker.chmod(0o755)

    subprocess.run(
        [
            "bash",
            "-n",
            str(new_worker),
        ],
        check=True,
    )

    return sha256_file(new_worker)


def compile_converter(
    converter: Path,
    binary: Path,
) -> None:
    setup = Path(
        "/cvmfs/sft.cern.ch/lcg/views/"
        "LCG_106/x86_64-el9-gcc13-opt/setup.sh"
    )

    require(
        setup.is_file(),
        f"missing LCG setup {setup}",
    )

    command = (
        f"source {shlex.quote(str(setup))} && "
        f"g++ -O2 -std=c++17 "
        f"{shlex.quote(str(converter))} "
        f"-o {shlex.quote(str(binary))} "
        "$(pythia8-config --cxxflags) "
        "$(HepMC3-config --cxxflags) "
        "$(pythia8-config --libs) "
        "$(HepMC3-config --libs)"
    )

    subprocess.run(
        [
            "/bin/bash",
            "-lc",
            command,
        ],
        check=True,
    )

    require(
        binary.is_file()
        and binary.stat().st_size > 0,
        "converter compilation produced no binary",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--store",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()

    freeze_dir = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_worker_inputs_freeze_20260722_v1"
    )

    freeze_path = (
        freeze_dir
        / "triboson_worker_input_freeze.json"
    )

    old_dir = (
        store
        / "condor_payloads"
        / "final_background_triboson_zbb_payloads_20260722_v1"
    )

    old_worker = (
        old_dir
        / "run_mg5_frozen_v2_exact_zbb_bundle.sh"
    )

    converter = (
        freeze_dir
        / "source"
        / "lhe_to_hepmc3_triboson_zbb_exact_target.cc"
    )

    base_payload = (
        store
        / "condor_payloads"
        / "final_background_hbb_force_payloads_20260722_v1"
        / "ggh_hbb_inputs.tar.gz"
    )

    hbb_pilot_audit = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_pilots10k_audit_20260722_v1"
        / "hbb_pilots10k_final_audit.json"
    )

    new_dir = (
        store
        / "condor_payloads"
        / "final_background_triboson_zbb_payloads_20260722_v2"
    )

    outdir = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_worker_v2_build_20260722_v1"
    )

    for path in (
        new_dir,
        outdir,
    ):
        require(
            not path.exists(),
            f"refusing to overwrite {path}",
        )

    for path in (
        freeze_path,
        old_worker,
        converter,
        base_payload,
        hbb_pilot_audit,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        sha256_file(converter)
        == EXPECTED_CONVERTER_SHA256,
        "validated converter hash mismatch",
    )

    freeze = json.loads(
        freeze_path.read_text()
    )

    require(
        freeze.get("worker_v2_build_inputs_valid")
        is True,
        "worker-v2 input freeze is not valid",
    )

    subprocess_rows = {
        row["technical_subprocess"]: row
        for row in freeze["technical_subprocesses"]
    }

    require(
        set(subprocess_rows) == set(FAMILIES),
        "wrong triboson family set in freeze",
    )

    pilot_audit = json.loads(
        hbb_pilot_audit.read_text()
    )

    ggh_rows = [
        row
        for row in pilot_audit["pilots"]
        if row["family"] == "ggh_hbb"
    ]

    require(
        len(ggh_rows) == 1,
        "could not identify audited ggH base payload",
    )

    require(
        sha256_file(base_payload)
        == ggh_rows[0]["payload_sha256"],
        "base payload differs from audited Hbb payload",
    )

    new_dir.mkdir(parents=True)
    outdir.mkdir(parents=True)

    worker = (
        new_dir
        / "run_mg5_frozen_v2_triboson_zbb_exact_target_bundle.sh"
    )

    worker_sha = build_worker(
        old_worker,
        worker,
    )

    validation_dir = (
        outdir
        / "validation"
    )

    validation_dir.mkdir()

    compiled_converter = (
        validation_dir
        / "lhe_to_hepmc3_triboson_zbb_exact_target"
    )

    compile_converter(
        converter,
        compiled_converter,
    )

    current_head = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    build_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        row = subprocess_rows[family]

        process_dir = Path(
            row["process_directory"]
        )

        require(
            process_dir.is_dir(),
            f"{family}: missing process directory",
        )

        with tempfile.TemporaryDirectory(
            prefix=f"triboson_payload_{family}_",
            dir=str(store),
        ) as temporary:
            temporary_root = Path(temporary)

            with tarfile.open(
                base_payload,
                "r:gz",
            ) as archive:
                safe_extract(
                    archive,
                    temporary_root,
                )

            payload_root = (
                temporary_root
                / "payload"
            )

            require(
                payload_root.is_dir(),
                f"{family}: base payload lacks payload/",
            )

            mg5_template = (
                payload_root
                / "mg5_template"
            )

            shutil.rmtree(
                mg5_template
            )

            shutil.copytree(
                process_dir,
                mg5_template,
                symlinks=False,
            )

            for generated_directory in (
                mg5_template / "Events",
                mg5_template / "HTML",
            ):
                if generated_directory.exists():
                    shutil.rmtree(
                        generated_directory
                    )

                generated_directory.mkdir()

            payload_repo = (
                payload_root
                / "repo"
            )

            payload_converter = (
                payload_repo
                / "scripts/production"
                / "lhe_to_hepmc3_triboson_zbb_exact_target.cc"
            )

            payload_converter.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                converter,
                payload_converter,
            )

            old_seeded_converter = (
                payload_repo
                / "scripts/production"
                / "lhe_to_hepmc3_seeded.cc"
            )

            if old_seeded_converter.exists():
                old_seeded_converter.unlink()

            manifest = (
                payload_root
                / "manifest.txt"
            )

            require(
                manifest.is_file(),
                f"{family}: base manifest missing",
            )

            process_card = (
                mg5_template
                / "Cards/proc_card_mg5.dat"
            )

            run_card = (
                mg5_template
                / "Cards/run_card.dat"
            )

            param_card = (
                mg5_template
                / "Cards/param_card.dat"
            )

            require(
                sha256_file(process_card)
                == row["proc_card_sha256"],
                f"{family}: process-card hash mismatch",
            )

            require(
                sha256_file(run_card)
                == row["run_card_sha256"],
                f"{family}: run-card hash mismatch",
            )

            require(
                sha256_file(param_card)
                == row["param_card_sha256"],
                f"{family}: parameter-card hash mismatch",
            )

            event_script = (
                payload_repo
                / "scripts/delphes"
                / "make_delphes_event_summary.py"
            )

            candidate_script = (
                payload_repo
                / "scripts/delphes"
                / "reconstruct_hh4b_candidates_v2.py"
            )

            parquet_writer = (
                payload_repo
                / "scripts/delphes"
                / "write_parquet_from_pickle.py"
            )

            delphes_card = (
                payload_repo
                / "cards/delphes"
                / "delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
            )

            require(
                sha256_file(event_script)
                == EXPECTED_EVENT_SUMMARY_SHA256,
                f"{family}: event-summary script mismatch",
            )

            require(
                sha256_file(candidate_script)
                == EXPECTED_CANDIDATE_SHA256,
                f"{family}: candidate script mismatch",
            )

            require(
                sha256_file(parquet_writer)
                == EXPECTED_PARQUET_WRITER_SHA256,
                f"{family}: Parquet writer mismatch",
            )

            require(
                sha256_file(delphes_card)
                == EXPECTED_DELPHES_CARD_SHA256,
                f"{family}: Delphes card mismatch",
            )

            updates = {
                "family": family,
                "worker_wrapper_sha256": worker_sha,
                "exact_target_converter_sha256":
                    EXPECTED_CONVERTER_SHA256,
                "proc_card_sha256":
                    str(row["proc_card_sha256"]),
                "run_card_sha256":
                    str(row["run_card_sha256"]),
                "param_card_sha256":
                    str(row["param_card_sha256"]),
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "inclusive_pythia_z_decays": "true",
                "all_z_forced_to_bb": "false",
                "exact_target_output": "true",
                "generator_xsec_semantics":
                    "inclusive_before_zbb_filter",
                "physics_yield_authorized": "false",
                "payload_build_git_head": current_head,
            }

            update_manifest(
                manifest,
                updates,
                remove_keys={
                    "seeded_converter_sha256",
                },
            )

            payload_path = (
                new_dir
                / f"{family}_inputs.tar.gz"
            )

            with tarfile.open(
                payload_path,
                "w:gz",
            ) as archive:
                archive.add(
                    payload_root,
                    arcname="payload",
                    recursive=True,
                )

            require(
                payload_path.is_file()
                and payload_path.stat().st_size > 0,
                f"{family}: payload archive was not built",
            )

            build_rows.append({
                "family": family,
                "payload": str(payload_path),
                "payload_sha256":
                    sha256_file(payload_path),
                "payload_size_bytes":
                    payload_path.stat().st_size,
                "worker": str(worker),
                "worker_sha256": worker_sha,
                "converter_sha256":
                    EXPECTED_CONVERTER_SHA256,
                "proc_card_sha256":
                    row["proc_card_sha256"],
                "run_card_sha256":
                    row["run_card_sha256"],
                "param_card_sha256":
                    row["param_card_sha256"],
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "exact_target_output": True,
                "physics_yield_authorized": False,
            })

    require(
        len(build_rows) == 3,
        "expected three payloads",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "tag": outdir.name,
        "worker": str(worker),
        "worker_sha256": worker_sha,
        "converter": str(converter),
        "converter_sha256":
            EXPECTED_CONVERTER_SHA256,
        "compiled_converter":
            str(compiled_converter),
        "payloads": build_rows,
        "worker_v2_shell_syntax_valid": True,
        "converter_compile_valid": True,
        "three_payloads_built": True,
        "canary_preparation_authorized": True,
        "condor_submission_authorized": False,
        "physics_yield_authorized": False,
    }

    summary_path = (
        outdir
        / "triboson_worker_v2_build.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    table_path = (
        outdir
        / "triboson_worker_v2_payloads.tsv"
    )

    columns = [
        "family",
        "payload",
        "payload_sha256",
        "payload_size_bytes",
        "worker",
        "worker_sha256",
        "converter_sha256",
        "proc_card_sha256",
        "run_card_sha256",
        "param_card_sha256",
        "filter_requirement",
        "exact_target_output",
        "physics_yield_authorized",
    ]

    with table_path.open("w") as handle:
        handle.write(
            "\t".join(columns)
            + "\n"
        )

        for row in build_rows:
            handle.write(
                "\t".join(
                    str(row[column])
                    for column in columns
                )
                + "\n"
            )

    print("===== TRIBOSON V2 PAYLOADS =====")

    for row in build_rows:
        print()
        print(f'family={row["family"]}')
        print(f'payload={row["payload"]}')
        print(
            f'payload_sha256={row["payload_sha256"]}'
        )
        print(
            "payload_size_bytes="
            f'{row["payload_size_bytes"]}'
        )

    print()
    print(
        f"worker_sha256={worker_sha}"
    )
    print(
        "converter_sha256="
        f"{EXPECTED_CONVERTER_SHA256}"
    )
    print()
    print("TRIBOSON_WORKER_V2_SHELL_SYNTAX_VALID")
    print("TRIBOSON_EXACT_TARGET_CONVERTER_COMPILE_VALID")
    print("TRIBOSON_THREE_PAYLOADS_BUILT")
    print("TRIBOSON_WORKER_V2_STATIC_VALID")
    print("TRIBOSON_CANARY_PREPARATION_AUTHORIZED")
    print("NO_TRIBOSON_CONDOR_SUBMISSION_AUTHORIZED_YET")
    print(f"summary_json={summary_path}")
    print(f"payload_table={table_path}")


if __name__ == "__main__":
    main()
