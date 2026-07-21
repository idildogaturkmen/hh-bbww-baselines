#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


USER = Path.home().name

REPO = Path(
    f"/uscms_data/d3/{USER}/repos/hh-bbww-baselines"
)

STORE = Path(
    f"/uscms_data/d3/{USER}/hh4b_delphes"
)

PLAN = (
    REPO
    / "metadata/production_plans"
    / "waveb_priority_smoke_templates_20260721.tsv"
)

PROCESS_ROOT = (
    STORE
    / "mg5"
    / "waveb_priority_smokes_20260721"
)

AUDIT_ROOT = (
    REPO
    / "outputs/agent_runs"
    / "waveb_priority_lhe_smokes_20260721"
)

EXPECTED_EVENTS = 100

TOPOLOGY_REQUIREMENTS = {
    "tth_hbb": {
        6: 2,
        5: 2,
        25: 1,
    },
    "ttz_zbb": {
        6: 2,
        5: 2,
        23: 1,
    },
    "tttt": {
        6: 4,
    },
    "vbf_hbb": {
        5: 2,
        25: 1,
    },
}


def command_output(
    command: list[str],
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    return result.stdout.strip()


def check_repository() -> str:
    local_head = command_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
    )

    remote_head = command_output(
        [
            "git",
            "rev-parse",
            "origin/delphes-hh4b-production",
        ],
        cwd=REPO,
    )

    status = command_output(
        ["git", "status", "--porcelain"],
        cwd=REPO,
    )

    if remote_head != local_head:
        raise RuntimeError(
            "Local and remote commits differ"
        )

    if status:
        dirty_summary = "; ".join(
            status.splitlines()
        )

        raise RuntimeError(
            "Repository is not clean: "
            + dirty_summary
        )

    return local_head


def load_plan() -> dict[str, dict[str, str]]:
    if not PLAN.is_file():
        raise RuntimeError(
            f"Missing smoke plan: {PLAN}"
        )

    with PLAN.open(newline="") as handle:
        rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    result = {
        row["family"]: row
        for row in rows
    }

    if set(result) != set(TOPOLOGY_REQUIREMENTS):
        raise RuntimeError(
            "Smoke-plan families do not match "
            "the expected set"
        )

    for family, row in result.items():
        if int(row["events"]) != EXPECTED_EVENTS:
            raise RuntimeError(
                f"{family} does not request 100 events"
            )

        if row["dataset_split"] == "test":
            raise RuntimeError(
                "Final test data are not authorized"
            )

    return result


def open_lhe(path: Path):
    if path.suffix == ".gz":
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    return path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    )


def parse_lhe(
    path: Path,
) -> tuple[
    int,
    float,
    list[Counter[int]],
]:
    event_particle_counts: list[Counter[int]] = []
    current_particles: Counter[int] | None = None
    inside_init = False
    init_data_lines: list[str] = []

    with open_lhe(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line == "<init>":
                inside_init = True
                continue

            if line == "</init>":
                inside_init = False
                continue

            if inside_init:
                if (
                    line
                    and not line.startswith("#")
                    and not line.startswith("<")
                ):
                    init_data_lines.append(line)

                continue

            if line == "<event>":
                current_particles = Counter()
                continue

            if line == "</event>":
                if current_particles is None:
                    raise RuntimeError(
                        "Malformed LHE event block"
                    )

                event_particle_counts.append(
                    current_particles
                )

                current_particles = None
                continue

            if current_particles is None:
                continue

            if (
                not line
                or line.startswith("#")
                or line.startswith("<")
            ):
                continue

            fields = line.split()

            if len(fields) < 2:
                continue

            try:
                pdg_id = int(fields[0])
                status = int(fields[1])
            except ValueError:
                continue

            if status in {-1, 1, 2}:
                current_particles[abs(pdg_id)] += 1

    if len(init_data_lines) < 3:
        raise RuntimeError(
            "Could not parse the LHE init block"
        )

    try:
        n_processes = int(
            init_data_lines[1].split()[0]
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not parse NPRUP"
        ) from exc

    process_lines = init_data_lines[
        2 : 2 + n_processes
    ]

    if len(process_lines) != n_processes:
        raise RuntimeError(
            "Incomplete LHE process table"
        )

    cross_section_pb = 0.0

    for line in process_lines:
        fields = line.split()

        if not fields:
            continue

        cross_section_pb += float(fields[0])

    return (
        len(event_particle_counts),
        cross_section_pb,
        event_particle_counts,
    )


def locate_lhe(
    process_dir: Path,
    run_name: str,
) -> Path:
    run_dir = (
        process_dir
        / "Events"
        / run_name
    )

    candidates = sorted(
        list(
            run_dir.glob(
                "*unweighted_events.lhe*"
            )
        )
        + list(
            run_dir.glob(
                "*events.lhe*"
            )
        )
    )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    unique_candidates = []

    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)

    if len(unique_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one LHE file in "
            f"{run_dir}, found "
            f"{len(unique_candidates)}"
        )

    return unique_candidates[0]


def choose_generation_command(
    generator: Path,
    run_name: str,
    help_text: str,
) -> list[str]:
    if "--parton" in help_text:
        return [
            str(generator),
            run_name,
            "--parton",
            "-f",
        ]

    if "--laststep" in help_text:
        return [
            str(generator),
            run_name,
            "--laststep=parton",
            "-f",
        ]

    raise RuntimeError(
        "generate_events help exposes neither "
        "--parton nor --laststep"
    )


def validate_topology(
    family: str,
    events: list[Counter[int]],
) -> dict[str, object]:
    requirements = TOPOLOGY_REQUIREMENTS[
        family
    ]

    minima = {
        str(pdg_id): min(
            event.get(pdg_id, 0)
            for event in events
        )
        for pdg_id in requirements
    }

    failing_events = []

    for index, event in enumerate(events):
        missing = {
            str(pdg_id): {
                "required": required,
                "observed": event.get(
                    pdg_id,
                    0,
                ),
            }
            for pdg_id, required
            in requirements.items()
            if event.get(pdg_id, 0) < required
        }

        if missing:
            failing_events.append({
                "event_index": index,
                "missing": missing,
            })

    return {
        "requirements": {
            str(key): value
            for key, value
            in requirements.items()
        },
        "minimum_particle_counts": minima,
        "failing_event_count": len(
            failing_events
        ),
        "first_failing_events": (
            failing_events[:5]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(
            TOPOLOGY_REQUIREMENTS
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    plan = load_plan()
    row = plan[args.family]

    process_dir = (
        PROCESS_ROOT
        / row["template_name"]
    )

    generator = (
        process_dir
        / "bin"
        / "generate_events"
    )

    run_name = (
        f"waveb_{args.family}"
        "_lhe_smoke100_20260721"
    )

    run_dir = (
        process_dir
        / "Events"
        / run_name
    )

    audit_dir = (
        AUDIT_ROOT
        / args.family
    )

    log_path = (
        audit_dir
        / "generate_events.log"
    )

    summary_path = (
        audit_dir
        / "lhe_validation.json"
    )

    if not process_dir.is_dir():
        raise RuntimeError(
            f"Missing process directory: "
            f"{process_dir}"
        )

    if not generator.is_file():
        raise RuntimeError(
            f"Missing generate_events: "
            f"{generator}"
        )

    if not (
        process_dir
        / "Cards"
        / "run_card.dat"
    ).is_file():
        raise RuntimeError(
            "Missing configured run card"
        )

    help_result = subprocess.run(
        [
            str(generator),
            "--help",
        ],
        cwd=process_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    command = choose_generation_command(
        generator,
        run_name,
        help_result.stdout,
    )

    print(f"family={args.family}")
    print(f"process_directory={process_dir}")
    print(f"run_name={run_name}")
    print(
        "requested_events="
        f"{row['events']}"
    )
    print(
        "generation_command="
        + " ".join(command)
    )

    if args.validate_only:
        print(
            "repository_gate=skipped_for_validate_only"
        )
        print(
            "WAVEB_LHE_SMOKE_RUNNER_VALID"
        )
        return

    local_head = check_repository()

    print(f"git_head={local_head}")

    if run_dir.exists():
        raise RuntimeError(
            "Refusing to overwrite existing run: "
            f"{run_dir}"
        )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open("w") as log_handle:
        result = subprocess.run(
            command,
            cwd=process_dir,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(
        "generate_events_exit_code="
        f"{result.returncode}"
    )

    print(f"generation_log={log_path}")

    if result.returncode != 0:
        raise RuntimeError(
            "MG5 event generation failed"
        )

    lhe_path = locate_lhe(
        process_dir,
        run_name,
    )

    (
        event_count,
        cross_section_pb,
        event_particle_counts,
    ) = parse_lhe(lhe_path)

    topology = validate_topology(
        args.family,
        event_particle_counts,
    )

    summary = {
        "schema_version": 1,
        "family": args.family,
        "process_directory": str(
            process_dir
        ),
        "run_name": run_name,
        "payload_git_head": local_head,
        "requested_events": (
            EXPECTED_EVENTS
        ),
        "lhe_file": str(lhe_path),
        "lhe_events": event_count,
        "generator_cross_section_pb": (
            cross_section_pb
        ),
        "topology": topology,
        "generation_log": str(log_path),
        "authorized_for_pilot_scaleout": (
            False
        ),
        "authorized_for_final_test": (
            False
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"lhe_file={lhe_path}")
    print(f"lhe_events={event_count}")
    print(
        "generator_cross_section_pb="
        f"{cross_section_pb:.12g}"
    )
    print(
        "topology_failing_events="
        f"{topology['failing_event_count']}"
    )
    print(f"summary={summary_path}")

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            "LHE event count is not 100"
        )

    if cross_section_pb <= 0:
        raise RuntimeError(
            "Generator cross section is not positive"
        )

    if topology[
        "failing_event_count"
    ] != 0:
        raise RuntimeError(
            "Generated events failed topology "
            "validation"
        )

    print(
        "WAVEB_LHE_SMOKE_COMPLETE_AND_VALID"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
