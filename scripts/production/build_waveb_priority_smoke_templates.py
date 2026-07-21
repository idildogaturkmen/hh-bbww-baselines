#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
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

OUTPUT_ROOT = (
    STORE
    / "mg5"
    / "waveb_priority_smokes_20260721"
)

AUDIT_DIR = (
    REPO
    / "outputs/agent_runs"
    / "waveb_priority_smoke_templates_20260721"
)

CARD_DIR = AUDIT_DIR / "process_cards"
LOG_DIR = AUDIT_DIR / "template_build_logs"

SUMMARY_PATH = (
    AUDIT_DIR
    / "template_build_summary.json"
)

MG5 = Path(
    "/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/"
    "madgraph5amc/3.5.3.atlas7-2c347/"
    "x86_64-el9-gcc13-opt/bin/mg5_aMC"
)

REFERENCE_RUN_CARD = (
    STORE
    / "mg5"
    / "Zbbbb_presel_smoke"
    / "Cards"
    / "run_card.dat"
)

EXPECTED_FAMILIES = {
    "tth_hbb",
    "ttz_zbb",
    "tttt",
    "vbf_hbb",
}

PROHIBITED_TOKENS = {
    "m_bb",
    "mbb",
    "m_hh",
    "mhh",
    "r_hh",
    "rhh",
    "classifier",
    "score",
    "signal_region",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_plan() -> list[dict[str, str]]:
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

    if len(rows) != 4:
        raise RuntimeError(
            f"Expected 4 smoke rows, found {len(rows)}"
        )

    families = {
        row["family"]
        for row in rows
    }

    if families != EXPECTED_FAMILIES:
        raise RuntimeError(
            f"Unexpected families: {families}"
        )

    templates = [
        row["template_name"]
        for row in rows
    ]

    if len(set(templates)) != len(templates):
        raise RuntimeError(
            "Template names are not unique"
        )

    seeds = [
        int(row["seed"])
        for row in rows
    ]

    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            "Seeds are not unique"
        )

    if any(
        int(row["events"]) != 100
        for row in rows
    ):
        raise RuntimeError(
            "Every smoke template must request 100 events"
        )

    for row in rows:
        process = row["process_line"].lower()

        for token in PROHIBITED_TOKENS:
            if token in process:
                raise RuntimeError(
                    "Prohibited analysis-variable selection "
                    f"in {row['family']}: {token}"
                )

        if row["dataset_split"] == "test":
            raise RuntimeError(
                "Final test data are not authorized"
            )

    return rows


def check_repository() -> tuple[str, str]:
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

    if local_head != remote_head:
        raise RuntimeError(
            "Local and remote commits do not match"
        )

    if status:
        raise RuntimeError(
            "Repository is not clean"
        )

    return local_head, remote_head


def process_card(
    process_line: str,
    output_dir: Path,
) -> str:
    return "\n".join([
        "import model sm",
        (
            "define p = "
            "g u c d s u~ c~ d~ s~"
        ),
        (
            "define j = "
            "g u c d s u~ c~ d~ s~"
        ),
        "define l+ = e+ mu+",
        "define l- = e- mu-",
        "define vl = ve vm vt",
        "define vl~ = ve~ vm~ vt~",
        process_line,
        f"output {output_dir} -f",
        "",
    ])


def replace_setting(
    text: str,
    key: str,
    value: str,
    required: bool = True,
) -> str:
    pattern = re.compile(
        rf"^(?P<line>.*=\s*{re.escape(key)}\b.*)$",
        flags=re.MULTILINE,
    )

    matches = list(pattern.finditer(text))

    if not matches:
        if required:
            raise RuntimeError(
                f"Run-card setting not found: {key}"
            )
        return text

    if len(matches) != 1:
        raise RuntimeError(
            f"Run-card setting appears "
            f"{len(matches)} times: {key}"
        )

    old_line = matches[0].group("line")

    comment = ""

    if "!" in old_line:
        comment = " !" + old_line.split(
            "!",
            1,
        )[1]

    new_line = f"  {value} = {key}{comment}"

    return (
        text[:matches[0].start()]
        + new_line
        + text[matches[0].end():]
    )


def configure_run_card(
    path: Path,
    row: dict[str, str],
) -> None:
    shutil.copy2(
        REFERENCE_RUN_CARD,
        path,
    )

    text = path.read_text()

    is_ew_hjj = (
        row["family"] == "vbf_hbb"
    )

    required_settings = {
        "nevents": str(int(row["events"])),
        "iseed": str(int(row["seed"])),
        "lpp1": "1",
        "lpp2": "1",
        "ebeam1": "6500.0",
        "ebeam2": "6500.0",
        "pdlabel": "nn23lo1",
        "lhaid": "230000",
        "event_norm": "average",
        "cut_decays": "False",
        "ptb": "0.0",
        "ptbmax": "-1.0",
        "etab": "-1.0",
        "etabmin": "0.0",
        "drbb": "0.0",
        "drbbmax": "-1.0",
        "drbj": "0.0",
        "ihtmin": "0.0",
        "ihtmax": "-1.0",
        "ptj": (
            "20.0"
            if is_ew_hjj
            else "0.0"
        ),
        "etaj": (
            "5.0"
            if is_ew_hjj
            else "-1.0"
        ),
    }

    for key, value in required_settings.items():
        text = replace_setting(
            text,
            key,
            value,
            required=True,
        )

    optional_settings = {
        "ickkw": "0",
        "drjl": "0.0",
        "auto_ptj_mjj": "False",
    }

    for key, value in optional_settings.items():
        text = replace_setting(
            text,
            key,
            value,
            required=False,
        )

    path.write_text(text)


def build_templates(
    rows: list[dict[str, str]],
    git_head: str,
) -> None:
    if not MG5.is_file():
        raise RuntimeError(
            f"Missing MG5 executable: {MG5}"
        )

    if not REFERENCE_RUN_CARD.is_file():
        raise RuntimeError(
            "Missing reference run card: "
            f"{REFERENCE_RUN_CARD}"
        )

    CARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for row in rows:
        template_name = row["template_name"]
        output_dir = OUTPUT_ROOT / template_name
        card_path = CARD_DIR / f"{template_name}.mg5"
        log_path = LOG_DIR / f"{template_name}.log"

        if output_dir.exists():
            raise RuntimeError(
                "Refusing to overwrite existing "
                f"template: {output_dir}"
            )

        card_path.write_text(
            process_card(
                row["process_line"],
                output_dir,
            )
        )

        with log_path.open("w") as log_handle:
            result = subprocess.run(
                [str(MG5), str(card_path)],
                cwd=REPO,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if result.returncode != 0:
            raise RuntimeError(
                "MG5 template construction failed "
                f"for {row['family']}; "
                f"see {log_path}"
            )

        version_path = (
            output_dir
            / "MGMEVersion.txt"
        )

        generator = (
            output_dir
            / "bin"
            / "generate_events"
        )

        run_card_path = (
            output_dir
            / "Cards"
            / "run_card.dat"
        )

        if not version_path.is_file():
            raise RuntimeError(
                f"Missing MGMEVersion: {output_dir}"
            )

        version = version_path.read_text().strip()

        if version != "3.5.3":
            raise RuntimeError(
                f"Unexpected MG5 version: {version}"
            )

        if not generator.is_file():
            raise RuntimeError(
                f"Missing generate_events: {generator}"
            )

        configure_run_card(
            run_card_path,
            row,
        )

        records.append({
            **row,
            "process_directory": str(output_dir),
            "process_card": str(card_path),
            "process_card_sha256": sha256(card_path),
            "run_card": str(run_card_path),
            "run_card_sha256": sha256(run_card_path),
            "mg5_version": version,
            "payload_git_head": git_head,
            "template_status": "built_not_generated",
        })

        print(
            f"built={row['family']} "
            f"directory={output_dir}"
        )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "git_head": git_head,
                "events_generated": 0,
                "templates_built": len(records),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"summary={SUMMARY_PATH}")
    print(f"built_templates={len(records)}")
    print("WAVEB_PRIORITY_SMOKE_TEMPLATES_BUILT")


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True,
    )

    mode.add_argument(
        "--validate-only",
        action="store_true",
    )

    mode.add_argument(
        "--build",
        action="store_true",
    )

    args = parser.parse_args()

    rows = load_plan()

    print(f"plan={PLAN}")
    print(f"pilot_rows={len(rows)}")
    print(
        "requested_smoke_events="
        f"{sum(int(row['events']) for row in rows)}"
    )
    print("WAVEB_PRIORITY_SMOKE_PLAN_VALID")

    if args.validate_only:
        return

    local_head, remote_head = (
        check_repository()
    )

    print(f"local_head={local_head}")
    print(f"remote_head={remote_head}")

    build_templates(
        rows,
        local_head,
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
