#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
MG5_ROOT = STORE / "mg5"

OUTDIR = REPO / "outputs/audits/ak4ak8_normalization_inputs_2026_07_15"
CARDS_OUT = OUTDIR / "cards"
OUTDIR.mkdir(parents=True, exist_ok=True)
CARDS_OUT.mkdir(parents=True, exist_ok=True)

CARD_NAMES = [
    "proc_card_mg5.dat",
    "run_card.dat",
    "param_card.dat",
    "pythia8_card.dat",
    "madspin_card.dat",
]

RUN_KEYS = {
    "ebeam1",
    "ebeam2",
    "nevents",
    "iseed",
    "pdlabel",
    "lhaid",
    "ickkw",
    "xqcut",
    "ptj",
    "ptb",
    "pta",
    "ptl",
    "etaj",
    "etab",
    "drjj",
    "drbb",
    "drbj",
    "mmjj",
    "mmbb",
    "maxjetflavor",
    "cut_decays",
    "auto_ptj_mjj",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def classify(name: str) -> str:
    low = name.lower()

    rules = [
        ("ggF_HH_4b", ["hh4b", "ggf"]),
        ("VBF_HH_4b", ["hh4b", "vbf"]),
        ("ttbar", ["ttbar"]),
        ("QCD_bbbb", ["qcd", "bbbb"]),
        ("Zbbbb", ["zbbbb"]),
        ("ZZ_4b", ["zz4b"]),
        ("ZH_4b", ["zh4b"]),
    ]

    for label, required in rules:
        if all(token in low for token in required):
            return label

    return "other"

def process_lines(path: Path):
    if not path.exists():
        return []

    keep = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        low = line.lower()
        if (
            low.startswith("import model")
            or low.startswith("define ")
            or low.startswith("generate ")
            or low.startswith("add process ")
            or low.startswith("output ")
        ):
            keep.append(line)
    return keep

def run_parameters(path: Path):
    found = {}

    if not path.exists():
        return found

    for raw in path.read_text(errors="replace").splitlines():
        # MG5 cards commonly use both ! and # for comments.
        line = re.split(r"[#!]", raw, maxsplit=1)[0].strip()

        if "=" not in line:
            continue

        left, right = [part.strip() for part in line.split("=", 1)]

        # Support both:
        #     10000 = nevents
        # and:
        #     nevents = 10000
        if right in RUN_KEYS:
            found[right] = left
        elif left in RUN_KEYS:
            found[left] = right

    return found

rows = []
process_cards = sorted(MG5_ROOT.rglob("Cards/proc_card_mg5.dat"))

for proc_card in process_cards:
    process_dir = proc_card.parent.parent
    relative = process_dir.relative_to(MG5_ROOT)
    safe_name = str(relative).replace("/", "__")
    snapshot_dir = CARDS_OUT / safe_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    card_hashes = {}

    for card_name in CARD_NAMES:
        source = process_dir / "Cards" / card_name
        if source.exists():
            destination = snapshot_dir / card_name
            shutil.copy2(source, destination)
            card_hashes[card_name] = sha256(source)

    row = {
        "classification": classify(str(relative)),
        "process_directory": str(process_dir),
        "relative_process_directory": str(relative),
        "process_definition": " | ".join(process_lines(proc_card)),
        "run_parameters": json.dumps(
            run_parameters(process_dir / "Cards" / "run_card.dat"),
            sort_keys=True,
        ),
        "card_hashes": json.dumps(card_hashes, sort_keys=True),
    }
    rows.append(row)

delphes_card = REPO / "cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl"
delphes_info = {
    "path": str(delphes_card),
    "exists": delphes_card.exists(),
    "sha256": sha256(delphes_card) if delphes_card.exists() else None,
}

with (OUTDIR / "mg5_process_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        "classification",
        "process_directory",
        "relative_process_directory",
        "process_definition",
        "run_parameters",
        "card_hashes",
    ])
    writer.writeheader()
    writer.writerows(rows)

with (OUTDIR / "mg5_process_manifest.json").open("w") as f:
    json.dump(
        {
            "processes": rows,
            "delphes_card": delphes_info,
        },
        f,
        indent=2,
    )

md = [
    "# AK4/AK8 normalization-input audit",
    "",
    "## Delphes card",
    "",
    f"- Path: `{delphes_info['path']}`",
    f"- Exists: `{delphes_info['exists']}`",
    f"- SHA256: `{delphes_info['sha256']}`",
    "",
    "## MG5 process definitions",
    "",
]

for row in rows:
    md.extend([
        f"### {row['classification']}: `{row['relative_process_directory']}`",
        "",
        "**Process definition**",
        "",
        "```text",
        row["process_definition"] or "(none found)",
        "```",
        "",
        "**Important run-card parameters**",
        "",
        "```json",
        row["run_parameters"],
        "```",
        "",
    ])

(OUTDIR / "process_definitions.md").write_text("\n".join(md) + "\n")

print(f"Found {len(rows)} MG5 process directories")
print(f"Wrote: {OUTDIR / 'process_definitions.md'}")
print(f"Wrote: {OUTDIR / 'mg5_process_manifest.csv'}")
print(f"Wrote: {OUTDIR / 'mg5_process_manifest.json'}")
print(f"Copied card snapshots to: {CARDS_OUT}")
