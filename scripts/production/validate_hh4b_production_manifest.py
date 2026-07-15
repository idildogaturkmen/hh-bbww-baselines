#!/usr/bin/env python3

from pathlib import Path
import hashlib
import sys
import yaml

manifest_path = Path(
    "config/production/hh4b_final_production_v1.yaml"
)

with manifest_path.open() as handle:
    cfg = yaml.safe_load(handle)

errors = []
warnings = []

if cfg["metadata"]["shard_size_events"] != 10000:
    errors.append("The production shard size must remain 10000.")

fractions = cfg["dataset_splits"]
fraction_sum = (
    fractions["train_fraction"]
    + fractions["validation_fraction"]
    + fractions["test_fraction"]
)

if abs(fraction_sum - 1.0) > 1.0e-12:
    errors.append(f"Dataset split fractions sum to {fraction_sum}, not 1.")

run2_card = Path(cfg["detector_cards"]["run2"]["path"])

if not run2_card.exists():
    errors.append(f"Missing Run-2 detector card: {run2_card}")
else:
    digest = hashlib.sha256(run2_card.read_bytes()).hexdigest()
    expected = cfg["detector_cards"]["run2"]["sha256"]

    if digest != expected:
        errors.append(
            "Run-2 detector card checksum mismatch:\n"
            f"  expected: {expected}\n"
            f"  observed: {digest}"
        )

combined = cfg["campaigns"]["combined_350fb"]
component_lumi = sum(
    item["luminosity_fb"] for item in combined["components"]
)

if component_lumi != combined["total_luminosity_fb"]:
    errors.append(
        "The combined 350 fb^-1 components do not sum to 350."
    )

enriched = cfg["processes"]["ml_enrichment"]

for process, settings in enriched.items():
    if settings.get("physics_yields", False):
        errors.append(
            f"ML enrichment process {process} must not be used "
            "for physical yields."
        )

blocked_cards = [
    ("run3", cfg["detector_cards"]["run3"]),
    ("phase2", cfg["detector_cards"]["phase2"]),
]

for name, settings in blocked_cards:
    if settings["status"].startswith("pending"):
        warnings.append(
            f"{name} production remains blocked until its card is frozen."
        )

print("Manifest:", manifest_path)
print("Name:", cfg["metadata"]["name"])
print("Run-2 card hash: verified" if not errors else "Validation completed")

if warnings:
    print("\nWarnings:")
    for warning in warnings:
        print(" -", warning)

if errors:
    print("\nErrors:")
    for error in errors:
        print(" -", error)
    sys.exit(1)

print("\nMANIFEST VALID")
