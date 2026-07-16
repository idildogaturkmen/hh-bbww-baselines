#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_directory")
    args = parser.parse_args()

    base = Path(args.pilot_directory)

    metadata_files = sorted(
        (base / "metadata").glob("*_generator.json")
    )

    if not metadata_files:
        raise SystemExit(
            f"ERROR: no generator metadata under {base}"
        )

    rows = []

    for metadata_path in metadata_files:
        metadata = json.loads(
            metadata_path.read_text()
        )

        tag = metadata_path.name.removesuffix(
            "_generator.json"
        )

        event_path = (
            base
            / "parquet"
            / f"{tag}_event_summary.parquet"
        )

        candidate_path = (
            base
            / "parquet"
            / f"{tag}_hh4b_candidates_v2.parquet"
        )

        if not event_path.is_file():
            raise SystemExit(
                f"ERROR: missing {event_path}"
            )

        if not candidate_path.is_file():
            raise SystemExit(
                f"ERROR: missing {candidate_path}"
            )

        events = pd.read_parquet(event_path)
        candidates = pd.read_parquet(candidate_path)

        n_events = len(events)

        n_4j = int(
            (
                events["n_jet_pt30_eta25"] >= 4
            ).sum()
        )

        n_2b = int(
            (
                events["n_bjet_pt30_eta25"] == 2
            ).sum()
        )

        n_3b = int(
            (
                events["n_bjet_pt30_eta25"] == 3
            ).sum()
        )

        n_4b = int(
            (
                events["n_bjet_pt30_eta25"] >= 4
            ).sum()
        )

        if (
            len(candidates) > 0
            and "r_hh" in candidates.columns
        ):
            n_rhh80 = int(
                (candidates["r_hh"] < 80).sum()
            )
            n_rhh50 = int(
                (candidates["r_hh"] < 50).sum()
            )
        else:
            n_rhh80 = 0
            n_rhh50 = 0

        sigma_pb = float(
            metadata["sigma_gen_pb"]
        )

        # Jeffreys smoothing prevents zero-count pilot
        # bins from receiving exactly zero priority.
        p4 = (
            (n_4b + 0.5)
            / (n_events + 1.0)
        )

        p_tail = (
            (n_rhh80 + 0.5)
            / (n_events + 1.0)
        )

        physics_score = (
            sigma_pb
            * np.sqrt(p4 * (1.0 - p4))
        )

        ml_tail_score = np.sqrt(p_tail)

        rows.append(
            {
                "tag": tag,
                "pthat_min_GeV":
                    metadata["pthat_min_GeV"],
                "pthat_max_GeV":
                    metadata["pthat_max_GeV"],
                "seed": metadata["seed"],
                "n_events": n_events,
                "sigma_gen_pb": sigma_pb,
                "n_4j": n_4j,
                "n_exactly_2b": n_2b,
                "n_exactly_3b": n_3b,
                "n_atleast_4b": n_4b,
                "n_rhh_lt80": n_rhh80,
                "n_rhh_lt50": n_rhh50,
                "eff_atleast_4b":
                    n_4b / max(n_events, 1),
                "eff_rhh_lt80":
                    n_rhh80 / max(n_events, 1),
                "physics_neyman_score":
                    physics_score,
                "ml_tail_score":
                    ml_tail_score,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "pthat_min_GeV"
    )

    physics_sum = result[
        "physics_neyman_score"
    ].sum()

    tail_sum = result[
        "ml_tail_score"
    ].sum()

    if physics_sum > 0:
        result["physics_allocation_fraction"] = (
            result["physics_neyman_score"]
            / physics_sum
        )
    else:
        result["physics_allocation_fraction"] = 0.0

    if tail_sum > 0:
        result["ml_tail_allocation_fraction"] = (
            result["ml_tail_score"]
            / tail_sum
        )
    else:
        result["ml_tail_allocation_fraction"] = 0.0

    output = (
        base
        / "qcd_importance_pilot_summary.csv"
    )

    result.to_csv(output, index=False)

    columns = [
        "pthat_min_GeV",
        "pthat_max_GeV",
        "n_events",
        "sigma_gen_pb",
        "n_4j",
        "n_exactly_2b",
        "n_exactly_3b",
        "n_atleast_4b",
        "n_rhh_lt80",
        "n_rhh_lt50",
        "physics_allocation_fraction",
        "ml_tail_allocation_fraction",
    ]

    print()
    print("===== QCD IMPORTANCE PILOT =====")
    print(
        result[columns].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )
    print()
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
