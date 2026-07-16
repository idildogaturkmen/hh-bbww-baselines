#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Write and round-trip-check a trusted temporary pandas pickle as Parquet."
    )
    parser.add_argument("--input", required=True, help="Trusted temporary DataFrame pickle")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = pd.read_pickle(input_path)
    frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    frame.to_parquet(output_path, index=False)
    round_trip = pd.read_parquet(output_path)

    if len(round_trip) != len(frame) or list(round_trip.columns) != list(frame.columns):
        raise SystemExit("ERROR: Parquet round-trip shape or columns disagree")

    print(f"PARQUET_ROUND_TRIP_ROWS={len(round_trip)}")
    print(f"PARQUET_ROUND_TRIP_COLUMNS={len(round_trip.columns)}")

    # In the EL9 LCG 106 worker image, pandas/pyarrow can abort in C++ static
    # destruction after a valid empty (0-row, 0-column) Parquet round trip.
    # Everything durable is complete at this point. Flush diagnostics and skip
    # only interpreter/library teardown on the verified-success path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
