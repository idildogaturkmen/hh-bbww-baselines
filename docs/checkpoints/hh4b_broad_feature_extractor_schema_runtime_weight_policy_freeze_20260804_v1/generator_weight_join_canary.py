from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def req(ok, message):
    if not ok:
        raise RuntimeError(message)


def clean(value):
    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def yes(value):
    return clean(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def check_checksum_manifest(directory):
    directory = Path(directory)
    sums = directory / "SHA256SUMS"

    req(
        sums.is_file(),
        f"missing checksum manifest: {sums}",
    )

    checked = 0

    for raw in sums.read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw.strip():
            continue

        expected, relative = raw.split(
            "  ",
            1,
        )

        path = directory / relative

        req(
            path.is_file(),
            f"missing checksummed file: {path}",
        )

        req(
            sha256(path) == expected,
            f"checksum mismatch: {path}",
        )

        checked += 1

    req(
        checked > 0,
        f"empty checksum manifest: {sums}",
    )


def read_weight_table(path):
    path = Path(path)

    req(
        path.is_file(),
        f"missing weight table: {path}",
    )

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(
            path,
            sep="\t",
        )

    required = {
        "event",
        "generator_nominal_weight",
    }

    req(
        required.issubset(
            frame.columns
        ),
        (
            f"bad weight-table schema "
            f"for {path}: "
            f"{list(frame.columns)}"
        ),
    )

    frame = frame[
        [
            "event",
            "generator_nominal_weight",
        ]
    ].copy()

    frame["event"] = pd.to_numeric(
        frame["event"],
        errors="raise",
    ).astype("int64")

    frame[
        "generator_nominal_weight"
    ] = pd.to_numeric(
        frame[
            "generator_nominal_weight"
        ],
        errors="raise",
    ).astype("float64")

    req(
        frame["event"].is_unique,
        f"duplicate event keys: {path}",
    )

    req(
        np.isfinite(
            frame[
                "generator_nominal_weight"
            ]
        ).all(),
        f"non-finite weights: {path}",
    )

    return frame


def compare_weights(
    observed,
    expected,
    label,
):
    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    expected = np.asarray(
        expected,
        dtype=np.float64,
    )

    req(
        observed.shape == expected.shape,
        f"{label}: shape mismatch",
    )

    req(
        (
            np.isfinite(
                observed
            ).all()
            and np.isfinite(
                expected
            ).all()
        ),
        f"{label}: non-finite weights",
    )

    req(
        np.allclose(
            observed,
            expected,
            rtol=2e-6,
            atol=2e-9,
        ),
        (
            f"{label}: maximum absolute "
            "mismatch="
            f"{np.max(np.abs(observed - expected)):.17g}"
        ),
    )


def add_columns(
    table,
    values,
):
    result = table

    for name, array in values.items():
        if name in result.column_names:
            result = result.set_column(
                result.column_names.index(
                    name
                ),
                name,
                array,
            )
        else:
            result = result.append_column(
                name,
                array,
            )

    return result


def write_table(
    table,
    path,
):
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


parser = argparse.ArgumentParser()

parser.add_argument(
    "--manifest",
    type=Path,
    required=True,
)

parser.add_argument(
    "--transport",
    type=Path,
    required=True,
)

parser.add_argument(
    "--feature-out",
    type=Path,
    required=True,
)

parser.add_argument(
    "--output-dir",
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

transport = pd.read_csv(
    args.transport,
    sep="\t",
)

receipts = pd.read_csv(
    args.feature_out
    / (
        "seven_class_feature_"
        "materialization_receipts.tsv"
    ),
    sep="\t",
)

feature_summary = json.loads(
    (
        args.feature_out
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)


req(
    feature_summary.get("status")
    == "pass",
    (
        "feature materialization "
        "summary is not pass"
    ),
)

req(
    (
        len(manifest) == 464
        and manifest[
            "source_uid"
        ]
        .astype(str)
        .nunique()
        == 464
    ),
    "manifest identity closure failed",
)

req(
    int(
        pd.to_numeric(
            manifest[
                "generated_events"
            ]
        ).sum()
    )
    == 3_799_873,
    "manifest event closure failed",
)

req(
    len(transport) == 441,
    (
        f"transport rows={len(transport)}, "
        "expected 441"
    ),
)

req(
    (
        len(receipts) == 7
        and receipts[
            "status"
        ]
        .astype(str)
        .eq("pass")
        .all()
    ),
    (
        "feature receipts are not "
        "7/7 pass"
    ),
)


required_transport_columns = {
    "population_kind",
    "transport_id",
    "final_split",
    "generated_events",
    "transport_class",
    "generator_weight_provenance",
    "constant_generator_weight",
    "sidecar_or_fragment_path",
    "sidecar_or_fragment_sha256",
    "transport_complete",
    "validation_or_test_payload_opened",
    "physical_weight_application_authorized",
    "physical_yield_calculation_authorized",
}

req(
    required_transport_columns.issubset(
        transport.columns
    ),
    (
        "transport schema missing: "
        f"{sorted(required_transport_columns - set(transport.columns))}"
    ),
)


manifest_by_uid = (
    manifest.set_index(
        "source_uid",
        drop=False,
    )
)

transport_by_key = {}

for _, row in transport.iterrows():
    key = (
        clean(
            row[
                "population_kind"
            ]
        ),
        clean(
            row[
                "transport_id"
            ]
        ),
    )

    req(
        key not in transport_by_key,
        f"duplicate transport key: {key}",
    )

    transport_by_key[key] = row


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


stats = {
    "primary_sources": 0,
    "aux_sources": 0,
    "primary_rows": 0,
    "aux_rows": 0,
    "raw_rows": 0,
    "uniform": 0,
    "signed": 0,
    "qcd_variable": 0,
    "qcd_overlap": 0,
    "byte_identical": 0,
}

receipt_rows = []


for receipt in receipts.itertuples(
    index=False
):
    canary_class = clean(
        receipt.canary_class
    )

    source_uid = clean(
        receipt.source_uid
    )

    req(
        source_uid
        in manifest_by_uid.index,
        (
            "missing manifest source: "
            f"{source_uid}"
        ),
    )

    manifest_row = (
        manifest_by_uid.loc[
            source_uid
        ]
    )

    feature_path = Path(
        clean(
            receipt.parquet_a
        )
    )

    req(
        (
            feature_path.is_file()
            and sha256(
                feature_path
            )
            == clean(
                receipt.parquet_sha256
            )
        ),
        (
            "feature checksum mismatch: "
            f"{feature_path}"
        ),
    )

    table = pq.read_table(
        feature_path
    )

    data = table.to_pydict()
    row_count = table.num_rows

    req(
        (
            row_count == 512
            and data[
                "source_entry"
            ]
            == list(
                range(512)
            )
        ),
        (
            "feature identity drift: "
            f"{canary_class}"
        ),
    )

    req(
        set(
            data["source_uid"]
        )
        == {source_uid},
        (
            "source_uid drift: "
            f"{canary_class}"
        ),
    )


    workflow_population = clean(
        manifest_row[
            "workflow_population"
        ]
    )

    is_auxiliary = (
        workflow_population
        == (
            "auxiliary_qcd_"
            "classification_only"
        )
    )

    process = clean(
        manifest_row[
            "process_or_mode"
        ]
    )

    population_kind = (
        "hard_qcd"
        if process == "qcd_hardqcd"
        else "ordinary"
    )

    transport_id = clean(
        manifest_row[
            "transport_id"
        ]
    )

    transport_key = (
        population_kind,
        transport_id,
    )

    raw_available = np.asarray(
        data[
            "raw_event_weight_available"
        ],
        dtype=bool,
    )

    raw_weights = np.asarray(
        [
            (
                np.nan
                if value is None
                else float(value)
            )
            for value
            in data[
                "raw_event_weight"
            ]
        ],
        dtype=np.float64,
    )

    overlap_rows = 0


    if is_auxiliary:
        req(
            transport_key
            not in transport_by_key,
            (
                "auxiliary source unexpectedly "
                "has physical transport: "
                f"{source_uid}"
            ),
        )

        req(
            not yes(
                manifest_row[
                    "physical_evaluation_eligible"
                ]
            ),
            (
                "auxiliary source is "
                "physical-evaluation eligible: "
                f"{source_uid}"
            ),
        )

        transport_class = (
            "not_applicable_auxiliary_qcd"
        )

        provenance = (
            "classification_only_"
            "no_physical_transport"
        )

        validation_mode = (
            "excluded_by_auxiliary_qcd_contract"
        )

        enriched = add_columns(
            table,
            {
                (
                    "generator_weight_"
                    "transport_applicable"
                ): pa.array(
                    [False]
                    * row_count,
                    type=pa.bool_(),
                ),
                "population_kind": (
                    pa.array(
                        [
                            "auxiliary_qcd"
                        ]
                        * row_count,
                        type=pa.string(),
                    )
                ),
                "transport_id": (
                    pa.array(
                        [
                            transport_id
                        ]
                        * row_count,
                        type=pa.string(),
                    )
                ),
                (
                    "generator_weight_"
                    "transport_class"
                ): pa.array(
                    [
                        transport_class
                    ]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "generator_weight_"
                    "provenance"
                ): pa.array(
                    [provenance]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "generator_nominal_weight"
                ): pa.array(
                    [None]
                    * row_count,
                    type=pa.float64(),
                ),
                (
                    "generator_weight_sign"
                ): pa.array(
                    [None]
                    * row_count,
                    type=pa.int8(),
                ),
                (
                    "generator_weight_"
                    "join_status"
                ): pa.array(
                    [
                        validation_mode
                    ]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "physical_weight_"
                    "application_authorized"
                ): pa.array(
                    [False]
                    * row_count,
                    type=pa.bool_(),
                ),
            },
        )

        stats[
            "aux_sources"
        ] += 1

        stats[
            "aux_rows"
        ] += row_count

    else:
        req(
            transport_key
            in transport_by_key,
            (
                "missing transport: "
                f"{transport_key}"
            ),
        )

        transport_row = (
            transport_by_key[
                transport_key
            ]
        )

        req(
            clean(
                transport_row[
                    "final_split"
                ]
            )
            == "train",
            (
                "non-train transport: "
                f"{transport_key}"
            ),
        )

        req(
            int(
                transport_row[
                    "generated_events"
                ]
            )
            == int(
                manifest_row[
                    "generated_events"
                ]
            ),
            (
                "generated-event mismatch: "
                f"{transport_key}"
            ),
        )

        req(
            yes(
                transport_row[
                    "transport_complete"
                ]
            ),
            (
                "incomplete transport: "
                f"{transport_key}"
            ),
        )

        req(
            not yes(
                transport_row[
                    "validation_or_test_"
                    "payload_opened"
                ]
            ),
            (
                "sealed payload flag changed: "
                f"{transport_key}"
            ),
        )

        req(
            not yes(
                transport_row[
                    "physical_weight_"
                    "application_authorized"
                ]
            ),
            (
                "physical application "
                "unexpectedly authorized: "
                f"{transport_key}"
            ),
        )

        req(
            not yes(
                transport_row[
                    "physical_yield_"
                    "calculation_authorized"
                ]
            ),
            (
                "physical-yield calculation "
                "unexpectedly authorized: "
                f"{transport_key}"
            ),
        )

        req(
            (
                raw_available.all()
                and np.isfinite(
                    raw_weights
                ).all()
            ),
            (
                "primary raw ROOT weights "
                f"are incomplete: "
                f"{canary_class}"
            ),
        )


        transport_class = clean(
            transport_row[
                "transport_class"
            ]
        )

        provenance = clean(
            transport_row[
                "generator_weight_provenance"
            ]
        )


        if transport_class in {
            "uniform_member_constant",
            "qcd_uniform_constant",
        }:
            compare_weights(
                raw_weights,
                np.full(
                    row_count,
                    float(
                        transport_row[
                            "constant_"
                            "generator_weight"
                        ]
                    ),
                ),
                (
                    f"{canary_class} "
                    "uniform constant"
                ),
            )

            stats[
                "uniform"
            ] += 1

            validation_mode = (
                "raw_root_weight_matches_"
                "frozen_constant"
            )


        elif (
            transport_class
            == "signed_exact_event_sidecar"
        ):
            sidecar = Path(
                clean(
                    transport_row[
                        "sidecar_or_"
                        "fragment_path"
                    ]
                )
            )

            req(
                (
                    sidecar.is_file()
                    and sha256(
                        sidecar
                    )
                    == clean(
                        transport_row[
                            "sidecar_or_"
                            "fragment_sha256"
                        ]
                    )
                ),
                (
                    "signed sidecar checksum "
                    f"mismatch: {sidecar}"
                ),
            )

            sidecar_weights = (
                read_weight_table(
                    sidecar
                )
            )

            req(
                len(
                    sidecar_weights
                )
                == int(
                    transport_row[
                        "generated_events"
                    ]
                ),
                (
                    "signed sidecar is not "
                    f"member-wide: {sidecar}"
                ),
            )

            selected = pd.DataFrame({
                "event": np.arange(
                    row_count,
                    dtype=np.int64,
                )
            }).merge(
                sidecar_weights,
                on="event",
                how="left",
                validate="one_to_one",
            )

            req(
                selected[
                    "generator_nominal_weight"
                ]
                .notna()
                .all(),
                (
                    "signed sidecar join "
                    f"incomplete: {sidecar}"
                ),
            )

            compare_weights(
                raw_weights,
                selected[
                    "generator_nominal_weight"
                ].to_numpy(),
                (
                    f"{canary_class} "
                    "signed sidecar"
                ),
            )

            stats[
                "signed"
            ] += 1

            overlap_rows = row_count

            validation_mode = (
                "raw_root_weight_matches_"
                "member_wide_signed_sidecar"
            )


        elif (
            transport_class
            == "qcd_variable_frozen_fragment"
        ):
            fragment = Path(
                clean(
                    transport_row[
                        "sidecar_or_"
                        "fragment_path"
                    ]
                )
            )

            req(
                fragment.is_dir(),
                (
                    "missing QCD fragment: "
                    f"{fragment}"
                ),
            )

            sums = (
                fragment
                / "SHA256SUMS"
            )

            req(
                (
                    sums.is_file()
                    and sha256(
                        sums
                    )
                    == clean(
                        transport_row[
                            "sidecar_or_"
                            "fragment_sha256"
                        ]
                    )
                ),
                (
                    "QCD fragment manifest "
                    f"mismatch: {fragment}"
                ),
            )

            check_checksum_manifest(
                fragment
            )

            req(
                (
                    fragment
                    / "COMPLETE"
                ).is_file(),
                (
                    "missing QCD COMPLETE: "
                    f"{fragment}"
                ),
            )

            fragment_weights = (
                read_weight_table(
                    fragment
                    / (
                        "train_generator_"
                        "weight_sidecar.tsv"
                    )
                )
            )

            overlap = pd.DataFrame({
                "event": np.arange(
                    row_count,
                    dtype=np.int64,
                )
            }).merge(
                fragment_weights,
                on="event",
                how="inner",
                validate="one_to_one",
            )

            overlap_rows = len(
                overlap
            )

            if overlap_rows:
                event_indices = (
                    overlap[
                        "event"
                    ].to_numpy(
                        dtype=np.int64
                    )
                )

                compare_weights(
                    raw_weights[
                        event_indices
                    ],
                    overlap[
                        "generator_nominal_weight"
                    ].to_numpy(),
                    (
                        f"{canary_class} "
                        "QCD overlap"
                    ),
                )

            stats[
                "qcd_variable"
            ] += 1

            stats[
                "qcd_overlap"
            ] += overlap_rows

            validation_mode = (
                "raw_root_weight_plus_"
                "frozen_qcd_fragment"
                + (
                    "_with_overlap"
                    if overlap_rows
                    else (
                        "_no_overlap_"
                        "in_first512"
                    )
                )
            )

        else:
            raise RuntimeError(
                (
                    "unsupported transport "
                    f"class: {transport_class}"
                )
            )


        enriched = add_columns(
            table,
            {
                (
                    "generator_weight_"
                    "transport_applicable"
                ): pa.array(
                    [True]
                    * row_count,
                    type=pa.bool_(),
                ),
                "population_kind": (
                    pa.array(
                        [
                            population_kind
                        ]
                        * row_count,
                        type=pa.string(),
                    )
                ),
                "transport_id": (
                    pa.array(
                        [
                            transport_id
                        ]
                        * row_count,
                        type=pa.string(),
                    )
                ),
                (
                    "generator_weight_"
                    "transport_class"
                ): pa.array(
                    [
                        transport_class
                    ]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "generator_weight_"
                    "provenance"
                ): pa.array(
                    [provenance]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "generator_nominal_weight"
                ): pa.array(
                    raw_weights,
                    type=pa.float64(),
                ),
                (
                    "generator_weight_sign"
                ): pa.array(
                    np.sign(
                        raw_weights
                    ).astype(
                        np.int8
                    ),
                    type=pa.int8(),
                ),
                (
                    "generator_weight_"
                    "join_status"
                ): pa.array(
                    [
                        validation_mode
                    ]
                    * row_count,
                    type=pa.string(),
                ),
                (
                    "physical_weight_"
                    "application_authorized"
                ): pa.array(
                    [False]
                    * row_count,
                    type=pa.bool_(),
                ),
            },
        )

        stats[
            "primary_sources"
        ] += 1

        stats[
            "primary_rows"
        ] += row_count

        stats[
            "raw_rows"
        ] += int(
            raw_available.sum()
        )


    path_a = (
        output_a
        / f"{canary_class}.parquet"
    )

    path_b = (
        output_b
        / f"{canary_class}.parquet"
    )

    write_table(
        enriched,
        path_a,
    )

    write_table(
        enriched,
        path_b,
    )

    req(
        sha256(
            path_a
        )
        == sha256(
            path_b
        ),
        (
            "byte idempotency failed: "
            f"{canary_class}"
        ),
    )

    stats[
        "byte_identical"
    ] += 1


    receipt_rows.append({
        "canary_class": (
            canary_class
        ),
        "source_uid": (
            source_uid
        ),
        "workflow_population": (
            workflow_population
        ),
        "population_kind": (
            "auxiliary_qcd"
            if is_auxiliary
            else population_kind
        ),
        "transport_id": (
            transport_id
        ),
        "transport_class": (
            transport_class
        ),
        "rows": (
            row_count
        ),
        "raw_weight_available_rows": int(
            raw_available.sum()
        ),
        "transport_overlap_rows": (
            overlap_rows
        ),
        "validation_mode": (
            validation_mode
        ),
        "output_path": str(
            path_a
        ),
        "output_sha256": (
            sha256(
                path_a
            )
        ),
        "byte_identical": True,
        "physical_weight_application_authorized": (
            False
        ),
    })


    print(
        "GENERATOR_WEIGHT_SOURCE_RESULT="
        f"class={canary_class};"
        f"transport_class={transport_class};"
        "status=pass;"
        f"rows={row_count};"
        f"overlap_rows={overlap_rows};"
        f"source_uid={source_uid}"
    )


req(
    (
        stats[
            "primary_sources"
        ]
        == 5
        and stats[
            "aux_sources"
        ]
        == 2
    ),
    (
        "source-role closure failed: "
        f"{stats}"
    ),
)

req(
    (
        stats[
            "primary_rows"
        ]
        == 2560
        and stats[
            "aux_rows"
        ]
        == 1024
    ),
    (
        "row-role closure failed: "
        f"{stats}"
    ),
)

req(
    (
        stats[
            "raw_rows"
        ]
        == 2560
        and stats[
            "byte_identical"
        ]
        == 7
    ),
    (
        "weight/idempotency closure "
        f"failed: {stats}"
    ),
)


pd.DataFrame(
    receipt_rows
).to_csv(
    args.output_dir
    / (
        "seven_class_generator_"
        "weight_join_receipts.tsv"
    ),
    sep="\t",
    index=False,
)


(
    args.output_dir
    / "runtime_versions.json"
).write_text(
    json.dumps(
        {
            "python": (
                sys.version
            ),
            "python_executable": (
                sys.executable
            ),
            "platform": (
                platform.platform()
            ),
            "numpy": (
                np.__version__
            ),
            "pandas": (
                pd.__version__
            ),
            "pyarrow": (
                pa.__version__
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


summary = {
    "status": "pass",
    "current_head": (
        args.current_head
    ),
    (
        "primary_sources_"
        "generator_weight_joined"
    ): stats[
        "primary_sources"
    ],
    (
        "auxiliary_qcd_sources_"
        "excluded_from_physical_transport"
    ): stats[
        "aux_sources"
    ],
    (
        "primary_rows_"
        "generator_weight_joined"
    ): stats[
        "primary_rows"
    ],
    (
        "auxiliary_qcd_rows_"
        "excluded_from_physical_transport"
    ): stats[
        "aux_rows"
    ],
    (
        "primary_rows_with_"
        "raw_root_weight"
    ): stats[
        "raw_rows"
    ],
    (
        "uniform_constant_"
        "sources_validated"
    ): stats[
        "uniform"
    ],
    (
        "signed_sidecar_"
        "sources_validated"
    ): stats[
        "signed"
    ],
    (
        "qcd_variable_fragment_"
        "sources_validated"
    ): stats[
        "qcd_variable"
    ],
    (
        "qcd_variable_candidate_"
        "overlap_rows"
    ): stats[
        "qcd_overlap"
    ],
    "byte_identical_sources": (
        stats[
            "byte_identical"
        ]
    ),
    "generator_weight_transport_joined": (
        True
    ),
    "physical_weights_calculated": 0,
    "physical_yields_calculated": 0,
    "validation_payloads_opened": 0,
    "test_payloads_opened": 0,
    "models_trained": 0,
    "repository_modified": False,
    "result": (
        "SEVEN_CLASS_TRAIN_GENERATOR_"
        "WEIGHT_JOIN_CANARY_V1_PASS"
    ),
    "next": (
        "FREEZE_BROAD_FEATURE_EXTRACTOR_"
        "SCHEMA_RUNTIME_AND_WEIGHT_POLICY"
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
    "SEVEN_CLASS_WEIGHT_POLICY_SOURCES=7/7"
)

print(
    "PRIMARY_GENERATOR_WEIGHT_SOURCES_JOINED="
    f"{stats['primary_sources']}/5"
)

print(
    "AUXILIARY_QCD_SOURCES_EXCLUDED_"
    "FROM_PHYSICAL_TRANSPORT="
    f"{stats['aux_sources']}/2"
)

print(
    "PRIMARY_GENERATOR_WEIGHT_ROWS_JOINED="
    f"{stats['primary_rows']}/2560"
)

print(
    "PRIMARY_RAW_ROOT_WEIGHT_ROWS="
    f"{stats['raw_rows']}/2560"
)

print(
    "UNIFORM_CONSTANT_SOURCES_VALIDATED="
    f"{stats['uniform']}"
)

print(
    "SIGNED_SIDECAR_SOURCES_VALIDATED="
    f"{stats['signed']}"
)

print(
    "QCD_VARIABLE_FRAGMENT_SOURCES_VALIDATED="
    f"{stats['qcd_variable']}"
)

print(
    "QCD_VARIABLE_CANDIDATE_OVERLAP_ROWS="
    f"{stats['qcd_overlap']}"
)

print(
    "PARQUET_BYTE_IDENTICAL_SOURCES="
    f"{stats['byte_identical']}/7"
)

print(
    "GENERATOR_WEIGHT_TRANSPORT_JOINED=YES"
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
    "RESULT="
    "SEVEN_CLASS_TRAIN_GENERATOR_"
    "WEIGHT_JOIN_CANARY_V1_PASS"
)

print(
    "NEXT="
    "FREEZE_BROAD_FEATURE_EXTRACTOR_"
    "SCHEMA_RUNTIME_AND_WEIGHT_POLICY"
)
