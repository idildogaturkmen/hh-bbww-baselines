# Historical fixed RHH comparator freeze

This checkpoint freezes the unchanged historical comparator:

`r_hh_125_125 < 34.0`

No threshold scan or tuning was performed.

## Physical result at 138 fb^-1

- Signal yield: 417.46273293569186
- Background yield: 20240622849.775902
- Signal efficiency: 0.55308830588219915
- Background efficiency: 0.32491836209645031
- S/B: 2.0624994400323698e-08
- Statistical-only Asimov significance: 0.0029343085318647984
- Signal effective events: 31769.297520111777
- Background effective events: 18233.175821910761

The significance is a statistical-only benchmark, not a final nuisance-aware
physics result.

## Bootstrap

The 2,000-replica paired source-group bootstrap uses seed 20260727. The binary
draw-count Parquet is not copied into Git. Its exact external path, byte size,
and SHA-256 are frozen in `external_artifact_manifest.json`.

## Safety state

- Physical sources: 441
- Auxiliary QCD sources excluded: 23
- Validation payloads opened: 0
- Test payloads opened: 0
- Models trained: 0
- Common-table payload modified: false

## Next gate

Run the predeclared nested five-fold source-group out-of-fold optimized-cut
study. The fixed RHH34 result remains the historical comparator and must not be
retuned.
