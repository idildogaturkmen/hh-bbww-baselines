# HH4b train candidate event-multiplicity preflight

This checkpoint opens only the 441 frozen train candidate Parquet files and
the 14 frozen ordinary signed generator-weight sidecars needed for exact
train-event joins. Candidate paths under `/store` are resolved through the
local `/eos/uscms` mount when available, otherwise copied one at a time with
XRootD into node-local `/tmp`, checksum-verified, inspected, and deleted.

It proves:

- 255 ordinary plus 186 hard-QCD train transports are represented once;
- 30649 ordinary plus 56 hard-QCD candidate rows close to 30705 rows;
- every nonempty candidate file contains the exact `event` key;
- zero-row candidate files may omit `event` or have an empty schema because
  there is no selected source event to join;
- every candidate event lies within its source generated-event range;
- `(population_kind, transport_id, event)` is globally unique;
- every selected source event maps to exactly one candidate row;
- all 14 ordinary signed joins and all three variable-QCD candidate joins
  are exact;
- direct source-event physical-weight application is therefore safe for
  train candidates without multiplicity division.

No candidate physical weight or selected yield is calculated in this step.
Validation and final-test payloads remain sealed.

Source commit: `adbf749ae4603402ecf8aa9593a1c0bcf8c0c067`
