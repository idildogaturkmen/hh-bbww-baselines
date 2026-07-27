# HH4b ttbar8 exact-regeneration contract

Status: `hh4b_ttbar8_exact_regeneration_contract_frozen`

Next gate: `submit_hh4b_ttbar8_exact_regeneration_canary`

This checkpoint freezes the complete provenance and production contract for the
eight missing original `ttbar_100k` source members.  It describes one
deterministically selected canary (ttbar_100k_shard003, seed
105003) and a seven-member scaleout.  Both HTCondor descriptions are
safety-locked with `requirements = False` and `hold = True`; no jobs were
submitted in this gate.

The original production was not containerized, so the environment is classified
as `validated_equivalent_environment`, not as an available exact original
environment.  No regenerated output may be called exact until the canary passes
the frozen equivalence and exactness checks.  Byte-identical ROOT output is not
claimed.

The 307 legacy 15-column rows are validation-only.  They are not canonical-v2
inputs; only newly reconstructed, validated canonical-72 rows may enter v2.
No sealed candidate content, model training, predictions, yields, significance,
or limits were accessed or produced.
