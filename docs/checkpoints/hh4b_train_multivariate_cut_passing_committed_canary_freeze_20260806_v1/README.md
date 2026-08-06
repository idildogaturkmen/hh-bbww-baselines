# Passing committed multivariate-cut canary freeze

This checkpoint freezes the successful repository-native multivariate-cut
canary and the provenance-hash adjudication that followed it.

Six sentinel category/structure configurations were run twice with the
canary-only 7/16/4/2 search budget on the immutable 26,936-row canary fold
tables. Both complete passes were deterministic. Exact3tag and ge4tag passed
the canary-only absolute OOF tolerance.

The current execution and the earlier reduced-search reference have
byte-identical inner-crossfit TSVs and identical scientific result payloads.
Their raw canonical hashes differ only because repository_head is deliberately
included as execution provenance.

This is not a physics result and does not select a family or a cut. The
production contract remains 63 quantiles, beam width 128, 16 refinement
starts, 8 coordinate passes, target signal efficiency 0.585957, and zero
production tolerance.

This checkpoint authorizes preparation of a bounded full-settings HTCondor
pilot. It does not authorize the complete 270-job submission. The full scan
may be authorized only after the pilot passes its formal audit. Validation and
test remain sealed.
