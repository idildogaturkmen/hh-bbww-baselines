# HH4b ttbar27 canonical-72 recovery

Status: `hh4b_ttbar27_exact_regeneration_required`

Next gate: `prepare_hh4b_ttbar_missing_member_exact_regeneration`

The 18 pre-existing canonical-72 products passed full schema, content, identity,
row-count, and checksum validation.  The first retained-ROOT builder attempt
stalled with no output and is recorded as `aborted_stalled_no_output`.  A
controlled retry on a checksum-identical node-local ROOT copy reconstructed 29
rows with the protected builder and policy.  It passed an isolated deterministic
rerun comparison of schema, event order, integer/string values, nonfinite masks,
and floating values at the frozen tolerance.  The eight original
ROOT/HepMC/LHE/archive sources remain absent after local, provenance, archive,
EOS, and XRootD searches.

Exact deterministic regeneration is possible from the retained production
contract, but production was not started.  The next gate must first reproduce
seed 105000 and compare it with the retained original canary, then regenerate
the eight exact seeds only if that canary matches.

The 307 legacy candidate rows for those eight members are excluded from the
candidate registry.  No missing canonical features were synthesized.  No
sealed candidate content, predictions, models, scores, significance, limits,
or physical yields were opened or produced.

All tabular outputs are emitted as TSV, Markdown, and booktabs-compatible
LaTeX.  Event-level recovery products remain under ignored runtime output.
