# Seven-class broad-extractor access canary

This checkpoint freezes the successful seven-class train-only access and
basic ROOT-schema canary at source commit `308a5e10350619617015b136c8e0487d2e4bd3d0`.

## Verified

- 7/7 representative train sources passed.
- The common resolver handled direct ROOT access, ordinary bundle layouts,
  and the nested ggF reconstruction-bundle layout.
- ROOT entry counts matched the frozen generated-event counts.
- Rich AK4 jet branches were readable in all seven classes.
- Signal truth branches were readable for both ggF and VBF.
- The broad selection `Jet.PT > 30 GeV`, `abs(Jet.Eta) < 2.5`, and at least
  four selected AK4 jets was iterated successfully.
- Validation and test payloads remained sealed, and no model was trained.

## Important limitation

This is an access and basic-schema checkpoint, not yet a production feature
extractor. Before launching 464 train-source jobs, the reusable extractor must
materialize and read back the frozen ML output schema, event identities,
generator-weight joins, truth-matching labels, and deterministic jet
truncation/order rules on the same seven representative classes.

## Next gate

Run the seven-class feature-materialization canary. Only after that passes
should the 464-source train extraction be prepared.
