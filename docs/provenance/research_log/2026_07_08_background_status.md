# Background production status, 2026-07-08

Completed storage-safe campaigns:
- QCD bbbb 100k inclusive
- QCD bbbb inclusive-HT slices, 2k/slice and 10k/slice
- Zbbbb 100k
- inclusive ttbar 100k
- ttbb 10k diagnostic

QCD importance sampling:
- ptb/ptbmax slicing was tested but does not appear to form a complete event-level phase-space partition.
- inclusive-HT slicing closes well against the inclusive QCD 100k reference.
- QCD bbbb HT-sliced 10k/slice is frozen as the current importance-sampling strategy.

Top backgrounds:
- inclusive ttbar 100k gives low selected 4-b-tag candidate efficiency.
- ttbb 10k gives much higher selected-candidate efficiency and is therefore useful as a top + heavy-flavor diagnostic.
- ttbb should not be blindly added to inclusive ttbar without a matching/veto prescription, because inclusive ttbar can already contain extra heavy flavor from the shower.

Next:
- Finish ttbb 50k.
- Compare ttbb 50k to inclusive ttbar and decide whether to use ttbb as a separate diagnostic component or implement a truth-level split later.
