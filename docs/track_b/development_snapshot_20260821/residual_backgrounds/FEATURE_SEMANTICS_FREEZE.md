# Feature Semantics Freeze (Task B)

**DEVELOPMENT ONLY — matched 400k training-production cohort; not independent Stage-C; not physically normalized.**

Read directly from the frozen source
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_harvey_bdt_working_points_20260818_v1/step2_training/scripts/feature_extraction.py`
(quoted verbatim below, not reconstructed from memory) on 2026-08-21 for this phase.

## Candidate-jet definition (verbatim from source docstring)

```
selected = (jet_pt > 30 GeV) & (abs(jet_eta) < 2.5)
-> sort selected jets by descending pT
-> truncate to first N_SLOTS (10)
-> zero-pad
-> derive mask (True for real slots, False for padding)
```
`n_selected_jets` = number of jets passing the pT/eta cut **before** truncation to
N_SLOTS (can exceed 10).

## Exact 82-column layout (verbatim from source comment, `feature_extraction.py:28-44`)

| Columns | Field | Notes |
|---|---|---|
| `[0:10)` | `jet_pt` | 10 slots, pT-sorted descending |
| `[10:20)` | `jet_eta` | same event/slot order as pt |
| `[20:30)` | `jet_phi` | same event/slot order as pt |
| `[30:40)` | `jet_mass` | same event/slot order as pt |
| `[40:50)` | `mask` | 1.0 = real jet, 0.0 = zero-padded slot |
| `[50]` | `n_selected_jets` | scalar, pre-truncation count |
| `[51]` | `HT` | scalar, event-level |
| `[52:62)` | `probB` | Sophon AK4 b-tag probability, same slot order |
| `[62:72)` | `probC` | Sophon AK4 c-tag probability, same slot order |
| `[72:82)` | `probL` | Sophon AK4 light-tag probability, same slot order |

`K_WIDTH = 52` (columns `[0:52)`), `KF_WIDTH = 82` (columns `[0:82)` — K plus probB/C/L).

**Sort order proof**: `extract_padded_features()` computes
`order = ak.argsort(pt, axis=1, ascending=False)` once, then applies that *same*
`order` to `pt, eta, phi, mass, probB, probC, probL` (`feature_extraction.py:60-62`)
— so slot index `i` in every one of the seven per-jet fields refers to the *same
physical jet* (the (i+1)-th highest-pT selected jet in that event), for every
event. This is what makes it valid to read off "leading jet" as slot 0, "second
jet" as slot 1, etc. across all seven fields consistently.

**Padding proof**: `mask[:, i] = 1` iff slot `i < min(n_selected_jets, 10)`
(`feature_extraction.py:71`); all other fields are zero-filled beyond that point
(`ak.fill_none(..., 0.0)`, line 67). Any per-jet observable computed on a masked
(padding) slot is meaningless and must be excluded — this is enforced explicitly
in every Task C/D computation in this phase via the mask.

## What this phase reads from the matched-400k sidecar (`X`, shape `(400000, 82)`)

Leading four jets used for the four-jet HH topology (Task C) are columns:
- jet0: `pt[0], eta[10], phi[20], mass[30]`, mask `[40]`
- jet1: `pt[1], eta[11], phi[21], mass[31]`, mask `[41]`
- jet2: `pt[2], eta[12], phi[22], mass[32]`, mask `[42]`
- jet3: `pt[3], eta[13], phi[23], mass[33]`, mask `[43]`

An event is included in any four-jet topology observable **only if**
`mask[40]==mask[41]==mask[42]==mask[43]==1` (i.e. `n_selected_jets >= 4`) —
events with fewer than 4 real candidate jets are excluded from four-jet
topology observables and counted separately as "excluded (fewer than 4 real
jet slots)".
