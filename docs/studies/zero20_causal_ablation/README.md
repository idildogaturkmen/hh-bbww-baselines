# ZERO20 causal ablation (planned)

## Question

Is the harm found in the ParT active20 study (see
[`../part_representation_study/`](../part_representation_study/README.md)) caused
by widening SPA-Net's input embedding itself — an optimization/capacity effect
independent of what information the 20 extra dimensions actually carry — or does it
depend specifically on the ParT-derived content of those dimensions?

## Method (planned, not yet executed)

Train the identical SPA-Net architecture with 20 **zeroed** (uninformative) extra
input dimensions in place of the ParT-derived ones, holding everything else
constant — training data, architecture, optimization protocol, and evaluation
methodology — and compare against both the native-2M result and the ParT-active20
result under the same permutation-safe matched-comparison protocol. This isolates
"wider input embedding" as a variable from "ParT features specifically."

## Dataset

The same matched 2M training cohort / 400k evaluation cohort used throughout the
SPA-Net, training-scale, and ParT-representation studies.

## Main result

**Not yet available.** This is the designed next diagnostic; it has not been run
as of this writing. This page exists so the question and its design are findable
before the result lands.

## Key figures/tables

None yet.

## Code/config pointers

None yet. The design is fully specified (data, architecture, and comparison
protocol to reuse) in the frozen document under Provenance below.

## Frozen artifact

None yet. When this ablation is run, its natural home is a new artifact bundle
alongside `artifacts/hh4b_spanet_part_20260911/` (e.g. `artifacts/hh4b/zero20/`),
following the same internal shape (`tables/`, `plots/`, `metrics/`, `training/`,
`diagnosis/`, `SHA256SUMS`) — not created in this pass.

## Provenance

- `artifacts/hh4b_spanet_part_20260911/diagnosis/FOLLOWUP_DESIGNS.md` — the ZERO20
  ablation design.
- `artifacts/hh4b_spanet_part_20260911/diagnosis/ROOT_CAUSE_DIAGNOSIS.md` — the
  diagnosis this ablation is designed to test.
