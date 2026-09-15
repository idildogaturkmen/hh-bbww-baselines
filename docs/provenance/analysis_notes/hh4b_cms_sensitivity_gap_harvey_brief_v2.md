# HH→4b CMS sensitivity-gap update for Harvey v2

Ours is a train-only **Delphes simulation expected-limit diagnostic**. CMS HIG-24-010 uses full detector reconstruction, collision data, a data-driven multijet estimate, control/validation regions, and a multibin classifier likelihood. It is the closest benchmark, not an equivalent analysis.

## Computed facts

- CMS Run-2 resolved expected μHH95: **5.9**.
- Historical pooled `R_HH<34`: **74.02**, or **12.54×** CMS.
- Preserve exact3tag and ge4tag: **44.33**, gain **×1.670**.
- Add predeclared fixed RHH shapes: stat-only **31.96**, further gain **×1.387**; total information-recovery factor **×2.316**.
- Independent `1.96/Z` checks reproduce 74.01 and 44.31, closing the implementation.
- Selected physical QCD: **46,452,891.014**, **95.48%** of B, supported by **123 rows**, QCD Neff **52.40**.
- There are 186 importance-sampled physical QCD sources, 23 auxiliary/nonphysical QCD sources, and **no ordinary physical QCD source** in the authorized train manifest.
- About 75% of nominal shape information comes from ge4tag `[10,20)` (two QCD rows, Neff≈1.02) and `[30,34)` (zero QCD rows). These bins are unsupported for direct-QCD inference.
- Gaussian and gamma/effective-count finite-MC diagnostics give 545.7 and 522.1. Neither is an analysis sensitivity; both diagnose inadequate tail support.

## Diagnostic interpretations

The category and shape gains show that the historical count discarded useful information, but the apparent shape sensitivity cannot yet be claimed because its dominant bins have essentially no QCD support. The three largest quantified bottlenecks are: the 46.5-million-event QCD projection and extremely small S/B; QCD tail Neff/source-event concentration; and lack of a physically normalized lower-tag population for a transfer-based background constraint.

Importance-versus-ordinary physical closure is not testable: ordinary physical QCD is absent, while auxiliary QCD has no authorized physical normalization. Comparing their normalized yields would manufacture a closure test.

A CMS-inspired simulation-only lower-tag transfer is technically feasible from existing products; no new extraction is needed. The already-frozen 441 physical tables contain 1,042,397 resolved rows across all tag counts, and a five-source canary closes exact3/ge4 rows, yields, and sumw2 in all 10 checks. A predeclared source-fold split gives 2b→3b yield closure of 1.005 with moderate shape distances, but 2b→ge4 underpredicts by 71% (ratio 0.294) and has only 24 target closure rows. The ≥4-tag transfer is therefore not reliable yet. No campaign was submitted.

## Unresolved hypotheses

- The pending independent global BDT will test how much class separation exists when category/shape information is used more efficiently; no unfinished output is imported here.
- B-tag migration distributions and transfer closure can now be quantified from the frozen lower-tag tables; flavor-efficiency and pairing/mass-resolution ceilings still require a dedicated truth-level analysis.
- CMS-specific collision-data transfer constraints, detector calibrations, triggers, and nuisance correlations remain irreducible in Delphes.

`CMS_GAP_VALIDATION_PAYLOADS_OPENED=0`; `CMS_GAP_TEST_PAYLOADS_OPENED=0`.
