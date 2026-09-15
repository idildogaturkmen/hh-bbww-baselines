# HH4b CMS-gap QCD tail audit v2

## Computed facts

The authoritative 464-source train manifest contains 186 physical `qcd_hardqcd` sources and 23 auxiliary QCD classification sources. It contains **zero ordinary physical QCD sources**. The auxiliary sources intentionally have no physical coefficient and do not enter the likelihood. Therefore an ordinary-QCD versus importance-QCD physical closure test is not statistically or normalization-defined with current products.

All 186 physical QCD sources use the frozen hard-QCD importance/stitching contract. Each source is joined to its authoritative campaign, pTHat interval, generator cross section, production-mixture fraction, shard sum-of-generator-weights denominator, and Run-2 coefficient in `qcd_source_inventory.tsv`. Their selected closure is 123 rows and 46,452,891.0144 expected events, exactly reproducing the frozen QCD total within floating-point precision.

The fixed eight-bin model is fully tabulated in `artifacts/hh4b_cms_sensitivity_gap_v2/qcd_tail_bin_support.tsv`. The bins driving the nominal 44.33→31.96 stat-only improvement are:

- ge4tag `[10,20)`: 38.55% of small-signal shape information; only two physical-QCD rows, QCD Neff=1.023, and the largest source/event supplies 83.63% of total background. Unsupported.
- ge4tag `[30,34)`: 36.38% of information; zero physical-QCD rows. The non-QCD estimate has Neff=215, but absence of simulated QCD is not evidence that QCD is absent. QCD-unconstrained and unsupported.
- ge4tag `[20,30)`: 9.22%; two QCD rows, QCD Neff=1.006, largest source/event 93.44%. Unsupported.

All eight bins are marked unsupported or limited for direct-QCD inference. The best-supported QCD bin, exact3tag `[20,30)`, still has only 50 QCD rows and Neff=23.03.

## Finite-MC robustness

The frozen independent-Gaussian diagnostic remains μ95=545.70 for the shape model. An independent positive-bin Poisson-equivalent gamma/effective-count constraint gives μ95=522.06. All background bins are positive and the effective-count construction was not applied to an invalid signed/negative bin. The numerical values are model diagnostics, not sensitivity projections. Their agreement in scale establishes the robust qualitative result: direct weighted QCD has inadequate tail support.

## Lower-tag feasibility and prototype

The provenance audit found that a new extraction is unnecessary. The already-frozen common train production contains 441 physically authorized source tables and 1,042,397 resolved rows with `N_b=0,1,2,3,≥4`, source/fold/event identity, signed weights, and the required reconstruction features. A deterministic five-source canary across ggHH, hard QCD, ttbar, Z+bbbb, and ttH closes exact3 and ge4 row counts, signed yields, and sumw2 to the fixed frozen fold tables in all 10 checks.

A predeclared simulation-only hard-QCD transfer used source folds 0–2 for development and folds 3–4 for closure. Fixed transfer cells used RHH, mHH, and candidate HT. The 2b→3b yield closes at 1.0054 predicted/target; distribution total-variation distances range 0.025–0.146. The 2b→ge4 prediction closes poorly at 0.2943, with only 24 target closure rows and RHH distance 0.484. Thus lower-tag transfer is technically feasible, but the ≥4-tag transfer is not reliable with the current physical QCD sample and simple predeclared model.

No new extraction package, resource pilot, or 441-job campaign is required or scientifically justified. No Condor submission was made.
