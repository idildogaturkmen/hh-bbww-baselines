# Unified 5M Wave-B importance-sampling policy

## Goal

Increase the number of physically relevant HH->4b-like background
events without sculpting the reconstructed observables used by the
analysis.

## Allowed enrichment mechanisms

1. Force physical H->bb or Z->bb decays and restore the corresponding
   branching fraction in the normalization.
2. Generate explicit additional b quarks at matrix-element level using
   broad acceptance cuts:
   - b-quark pT > 20 GeV
   - |eta(b)| < 2.7
   - deltaR(b,b) > 0.4
3. Use broad, exclusive HT or boson-pT strata only after the 10k pilots
   demonstrate a statistically useful gain.
4. Keep every enrichment stratum in a separate manifest and retain its
   generator cross section and total generated-event denominator.

## Prohibited enrichment mechanisms

Do not select on reconstructed or truth-proxy analysis variables such as:

- reconstructed m_bb near 125 GeV
- reconstructed m_HH
- r_HH
- classifier score
- fitted Higgs mass
- a signal-region selection copied from the final analysis

Such cuts would sculpt the analysis distributions.

## Overlap policy

- Inclusive and heavy-flavor-enriched samples are not directly additive.
- tchannel_bb_hf remains training-only until a truth-level stitching
  prescription with inclusive t-channel production is validated.
- Future tW+bb enrichment requires an explicit tW/ttbar interference
  and overlap prescription.
- ttH belongs to rare-top, not generic single-Higgs.
- dedicated ZH4b and ZZ4b remain separate from generic VH and diboson.
- WWbb is treated as an explicit exclusive heavy-flavor process.
- enrichment samples may be balanced for training but require
  stratum-aware normalization for physical yields.

## Weighting policy

For each physical stratum:

    event weight =
        reference or exclusive generator cross section
        x branching fraction
        x filter efficiency
        x luminosity
        / total generated events in that stratum

For an ML-enrichment stratum, final physical weights remain unauthorized
until overlap removal or stitching is frozen.

## Pilot decision rule

After each 10k pilot, record:

- generated cross section and its integration uncertainty
- reconstructed candidate efficiency
- number of candidate events
- effective weighted yield in the target phase space
- maximum event-weight fraction
- overlap category
- whether an additional HT or boson-pT stratum is justified

Do not scale a family solely because it has a high raw candidate count.
