# Signal normalization issue, 2026-07-08

Signal audit results:

- ggF HH4b generated sample:
  - MG5 median cross section: about 0.994 fb
  - candidate efficiency: about 6.41%

- VBF HH4b generated sample:
  - MG5 median cross section: about 0.948 fb
  - candidate efficiency: about 6.94%

For comparison, using approximate 13 TeV SM HH cross sections:
- ggF HH inclusive: about 31.1 fb
- VBF HH inclusive: about 1.73 fb
- BR(H→bb)^2 ≈ 0.5824^2 ≈ 0.339

This gives approximate SM HH→4b cross sections:
- ggF HH→4b ≈ 10.55 fb
- VBF HH→4b ≈ 0.587 fb

Naive normalization scale factors relative to current MG5 sample cross sections:
- ggF: about 10.6
- VBF: about 0.62

These should not yet be interpreted as clean K-factors. The mismatch is asymmetric and may reflect differences in process definitions, model choice, generator-level cuts, decay handling, or center-of-mass energy. Before interpreting signal yields or significance estimates, the signal cards and normalization convention need to be checked with Harvey.
