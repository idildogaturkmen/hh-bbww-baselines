# HH4b physical-normalization provenance policy

Physical event weights are kept separate from model-training weights.

For a process \(p\), the eventual nominal event weight will have the form

\[
w_{i,p} =
\mathcal{L}\,
\sigma_p^{\mathrm{ref}}\,
k_p\,
\epsilon_{\mathrm{filter},p}\,
\mathcal{B}_p\,
\frac{w_{i,p}^{\mathrm{gen}}}
{\sum_{j\in p} w_{j,p}^{\mathrm{gen}}},
\]

with additional importance-sampling, stitching, and detector-correction factors only
when their provenance is explicitly documented.

No process may receive a physical weight until the following are resolved:

1. exact generator process and decay definition;
2. beam energy, generator versions, PDF, tune, and perturbative order;
3. generator cross section and process-matched reference cross section;
4. branching-fraction and filter-efficiency convention;
5. sum of signed and absolute generator weights;
6. importance-sampling correction, when applicable;
7. overlap-removal and stitching policy, when applicable.

The dominant QCD multijet background is assigned a lower-b-tag control-region strategy
as the intended primary result. Directly normalized QCD simulation is retained only as
a secondary projection after overlap and stitching closure.

The first provenance checkpoint is inventory-only. It opens no candidate parquet,
assigns no cross section, and calculates no physical yield or threshold.
