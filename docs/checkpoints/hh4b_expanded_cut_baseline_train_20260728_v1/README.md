# Expanded HH4b train-only cut baseline

## Scope

This checkpoint freezes the tables and CMS-publication-inspired figures for the
expanded 5M-background/200k-signal source study. The plotted population is the
53,162-row train cache: 9,337 signal candidates and 43,825 background candidates.

The figures are explicitly labeled **Delphes simulation** and do not claim official
CMS status. All efficiencies use hierarchical development-balancing weights, not
cross-section or luminosity normalization.

## Frozen nominal result

The predeclared nominal selection is

\\[
R_{HH}^{125,120} < 34.
\\]

It selects 5,758 signal candidates and
9,971 background candidates, with

\\[
\\epsilon_S=0.615642,\qquad
\\epsilon_B=0.249696,\qquad
1/\\epsilon_B=4.005.
\\]

Its development-balanced proxy is
\\(\\epsilon_S/\\sqrt{\\epsilon_B}=1.232033\\).

## Working-point interpretation

- The CMS-reference point, \\(R_{HH}^{125,120}<30\\), has the strongest background
  rejection: \\(1/\\epsilon_B=5.015\\).
- The higher-purity point, \\(R_{HH}^{125,120}<31.5\\), raises the balanced proxy to
  1.237005 while retaining less background than the nominal point.
- The higher-efficiency point, \\(R_{HH}^{125,120}<35.5\\), has the largest balanced proxy among
  the four predeclared points, 1.243677.
- The nominal threshold remains 34 because the protocol was frozen before this result;
  post-result retuning is not authorized.

These proxies are controlled development diagnostics, not physical significances or
expected limits.

## Contents

- `figures/`: eight figures in paired PNG and PDF formats.
- `tables/`: TSV and LaTeX tables for working points, regions, signal modes, and background families.
- `source/`: frozen train-only source products copied for reproducibility.
- `figure_manifest.tsv` and `table_manifest.tsv`: checksums and file inventory.
- `protected_source_audit.tsv`: source provenance.

## Next gate

Implement and commit the reusable expanded BDT runner, then derive grouped train-only
out-of-fold scores and thresholds without opening validation.
