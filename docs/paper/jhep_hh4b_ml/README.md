# HH to four-b train-only paper assets

This is a Delphes-based simulation study evaluated only with train-source-group OOF predictions. No validation/test result, observed data, or CMS approval is claimed. Direct $\geq4b$ QCD is secondary closure only; the primary multijet prediction is the frozen lower-b-tag transfer, whose uncertainty currently dominates the sensitivity.

All tables are machine generated from checksum-verified c7q--c7u inputs. Figure PDFs are authoritative; PNGs are review companions. Exact plotting inputs are in `source_data/`, caption fragments are in `captions/`, and `compile_fragments.tex` is a lightweight syntax wrapper, not a JHEP template.

This export supersedes c7v v2 for support diagnostics, labels, and uncertainty presentation. All primary frozen-model comparisons use the common paired source-member registry; its SHA-256 is `37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29`.

The reviewed Phase-1 train-only nested categorized-BDT package is in `categorized_bdt/`. It uses the same common 1000-replica source-member registry, provides asymmetric intervals for all supported primary and per-category metrics, and records paired difference intervals for inclusive, selected categorized, fixed-five-category, category-count-scan, and mass-aware versus mass-plane-blind comparisons.
