# c7v checkpoint provenance

The authoritative immutable checkpoint is:

`/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7v_jhep_results_and_figures_20260803_v2`

Its `SHA256SUMS` file has SHA-256:

`5c7daba43f7aa6cf2bbc1d91411a819f68cee482a49ba4c635724e3ff5c6174c`

The persistent mechanical paper export is:

`/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/pn_c7v_jhep_results_and_figures_20260803_v2`

The generating script has SHA-256 `ff6a6cd13becf4bf8ae21ce483e928ea72df3763624b36bf5ac0dc269ed371fb`. The checkpoint records generation base HEAD `3ecfdbef03a8acd1a4cc460642828c2e255afe53`; the reviewed script and paper export are included in the accompanying local c7v completion commit.

All 80 checkpoint files passed `SHA256SUMS`; files are mode `0444` and directories are mode `0555`. The 73 source-evidence members are confined to sealed c7q--c7u train artifacts. The run contract records zero validation payloads and zero test/evaluation payloads opened.

The completed `pn_c7v_jhep_results_and_figures_20260802_v1` checkpoint is retained unchanged but is superseded. Visual review of v1 exposed truncated high-tail display ranges and legend placement issues. Version 2 uses full-range logarithmic binning where required, documents the degenerate promoted-jet b-tag input, and passed the final render review recorded in `figure_inspection_report.tsv`.
