# c7v errata checkpoint provenance

The authoritative immutable checkpoint is:

`/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7v_jhep_results_and_figures_20260803_v4`

Its `SHA256SUMS` file has SHA-256:

`40228a4ad86337d1a505c6ec64611ecf0a84c3f1c5a49adc73ff5f50bf776676`

The persistent mechanical paper export is:

`/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/pn_c7v_jhep_results_and_figures_20260803_v4`

The generating errata script has SHA-256 `ac7d782fac6a207c5a9314cc4cb2dccd47e99c36c6a88aa916c62c4310a6f047`. Generation used base HEAD `b684bc4436252bfd789bcb64c3d2cacfa23a075f` and the frozen c7q--c7v-v2 inputs recorded in the checkpoint evidence manifests.

All 233 members listed by `SHA256SUMS` passed independent verification. Checkpoint files are mode `0444` and directories are mode `0555`. The run contract records zero validation payloads, zero test/evaluation payloads, and zero observed-data payloads opened.

All primary frozen-model performance intervals use one 1000-replica c7t-compatible source-member registry with seed `20260802`. Its SHA-256 is:

`37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29`

The compact summaries and registry provenance are included in this Git paper package. The 1000-row registry and replica-level Parquet tables remain only in the external immutable checkpoint.

Version 4 supersedes c7v v2 without modifying it. It corrects the systematics-aware scan cancellation and support display, selected-background composition, physics labels, LaTeX notation, and uncertainty presentation. Version 3 is retained unchanged as an intermediate complete checkpoint; v4 supersedes it because the v3 standalone paper export omitted the requested-figure registry and visual-inspection report named by the package index.
