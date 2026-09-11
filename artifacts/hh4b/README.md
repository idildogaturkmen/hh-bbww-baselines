# `artifacts/hh4b/` — forward convention for SPA-Net / ParT / representation-study bundles

This directory is the **forward home** for lightweight, citable result bundles
in the SPA-Net / ParT / representation-learning line of work, adopted starting
with the diagnostic that follows the 2026-09-11 snapshot.

## Relationship to the existing frozen bundle

`../hh4b_spanet_part_20260911/` is the authoritative, hash-stamped (`SHA256SUMS`)
record of:

- the native SPA-Net 2M-vs-10M scaling result, and
- the frozen-ParT "active20" harm result and its root-cause diagnosis,

as of 2026-09-11. **That bundle is not reorganized or split by this
convention and is never modified.** Splitting its internal layout after the
fact would invalidate the checksums that make it a trustworthy frozen record.
If you are looking for the 2026-09-11 native-scaling or ParT-active20 numbers,
go there directly.

## Convention adopted here, going forward

Each new experiment in this line of work gets its own subfolder, following the
same internal shape already established by the 09-11 bundle
(`metrics/`, `tables/`, `plots/`, `training/`, `diagnosis/`, `SHA256SUMS`):

```
artifacts/hh4b/
├── native_scaling/    # future refreshes of the 2M-vs-10M native SPA-Net scaling result
├── part_active20/     # future refreshes of the frozen-ParT active20 result
└── zero20/            # the ZERO20 ablation (zeroed, not ParT-derived, extra input
                        # dimensions) — isolates "wider input embedding" from
                        # "ParT features specifically" as the cause of the
                        # active20 harm result. Currently the active diagnostic;
                        # see docs/results/RESULTS_OVERVIEW.md for status.
```

Subfolders are created as their corresponding result is actually produced and
frozen — this directory intentionally starts empty of data beyond this
README. A future refresh of native-scaling or part-active20 results is
cross-linked from the frozen 09-11 bundle rather than replacing it there.

This is documentation/convention only — no existing files were moved or
modified to create this directory.
