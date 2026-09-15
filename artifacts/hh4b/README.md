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

Each new experiment in this line of work gets its own subfolder under
`pretrained_jet_representations/`, following the same internal shape already
established by the 09-11 bundle, plus an added `provenance/` and `code/`
(`metrics/`, `tables/`, `plots/`, `provenance/`, `code/`):

```
artifacts/hh4b/
└── pretrained_jet_representations/
    └── zero20_20260914/   # ZERO20 width-control ablation, FINAL result (2026-09-14).
                            # See its own README.md. Causal branch: B (falsifies the
                            # pure-width explanation for the active20 harm result).
```

**Supersedes the layout originally anticipated below**, which guessed a
flatter `artifacts/hh4b/zero20/` path before any ZERO20 result existed. The
convention actually adopted groups all representation-learning bundles
(ParT active20 refreshes, ZERO20, and any future frozen-embedding study)
under one `pretrained_jet_representations/` umbrella — the same name used
for the narrative deep-dive at
`docs/studies/07_pretrained_jet_representations/`, so the code-facing and
docs-facing trees mirror each other. Future refreshes of native-scaling or
part-active20 results, and any subsequent representation study, follow
this same nested shape.

*(Original anticipated layout, kept for history — not what was actually used):*

```
artifacts/hh4b/
├── native_scaling/    # future refreshes of the 2M-vs-10M native SPA-Net scaling result
├── part_active20/     # future refreshes of the frozen-ParT active20 result
└── zero20/            # superseded -- see pretrained_jet_representations/zero20_20260914/ above
```

Subfolders are created as their corresponding result is actually produced and
frozen. A future refresh of native-scaling or part-active20 results is
cross-linked from the frozen 09-11 bundle rather than replacing it there.

This is documentation/convention only — no existing files were moved or
modified to create this directory.
