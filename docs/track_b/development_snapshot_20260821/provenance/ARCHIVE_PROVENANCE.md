# Archive provenance and exclusions

The archive was created from Git commit
`c8f5ba4b5ca1ff3238d14690619109793b9d7f90` on
`origin/delphes-hh4b-production`. All source trees were read only.

Original source-manifest SHA256 values:

| Component | Original `SHA256SUMS` SHA256 |
|---|---|
| model comparison | `fb570116ea9e80f2ecdabeeded496304102f27f8c5d9caaf9aa3ffa176072b13` |
| residual backgrounds | `e8b3b9fb2efae664b2749ef644f2781f2697cc6e14a1903a90091c6640f3cb8a` |
| BDT convergence | `c21ab11b9e227940004e0e9e0e541e75cddb752333fb0b1085de1ba3736d2e39` |
| SPA-Net 10M | `ffde971c5ec95cd0eb32a66ce0e05bd1013bc1d9409c288eb865a65c4c456c3d` |

Deliberately omitted artifacts include:

- Source B `results/topology_full.npz` (67,207,120 bytes) and
  `results/scored_matched_400k.npz` (146,646,110 bytes).
- Source B's six topology SVG renderings larger than 20 MB; equivalent PDF
  and PNG renderings are present in the archive.
- Source D HDF5 files, checkpoints, TensorBoard events, training logs, and
  `/tmp` staging. The EOS 10M training HDF5 is
  `/eos/uscms/store/user/iturkmen/spanet_10M_scaling/production_hdf5_build_10M_v1/work/production_10M_train.h5`,
  SHA256 `2186ff1221eb7d42b26852167c29d61b91a646eaf4494314e5d55d427cd97790`.
- All trained BDT/XGBoost models, Condor payloads, raw ROOT event files,
  virtual environments, credentials, private correspondence, and temporary
  logs.

Component-local copies of `SOURCE_PROVENANCE.tsv` and `SHA256SUMS` preserve
the detailed original-path-to-hash mapping. The archived files are copied
byte-for-byte; this top-level documentation is the only newly authored
content.
