# HH4b SPA-Net recovery and next steps

## Current status

The original NRP PVC became temporarily inaccessible because of a Ceph/Rook CSI mount failure. Since some HH4b SPA-Net work was not pushed before the PVC issue, the main missing scripts were reconstructed in a temporary pod and pushed to the branch:

`recover-hh4b-spanet`

Recovered files:

- `scripts/hh4b_spanet/build_hh4b_resolved_spanet_dataset.py`
- `scripts/hh4b_spanet/diagnose_hh4b_genpart_truth.py`
- `notes/hh4b_spanet_recovery_status.md`

The existing resolved HH4b baseline work is already available on `pivot-channel-scouting`, including:

- resolved HH4b feature builder
- cut baseline
- BDT v2
- BDT stability resampling
- dense DNN v1
- LBN-inspired DNN v2
- model comparison summary script

## Previous HH4b SPA-Net dataset result

The recovered dataset builder previously produced the following result on the real NRP PVC data:

| Quantity | Value |
|---|---:|
| Total HH_4b events scanned | 19,984 |
| Events with two H to bb candidates | 1,606 |
| Events with at least four selected AK4 jets | 17,603 |
| Fully matchable events in selected top jets | 172 |
| Train / validation / test events | 120 / 25 / 27 |
| Baseline full-event pairing accuracy | about 0.209 |
| Baseline per-Higgs pairing accuracy | about 0.308 |

## Interpretation

The resolved HH4b SPA-Net dataset is not ready for serious SPA-Net training yet.

The fully matchable sample size is too small. With only 172 fully matchable events, a SPA-Net model would likely be statistically unstable and not scientifically meaningful.

The most likely next issue to debug is the truth-label definition, especially the GenPart Higgs and b-quark ancestry structure.

## Next steps when the PVC or data becomes available

1. Run the GenPart truth diagnostic:

    python scripts/hh4b_spanet/diagnose_hh4b_genpart_truth.py \
      --input-glob "outputs/collide_selected_backgrounds/HH_4b/*.parquet" \
      --outdir outputs/hh4b_spanet/genpart_truth_diagnostic \
      --max-events 2000

2. Check whether the low count comes from:

    - too few Higgs particles with descendant b quarks
    - Higgs status-copy structure
    - b quarks whose mother chain contains a Higgs but are missed by descendant tracing
    - reco-jet acceptance
    - jet ranking
    - delta-R matching

3. If truth tracing is the problem, revise the dataset builder to use a more robust Higgs-ancestor method.

4. Only train SPA-Net if the fully matchable event count becomes large enough for a stable train/validation/test split.

## Current scientific conclusion

The resolved HH4b baseline comparison is still valid:

- Cut baseline is weaker than the ML baselines.
- Dense DNN and LBN-DNN are competitive but do not clearly beat BDT v2.
- BDT v2 remains the strongest stable classical baseline so far.
- SPA-Net is still promising because HH4b has a real combinatorics problem, but the truth-matched assignment dataset must be fixed before training.

