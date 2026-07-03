# HH4b SPA-Net recovery status

The original NRP PVC became temporarily inaccessible due to a Ceph/Rook CSI mount failure, so the unpushed HH4b SPA-Net work was reconstructed in a temporary pod.

Recovered and pushed:

- `scripts/hh4b_spanet/build_hh4b_resolved_spanet_dataset.py`

Current known result from the previous NRP run:

- Total HH_4b events scanned: 19,984
- Events with two H→bb candidates: 1,606
- Events with at least four selected AK4 jets: 17,603
- Fully matchable events in top selected jets: 172
- Train/val/test: 120 / 25 / 27
- Baseline full-event pairing accuracy: about 0.209
- Baseline per-Higgs pairing accuracy: about 0.308

Interpretation:

The resolved HH4b SPA-Net dataset is not ready for training yet. The fully matchable event count is too small. The next step is to debug GenPart truth tracing and matching efficiency before training SPA-Net.
