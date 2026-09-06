# Email paragraph for Harvey

We reran the actual governing SPA-Net 10M checkpoint (the frozen
50-epoch primary, hash-verified) CPU-only over the full 400,000-event
development cohort and pulled the raw pre-softmax logits this time, not
just the stored probability. The spikes are confirmed real and exactly
where the float32 lattice near 1.0 predicts: 99 events at u=6.9237
(k=2) and 100 at u=6.6226 (k=4), plus 67 events at exact float32 1.0 --
tens of events, not hundreds, and every single one of them is signal;
zero QCD or ttbar events appear anywhere above u=5.5 in this cohort.
Using the recovered logits, we confirmed the mechanism directly: within
the u=6.9 and u=6.6 spikes (and everywhere else we checked, out to
u>3.5), essentially every event keeps its own distinct raw logit margin
(2,239/2,239 distinct at u>5.5) even though it collapses onto one of
only 27 stored float32 probabilities -- the classifier's internal
ranking is fully intact and varies smoothly; only the 32-bit probability
*storage* saturates. This is not evidence of a broken model, invalid
FP32 training, or lost discrimination power, and it does not
meaningfully affect the u>3.5/u>4.5 regions either, since no background
event reaches anywhere near that confidence level in this cohort and
every count-based physics quantity we use (efficiencies, rejections,
yields) only depends on threshold crossings, which are exact regardless
of these ties. Full per-event archive, census, and figures are in
`docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m/`.
