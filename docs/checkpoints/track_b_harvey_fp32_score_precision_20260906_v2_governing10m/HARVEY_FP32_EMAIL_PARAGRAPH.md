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
FP32 training, or lost discrimination power. We also directly
cross-checked, event by event, whether `u_prob32 > threshold` and
`u_logit > threshold` ever disagree: at u>3.5 they select the identical
9,258 events (zero disagreements); at u>4.5, 3 out of 6,454 passing
events (0.046%) disagree -- all 3 sit within one float32 rounding step
of the threshold. So FP32 quantization does not meaningfully affect the
u>3.5/u>4.5 regions in this cohort, with that small measured exception
disclosed rather than hidden; and it's zero QCD/ttbar events at any of
these thresholds **in this 400,000-event development cohort**
specifically -- this is an FP32/logit numerical audit, not a new
physical-background-yield determination, and it does not replace the
separate full physical-background tail study, where the much larger
pooled QCD/ttbar/minor-background samples do contain rare survivors.
Full per-event archive, census, and figures are in
`docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m/`.
