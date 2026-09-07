Subject: Fifth-jet causal test, QCD stability, spike kinematics, ttbar validation — closing out the open items

Hi Harvey,

This closes out the items still open after the last few rounds. Full
numbers and caveats are in the attached technical summary; this note hits
the headlines.

**Fifth jet — actually tested, not just observed.** We ran the real
counterfactual: same event, same frozen SPA-Net 10M checkpoint, once with
its native input and once with jets ranked ≥5 masked out. Every one of the
88,854 affected signal events changes score under masking — none are
unaffected. On average the extra jet(s) help a little (median shift
−0.09 in logit-space units, mean −0.15), but 42% of events actually move
the other way. At our working points, masking pushes far more events
*into* the tail than out of it (u>3.5: +2,578 vs. −330; u>4.5: +3,441 vs.
−610). The effect scales with the extra jet's pT and depends on its
angular separation from the leading four in a real, non-trivial way. So:
yes, the fifth jet causally perturbs the classification, and now we can
say by how much and for whom.

**QCD stability — checked empirically, not projected.** Using the existing
two QCD production lanes as they actually accumulated (not a hypothetical
future run), the u>3.5 tail rate is statistically consistent with one
constant underlying rate across the whole sample, and the two lanes agree
with each other (p=0.90). At u>4.5 we only have 12 raw events total, which
is genuinely too few to judge either way — we're saying that plainly
rather than forcing a verdict. One real caveat surfaced during this check:
we can prove a uniform per-file generation count for one of the two lanes
but not the other (the job/file ratio isn't an integer for lane 2), so
lane 2's per-file exposure axis is an efficiency-based approximation, not
an audited count — disclosed in the technical note.

**The u≈6.6/6.9 spikes — still just float32 bins, and now confirmed
kinematically boring.** We checked whether events landing in those two
bins look different from their neighbors across 13 kinematic variables.
None of the 26 comparisons show a meaningful effect. They're an
unremarkable draw from the surrounding tail, consistent with the
quantization story you already had.

**The ttbar cut doesn't hold up under a real cross-check.** Testing the
pT_H1<406 GeV candidate against the two independent ttbar production lanes
(11 and 16 events) instead of the pooled 27-event sample it was originally
tuned on, the rejection swings from 36% to 63% lane-to-lane, and a
leave-one-lane-out derivation gives thresholds 337 GeV vs. 550 GeV — a
1.6x spread. The qualitative picture (surviving ttbar is harder/more
boosted than signal) is real and shows up in both lanes for pT_H1 and
leading-jet pT, but the specific cut and its quoted rejection power
shouldn't be treated as validated with only 27 raw events behind it.

**Preselection, and the doubling-cost/single-precision/75-epoch/
single-event-dominance questions** are unchanged from what we've already
sent — reorganized into one table in the technical note for reference,
not recomputed, since nothing new applies to them.

Everything above used only the already-frozen governing checkpoint and the
already-existing production; the fifth-jet test is the only new inference
run (CPU-only, no training, no new MC, no holdout/Stage-C access). Happy
to walk through any of this on a call.

Best,
Idil
