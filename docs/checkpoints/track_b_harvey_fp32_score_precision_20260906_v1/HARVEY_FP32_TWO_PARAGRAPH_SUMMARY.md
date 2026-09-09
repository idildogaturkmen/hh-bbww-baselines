# Two-paragraph summary for Harvey

The u~6.6 and u~6.9 spikes are real and are exactly explained by FP32
quantization of the stored two-class softmax probability, not by a
model or training defect. Tracing the actual frozen evaluation code
(`evaluate_classification.py`/`evaluate_classification_10M.py`, both
read-only CPU forward passes with `precision="32-true"` training and no
autocast/fp16 anywhere) shows the stored score is exactly the float32
output of `torch.softmax(logits, dim=-1)[:,1]`, saved unchanged for the
2M sibling checkpoint but, for the governing 10M checkpoint, reduced to
summary statistics only -- no per-event array for the governing
checkpoint survives anywhere in the project's frozen record. Pure
arithmetic on the float32 lattice near 1 (spacing exactly `2^-24`)
predicts pile-up points at `u=6.9237` (k=2) and `u=6.6226` (k=4); using
the real, on-disk 400,000-event per-event score archive for the closely
related 2M sibling checkpoint (same code, same cohort, AUC within
1-4e-4 of the governing model), those exact lattice points hold **80**
and **53** signal events respectively, with **zero background events in
this 400,000-event development cohort** at either point across the
entire tail (u>5.5). The largest single pile-up bin is exact float32
`1.0`, at 128 signal events, also zero background events in this
cohort. This is a statement about this specific development archive,
not a physical background-yield determination -- it does not replace
the separate, much larger full physical-background tail study, where
the pooled QCD/ttbar/minor-background statistics can and do contain
rare survivors at tight working points.

So: the spikes are not "hundreds of events" -- they are tens, and they
are exclusively signal, never background, in the only real per-event
archive available. This is evidence of post-softmax float32
quantization (a well-separated classifier pushing many distinct
high-confidence logit margins onto a small set of representable
float32 values just below 1), and it is explicitly *not* evidence that
the model learned something pathological, that FP32 training is
invalid, or that logit ordering is lost -- but that last claim cannot
actually be checked yet, because no archive anywhere stores the raw
pre-softmax logits for either checkpoint. A small (1,256-event,
CPU-only, <2-minute) tail-only rerun of the governing 10M checkpoint,
restricted to the exact event identities already read off the frozen
2M archive, would recover real logits and settle whether the logit
margin stays smooth through these spikes; it has been fully prepared
(script + event-identity list) but was not launched in this
diagnostic-only pass, per instruction.
