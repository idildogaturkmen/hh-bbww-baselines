# Draft email to Guillaume/Antonin -- NOT SENT

Status: **DRAFT ONLY**. Not sent by this task, per explicit instruction.
If useful, a human should review, adjust addressing/tone, and send
manually.

---

**Subject: CPU-vs-GPU numerical gap in JP-JEPA Mini `jet_embedding` (~15-150x larger than a same-architecture ParT control) -- one question**

Hi Guillaume, hi Antonin,

We're using the released `jpjepa_mini_pretrained.ckpt` checkpoint
(commit `c68509e`) purely for frozen inference -- extracting
`all_layer_outputs[-1][0][0]` exactly as `inference.py` does, no
fine-tuning, no architecture changes -- as an input feature to a
downstream classifier, on resolved AK4 jets (out-of-domain relative to
JetClass, which we've accounted for separately).

We found that this `jet_embedding` differs measurably between a CPU and
a GPU (A100) forward pass on the *same* 12,089 fixed jets, same
checkpoint, `model.eval()`, `torch.no_grad()`, float32 throughout, no
autocast: `max_abs_diff = 0.060`, `mean_abs_diff = 0.00157`, cosine
similarity as low as `0.99999` for the worst jet, `relative_L2` up to
`0.0047`. We ruled out batch size (bit-identical across batch sizes
from 64-512 on CPU) and GPU run-to-run nondeterminism (two independent
GPU runs are bit-identical to each other).

The part that surprised us: we also run an architecturally-identical
model (same `embed_dims=[128,512,128]`, `num_heads=8`, `num_layers=8`,
`num_cls_layers=2`) that is ParT, JetClass-supervised-pretrained rather
than JEPA-pretrained, through the *same* extraction code, on the *same*
jets. ParT's CPU-vs-GPU gap is 15-150x smaller across every metric we
checked (e.g. `mean_abs_diff` 1.0e-05 for ParT vs 1.6e-03 for JP-JEPA
Mini). Both models go through PyTorch's non-fused reference attention
path (we call `nn.MultiheadAttention` with `need_weights=True` in both,
since we also read out the attention matrices), so it isn't a
fused-kernel-selection difference.

Our own layer-by-layer look (CPU-only, we don't have easy GPU access to
push this further ourselves right now) shows the sharpest activation-
magnitude jump in the whole 10-block stack happens at the two
class-attention blocks -- the same place `jet_embedding` is read from --
which on our AK4 jets are aggregating over a sequence that's >80%
zero-padding (mean ~18 real constituents against `maxlen=128`).

**Our question**: does anything about JP-JEPA Mini's pretraining (the
JEPA objective itself, a specific normalization/initialization choice,
or numerical properties of the released checkpoint's weights) make you
expect it to be meaningfully more sensitive to ordinary GPU-vs-CPU
float32 non-associativity than a comparably-sized, classically-trained
ParT model would be? Or would you also find a gap this size surprising
for a released checkpoint, in which case we'd suspect something
specific to our own AK4-jet, heavily-padded input distribution
interacting with the class-attention aggregation, rather than anything
about your checkpoint itself.

For context: we've bounded the practical impact -- even our single
worst-case jet's CPU/GPU difference is well under the jet-to-jet and
process-to-process separation we measure in the same embedding space,
so this isn't blocking us -- we're just trying to understand the
mechanism rather than just work around it.

Happy to share our exact reproduction script/data if useful.

Thanks,
[sender]

---

## Internal notes on this draft (not part of the email itself)

- Numbers cited are exact matches to `CPU_GPU_NUMERICS_AUDIT.md`
  sections 4-5; verify they have not drifted if this email is sent
  later than this document's freeze date (2026-09-09).
- The email deliberately does **not** ask them to fix anything or imply
  a bug report against their released code -- our own audit (section 3)
  found nothing in *our* extraction code that explicitly requests
  reduced precision, and their `inference.py`/`particle_transformer.py`
  are being used exactly as published.
- "One concrete question" per the task instruction: the single
  question is the "does anything about JP-JEPA's pretraining..."
  paragraph. Everything else is supporting context, kept short.
- Recipient identity: `UPSTREAM_PROVENANCE.md` records only Guillaume
  Letellier (`guillaume.letellier@unicaen.fr`) as the commit author of
  the single-commit `jetparticle-jepa` repository; "Antonin" was named
  in the task instruction and is assumed to be a paper co-author known
  to the user -- their email address was not independently looked up or
  verified by this task (no such lookup was in scope, and doing so
  would not be appropriate without the user's own knowledge of the
  right contact). **Do not send without confirming/filling in the
  actual recipient address(es) and a real sender line.**
