# Extraction Point Proof: JP-JEPA Jet and Particle Embeddings

Status: VERIFIED AGAINST UPSTREAM SOURCE
Upstream commit: `c68509eead1866c2c86714147023f5e8312634c4`

All line numbers below refer to the files as committed at that SHA
(also reproduced under `UPSTREAM_PROVENANCE.md`'s SHA256 table).

## 1. The claim to verify

The task prompt asserted upstream does, for pretrained checkpoints:

```python
encoder = model
_, all_layer_outputs = encoder.forward(features, lorentz_vectors, mask)
jet_embedding = all_layer_outputs[-1][0][0]
```

**Verified: this is an exact, verbatim match** to `inference.py` lines
38-39 and 51-55:

```python
if CKPT_TYPE == "pretrained":
    encoder = model
    cls_head = torch.nn.Identity()
...
_, all_layer_outputs = encoder.forward(
    features, lorentz_vectors, mask,
)

jet_embedding = all_layer_outputs[-1][0][0]
part_embedding = all_layer_outputs[-4][0].permute(1, 0, 2).contiguous()
print(jet_embedding.shape)  # (32, 384 or 128)
print(part_embedding.shape)  # (32, 128, 384 or 128)
```

`model` itself, for `CKPT_TYPE == "pretrained"`, is produced by
`part_mini(pretrained_weights=..., num_classes=None)` /
`part_small(pretrained_weights=..., num_classes=None)`
(`inference.py:33-36`), which resolve to `_part_init_fn` in
`particle_transformer.py:597-666`. With `num_classes=None` and the
default `fc_params=None`, `_part_init_fn` builds a bare
`ParticleTransformer` (`particle_transformer.py:614-627`), loads the
checkpoint's state dict directly onto it (`particle_transformer.py:630-635`,
the `not any(key.startswith("cls_head."))` branch, since a
self-supervised pretrained checkpoint has no `cls_head.*` keys), and
returns the `ParticleTransformer` instance itself -- **not** the
`ParticleTransformerWrapper`. So `encoder = model` is a raw
`ParticleTransformer`, and `encoder.forward(features, lorentz_vectors,
mask)` calls `ParticleTransformer.forward(x=features, v=lorentz_vectors,
mask=mask)` (positional args map to `forward(self, x, v=None,
mask=None, uu=None, uu_idx=None, ...)`, `particle_transformer.py:533`).
`uu`/`uu_idx` (explicit pairwise-interaction feature tensors) are never
passed by `inference.py` -- they stay `None`, so the pairwise attention
bias is built purely from the four Lorentz-vector components via
`PairEmbed` (see `PREPROCESSING_SPEC.md`).

## 2. What `all_layer_outputs` actually is, and why `[-1][0][0]` is a jet embedding

Inside `ParticleTransformer.forward` (`particle_transformer.py:546-578`):

```python
all_outputs = []
...
for block in self.blocks:
    x, attn_matrix = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
    all_outputs.append((x, attn_matrix))          # one entry per MAIN block

cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
for block in self.cls_blocks:
    cls_tokens, attn_matrix = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
    all_outputs.append((cls_tokens, attn_matrix))  # one entry per CLASS-ATTENTION block

x_cls = self.norm(cls_tokens).squeeze(0)

if self.fc is None:
    return x_cls, all_outputs
```

So `all_layer_outputs` (`= all_outputs`) is a flat list of
`(tensor, attn_matrix)` tuples, one per **main self-attention block**
(`self.blocks`, `num_layers` of them) followed by one per
**class-attention block** (`self.cls_blocks`, `num_cls_layers` of
them). `all_layer_outputs[-1]` is therefore always the tuple produced
by the **last class-attention block**, regardless of `num_layers` /
`num_cls_layers`. `all_layer_outputs[-1][0]` is that block's returned
`x_cls` tensor of shape `(1, batch, embed_dim)` (a class-attention
`Block.forward` with `x_cls is not None` returns a tensor shaped like
its `x_cls` input, `particle_transformer.py:404-454`, specifically the
post-residual update at line 452). `all_layer_outputs[-1][0][0]`
indexes away the leading size-1 axis, leaving shape `(batch,
embed_dim)` -- exactly the printed `(32, 384 or 128)`. **Verified.**

### Critical, easy-to-miss subtlety: this is the PRE-final-LayerNorm cls token

`ParticleTransformer.forward` computes a *second*, independently
normalized version of the class token, `x_cls = self.norm(cls_tokens
).squeeze(0)` (`particle_transformer.py:570`), and returns it as the
**first** element of the tuple (`return x_cls, all_outputs`,
line 574). `inference.py:51` discards this first return value with `_,
all_layer_outputs = encoder.forward(...)` and instead reconstructs the
jet embedding by reaching into `all_layer_outputs[-1][0][0]`, which is
`cls_tokens` **before** `self.norm(...)` is applied.

In other words: upstream's own officially-normalized per-jet
representation (`self.norm(cls_tokens)`, the tensor a from-scratch
fine-tuning `cls_head` would actually consume, per
`particle_transformer.py:573-578`, `output = self.fc(x_cls)`) is
**not** what `inference.py` extracts and prints as `jet_embedding`.
The upstream inference script uses the pre-`LayerNorm` version instead.

This is not a bug we get to silently "fix" -- per the task's own
framing, the goal is to reproduce upstream's *exact* published
inference procedure for a frozen, external, released model, not our
own idealized version of it. **This package therefore locks in
`all_layer_outputs[-1][0][0]` (pre-final-norm) as the frozen extraction
point for any future JP-JEPA embedding production**, and flags the
existence of the officially-normalized alternative
(`self.norm(cls_tokens)`) here so it is never silently confused with
it. If a future user wants the normalized variant instead, that is a
deliberate, documented deviation from upstream's own inference.py, not
a bug fix.

## 3. Jet embedding dimensionality: Mini vs. Small

`embed_dim` is set once, from the *last* entry of `embed_dims`
(`particle_transformer.py:486`, `embed_dim = embed_dims[-1] if
len(embed_dims) > 0 else input_dim`), and both `self.cls_token`
(line 514) and every `Block` (`cfg_block['embed_dim']`) are sized to
it. From the model-factory defaults (`particle_transformer.py`):

| Variant | `embed_dims` | `embed_dim` (= jet embedding width) | `num_layers` (main blocks) | `num_cls_layers` |
|---|---|---|---|---|
| `part_mini` (line 669-698) | `[128, 512, 128]` | **128** | 8 | 2 |
| `part_small` (line 733-762) | `[384, 512, 384]` | **384** | 12 | 3 |

**Verified: Mini jet embedding dimension = 128. Small jet embedding
dimension = 384.** This matches the task's stated expectation and
matches the existing frozen ParT contract's 128-d embedding width
(`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`),
making Mini the dimension-matched comparison point.

## 4. Particle (per-constituent) embedding extraction point -- and a real Mini/Small inconsistency

`part_embedding = all_layer_outputs[-4][0].permute(1, 0, 2).contiguous()`
(`inference.py:56`). `all_layer_outputs` has `num_layers +
num_cls_layers` entries total, indices `0 .. num_layers-1` for the main
blocks and `num_layers .. num_layers+num_cls_layers-1` for the
class-attention blocks. The fixed offset `-4` therefore lands at
absolute index `num_layers + num_cls_layers - 4`, and its distance from
the **last main block** (absolute index `num_layers - 1`) is
`(num_layers - 1) - (num_layers + num_cls_layers - 4) = 3 -
num_cls_layers` main-block-steps *before* the last one.

Concretely, verified by direct substitution of each variant's
`num_cls_layers`:

- **Small** (`num_cls_layers = 3`): offset = `3 - 3 = 0` steps before
  the last main block &rarr; `all_layer_outputs[-4]` **is** the last
  (12th) main block's output, shape `(seq_len=128, batch, 384)`, whose
  `.permute(1, 0, 2)` gives `(batch, 128, 384)` -- exactly the printed
  comment `(32, 128, 384 or 128)`.
- **Mini** (`num_cls_layers = 2`): offset = `3 - 2 = 1` step before the
  last main block &rarr; `all_layer_outputs[-4]` is the **7th of 8**
  main blocks (index 6, i.e. one layer *before* the deepest main-block
  output), shape `(128, batch, 128)`.

**This is a genuine, upstream-hardcoded inconsistency**: the same
literal `-4` offset in `inference.py` does not refer to the same
*relative* layer for Mini and Small, because the two variants use a
different number of class-attention blocks (2 vs. 3) and the offset is
counted from the end of the combined list, not from the end of the
main-block stack specifically. This is a fact about upstream's
published inference script, verified directly from its source, not a
guess or a "similar" approximation.

Practically, this only matters if a future extension of this project
consumes the **per-particle** JP-JEPA embedding
(`part_embedding`/`pf_pid`-level use). It does **not** affect the
primary comparison in this package: SPA-Net's frozen input is the
single per-jet `jet_embedding` (`all_layer_outputs[-1][0][0]`), whose
extraction point is identical and unambiguous for both Mini and Small
(section 2 above). This is documented here per the task's explicit
request to record the particle-embedding extraction point "even though
SPA-Net primary input will use one frozen per-jet vector."

## 5. Bottom line for this package

| Quantity | Extraction expression | Verified shape | Frozen for this package |
|---|---|---|---|
| Jet embedding (Mini) | `all_layer_outputs[-1][0][0]` | `(batch, 128)` | **Primary SPA-Net input** |
| Jet embedding (Small) | `all_layer_outputs[-1][0][0]` | `(batch, 384)` | Optional secondary |
| Particle embedding (Mini) | `all_layer_outputs[-4][0].permute(1,0,2)` | `(batch, 128, 128)`, taken from main-block index 6 of 8 | Documented only, not used |
| Particle embedding (Small) | `all_layer_outputs[-4][0].permute(1,0,2)` | `(batch, 128, 384)`, taken from main-block index 11 of 12 (the true last block) | Documented only, not used |
