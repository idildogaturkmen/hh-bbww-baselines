# Preprocessing Spec: Exact Upstream JP-JEPA Input Pipeline

Status: VERIFIED AGAINST UPSTREAM SOURCE (`dataset.py`, `inference.py`)
Upstream commit: `c68509eead1866c2c86714147023f5e8312634c4`

This document reproduces upstream's exact transformations. Nothing here
is "similar" preprocessing -- every formula below is copied or directly
derived from `dataset.py`/`inference.py` with a line citation. Any place
where this project's own data cannot satisfy an assumption is deferred
to `FEATURE_COMPATIBILITY_MATRIX.md`, not silently patched here.

## 1. Raw branches read from ROOT (`dataset.py:238`, `_read_files` default)

```python
branches: list[str] = ['part_*', 'jet_pt', 'jet_energy', 'jet_sdmass', 'label_*']
```

Read via `uproot.open(filepath)["tree"].arrays(filter_name=branches,
entry_start=start, entry_stop=stop, library="ak")`
(`dataset.py:192-205`). The tree name is hardcoded as `"tree"`
(`dataset.py:198`).

The **raw, on-disk** `part_*` branches actually consumed downstream
(traced through every `a['part_...']` read in `_preprocess`,
`dataset.py:338-383`, before any are computed) are:

```
part_px, part_py, part_pz, part_energy
part_deta, part_dphi
part_charge
part_isChargedHadron, part_isNeutralHadron, part_isPhoton, part_isElectron, part_isMuon
part_d0val, part_d0err
part_dzval, part_dzerr
```

plus the jet-level `jet_pt`, `jet_energy` and the ten JetClass one-hot
`label_*` columns (`JETCLASS_LABEL_LIST`, `dataset.py:22`) used only to
build the classification label (irrelevant to embedding extraction).

Two of these raw names are easy to confuse with *derived* feature names
of the same stem: the raw branches are `part_d0val`/`part_dzval`
(impact-parameter **values**), not `part_d0`/`part_dz` -- those two are
computed (tanh-transformed) below. Likewise `part_isChargedHadron`/
`part_isNeutralHadron` are the raw branch names; `part_isCHad`/
`part_isNHad` are renamed copies used in the feature list.

## 2. Derived per-constituent quantities (`_preprocess`, `dataset.py:348-359`)

Computed in this exact order, on the awkward array `a`, before any
padding:

```python
a['part_mask']      = ak.ones_like(a['part_energy'])
a['part_pt']         = np.hypot(a['part_px'], a['part_py'])
a['part_pt_log']     = np.log(a['part_pt'])
a['part_e_log']      = np.log(a['part_energy'])
a['part_logptrel']   = np.log(a['part_pt'] / a['jet_pt'])
a['part_logerel']    = np.log(a['part_energy'] / a['jet_energy'])
a['part_deltaR']     = np.hypot(a['part_deta'], a['part_dphi'])
a['part_d0']         = np.tanh(a['part_d0val'])
a['part_dz']         = np.tanh(a['part_dzval'])
a['part_isCHad']     = a['part_isChargedHadron']
a['part_isNHad']     = a['part_isNeutralHadron']
```

Notes verified directly from source:

- `part_mask` is **always exactly 1 for every stored (real) constituent
  at this stage** -- it is `ones_like`, not derived from any kinematic
  cut. Zeros only appear later, from padding (section 4).
- `part_deltaR` is `hypot(deta, dphi)` -- i.e. `sqrt(deta**2 +
  dphi**2)`, the flat-metric jet-frame angular distance between the
  constituent and the jet axis, **not** a rapidity-based `deltaR`
  formula and **not** an inter-particle distance; `deta`/`dphi` here
  are each constituent's own displacement from the jet axis.
- `part_d0`/`part_dz` are `tanh` of the *raw* `part_d0val`/`part_dzval`
  branches (no subtraction, no scaling before the `tanh`). `tanh`
  bounds them to `(-1, 1)` and heavily compresses any large impact
  parameter (mis-reconstructed tracks, pileup-associated tracks) toward
  the same `±1` corner as a genuinely enormous displacement -- an
  important saturation behavior to keep in mind if the source
  distribution's `d0`/`dz` units or typical magnitude differ from
  JetClass's (see `FEATURE_COMPATIBILITY_MATRIX.md`, track-displacement
  audit).
- `part_isCHad`/`part_isNHad` are pure renames of
  `part_isChargedHadron`/`part_isNeutralHadron` -- identical values,
  different key.

## 3. Feature groups and their exact final column order (`COLUMNS_GROUPS`, `dataset.py:69-151`)

```python
COLUMNS_GROUPS = {
  'pf_points':  ['part_deta', 'part_dphi'],
  'pf_features': [
      'part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel', 'part_deltaR',
      'part_charge', 'part_isCHad', 'part_isNHad', 'part_isPhoton', 'part_isElectron', 'part_isMuon',
      'part_d0', 'part_d0err', 'part_dz', 'part_dzerr',
      'part_deta', 'part_dphi',
  ],
  'pf_vectors': ['part_px', 'part_py', 'part_pz', 'part_energy'],
  'pf_mask':    ['part_mask'],
  'label':      ['label'],
}
```

**Verified: 17 `pf_features` in exactly this order**, matching the
task prompt's list verbatim, with `part_deta`/`part_dphi` appended at
the **end**, not the start, of the stored `pf_features` tensor.

`_finalize_inputs` (`dataset.py:386-427`) stacks each group's named
columns along a new last axis (`axis=2`) after per-jet padding, so
`pf_features` has shape `(num_jets, maxlen, 17)`, `pf_vectors`
`(num_jets, maxlen, 4)`, `pf_mask` `(num_jets, maxlen, 1)`.

### 3a. Standardization / clipping applied only to a subset of `pf_features` (`dataset.py:405-420`)

Applied to the *unpadded* awkward arrays, **before** padding, inside
`_finalize_inputs`'s `k == "pf_features"` branch:

```python
part_pt_log    = clip((part_pt_log  - 1.7)  * 0.7, -5, 5)
part_e_log     = clip((part_e_log   - 2.0)  * 0.7, -5, 5)
part_logptrel  = clip((part_logptrel - (-4.7)) * 0.7, -5, 5)
part_logerel   = clip((part_logerel  - (-4.7)) * 0.7, -5, 5)
part_deltaR    = clip((part_deltaR  - 0.2)  * 4.0, -5, 5)
part_d0err     = clip(part_d0err, 0, 1)     # only if present in the kept feature list
part_dzerr     = clip(part_dzerr, 0, 1)     # only if present in the kept feature list
```

The remaining 10 `pf_features` columns (`part_charge`, `part_isCHad`,
`part_isNHad`, `part_isPhoton`, `part_isElectron`, `part_isMuon`,
`part_d0`, `part_dz`, `part_deta`, `part_dphi`) receive **no further
standardization at this stage** beyond whatever was already applied
when they were computed (e.g. `part_d0`/`part_dz` already went through
`tanh` in section 2; `part_deta`/`part_dphi` and the categorical/PID
flags are passed through completely raw). These five affine
shift-then-scale-then-clip constants (`1.7`, `2.0`, `-4.7`, `-4.7`,
`0.2`, each with its own scale factor) are exactly the standard
Particle Transformer / ParticleNet JetClass preprocessing constants
from the original ParT/weaver data-config convention -- they are
**JetClass-population-specific normalization constants**, not
universal physics constants, and are frozen exactly as upstream wrote
them (no attempt is made here to re-derive or "improve" them for a
different jet population; see `STATUS.md` domain-shift discussion).

### 3b. `pf_td` group note (`dataset.py:418-420`)

A separate `pf_td` group (`['part_d0','part_d0err','part_dz','part_dzerr']`)
independently re-applies the same `d0err`/`dzerr` clip. This group is
not consumed by `inference.py` (which only reads `pf_features`,
`pf_vectors`, `pf_mask`, `label`) -- it exists in `dataset.py` for other
downstream uses (e.g. visualization) and is irrelevant to the embedding
extraction path documented in `EXTRACTION_POINT_PROOF.md`. Recorded here
for completeness per the task's request to reproduce the full published
pipeline faithfully, not to selectively read only the parts that are
convenient.

## 4. Padding and mask semantics (`_pad`, `dataset.py:298-313`)

```python
def _pad(a, maxlen, value=0, dtype='float32'):
    ...
    a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
    return ak.values_astype(a, dtype)
```

- Every per-constituent quantity (all `pf_features`, `pf_vectors`
  columns, and `pf_mask`) is padded/truncated to a **fixed length of
  `maxlen`** per jet with `ak.pad_none(..., clip=True)` (truncates if a
  jet has more than `maxlen` constituents) then `ak.fill_none(...,
  value)` (fills missing slots up to `maxlen` with `value`, default
  `0`).
- Consequently: **padded slots get `part_mask = 0`** (since
  `part_mask`'s pre-pad value is 1 for every real constituent, and the
  fill value is 0), **and get `0.0` for every `pf_features`/`pf_vectors`
  column** (no special sentinel value; `0` also happens to be a
  plausible in-range value for several standardized features, so
  masking must always be applied alongside the raw feature tensor --
  never inferred from the feature values themselves).
- All output dtype is `float32` (`dtype='float32'` default).
- `maxlen = 128` in `inference.py:22`
  (`JetClassTaggingIterableDataModule(root=..., maxlen=128,
  batch_size=32)`). **Verified: `maxlen = 128` exactly**, matching the
  task prompt.
- Constituent **ordering within a jet is whatever order the ROOT branch
  arrays store them in** -- `dataset.py` performs no explicit sort (no
  `argsort`, no reference to `part_pt` for ordering) anywhere in the
  read/pad/stack path. See `FEATURE_COMPATIBILITY_MATRIX.md` and the
  ordering-specific audit for what order JetClass itself stores
  constituents in, and what order the Yang-Li source provides.
- `min_num_particles` (an optional per-dataset filter dropping jets
  with too few stored constituents, `dataset.py:207-212`) defaults to
  `None` (no filtering) unless a datamodule explicitly sets
  `train_min_num_particles`/`val_min_num_particles`/
  `test_min_num_particles`; `inference.py` never sets these, so no jet
  is dropped by this mechanism during inference.

## 5. Feature-order permutation performed by `inference.py` itself (`inference.py:47`)

```python
features = batch["pf_features"]          # (batch, maxlen, 17), order = COLUMNS_GROUPS['pf_features']
features = torch.concatenate((features[:, :, -2:], features[:, :, :-2]), dim=-1).permute(0, 2, 1).contiguous()
```

`features[:, :, -2:]` selects the **last two** columns of the 17
(`part_deta`, `part_dphi`, in that order, per section 3), and
`torch.concatenate((..., features[:, :, :-2]), dim=-1)` places them
**first**, ahead of the remaining 15. The resulting column order fed
into the encoder is:

```
[part_deta, part_dphi,
 part_pt_log, part_e_log, part_logptrel, part_logerel, part_deltaR,
 part_charge, part_isCHad, part_isNHad, part_isPhoton, part_isElectron, part_isMuon,
 part_d0, part_d0err, part_dz, part_dzerr]
```

then `.permute(0, 2, 1)` transposes from `(batch, maxlen, 17)` to
`(batch, 17, maxlen)` -- the channel-first layout
`ParticleTransformer.forward` expects for `x` (`particle_transformer.py:534`,
`x: (N, C, P)`). **Verified: this exact reorder-then-transpose is what
the task prompt described** ("reorders the final two deta/dphi features
to the front").

`lorentz_vectors = batch["pf_vectors"].permute(0, 2, 1)` &rarr;
`(batch, 4, maxlen)`, column order `[px, py, pz, energy]` (section 3,
no reordering applied to this tensor). **Verified: lorentz vectors =
(px, py, pz, energy)**, matching the task prompt.

`mask = batch["pf_mask"].permute(0, 2, 1)` &rarr; `(batch, 1, maxlen)`,
boolean-valued-as-float (`1.0` real / `0.0` padded, section 4).

## 6. Where the physics-motivated pairwise attention bias comes from

Not explicitly asked for by the task's numbered list, but essential to
faithful reproduction: `ParticleTransformer.forward` builds an
attention bias from `v` (the Lorentz vectors) via `self.pair_embed(v,
uu)` with `uu=None` (`particle_transformer.py:551-552`), i.e. purely
from `pairwise_lv_fts` (`particle_transformer.py:77-113`) computed on
`(px, py, pz, E)` pairs -- default `pair_input_dim=4` yields
`(ln kT, ln z, ln ΔR, ln m²)` per constituent pair
(`pairwise_lv_fts` `num_outputs=4` branch, lines 86-95). This means
**any error in the reconstructed `(px, py, pz, E)` four-vectors biases
every attention weight in the encoder**, not just the 4 raw
`pf_vectors` input channels -- so the four-vector reconstruction
(section on Lorentz vectors in `FEATURE_COMPATIBILITY_MATRIX.md`) is at
least as important to get exactly right as the 17 `pf_features`.

## 7. Summary of frozen constants (for implementers)

| Constant | Value | Source |
|---|---|---|
| `maxlen` | 128 | `inference.py:22` |
| `batch_size` (upstream default, irrelevant to embedding correctness) | 32 | `inference.py:22` |
| pad/truncate fill value | `0.0` (`float32`) | `dataset.py:298` default `value=0` |
| `part_pt_log` shift, scale | `1.7`, `0.7` | `dataset.py:408` |
| `part_e_log` shift, scale | `2.0`, `0.7` | `dataset.py:409` |
| `part_logptrel` shift, scale | `-4.7`, `0.7` | `dataset.py:410` |
| `part_logerel` shift, scale | `-4.7`, `0.7` | `dataset.py:411` |
| `part_deltaR` shift, scale | `0.2`, `4.0` | `dataset.py:412` |
| clip range (all 5 above) | `[-5, 5]` | `dataset.py:408-412` |
| `part_d0err`/`part_dzerr` clip | `[0, 1]` | `dataset.py:414,416` |
| `part_d0`/`part_dz` transform | `tanh(raw val/dzval)` | `dataset.py:356-357` |
| pairwise LV feature count | 4 (`lnkt, lnz, lndelta, lnm2`) | `particle_transformer.py:86-95`, default `pair_input_dim=4` |
