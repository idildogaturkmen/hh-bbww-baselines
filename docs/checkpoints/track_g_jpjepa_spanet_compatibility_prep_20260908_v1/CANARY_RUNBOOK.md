# Canary Runbook: JP-JEPA Mini on Real Yang-Li Jets

Status: **EXECUTED** (small scale, per task authorization -- "O(100-1000) jets")
Scale: 1,050 real jets (350 QCD + 350 signal HH4b + 350 ttbar), sampled
from the first 150 events of one representative file per process.

This is a canary, not production. No SPA2M cohort selection, no 2M/400K
manifest was touched, nothing was written outside this package's own
`work/` directory, and no shared/production location was written to.

## 1. Environment used

The default LPC `python3` (system) has `awkward`/`uproot`/`numpy` but
**no `torch`** (`UPSTREAM_PROVENANCE.md`). Rather than install into any
shared environment, this task created an **isolated, throwaway virtual
environment** in scratch space (`python3 -m venv`, CPU-only `torch`
from `download.pytorch.org/whl/cpu`, plus `uproot`, `awkward`, `numpy`,
`fsspec-xrootd`). Nothing was installed system-wide or into any
conda/pixi environment used by other work in this project.

Resulting versions (this venv):

```
torch      2.8.0+cpu
uproot     5.6.9
awkward    2.8.12
numpy      2.0.2
```

Coincidentally, `torch==2.8.0+cpu` matches the version already recorded
in this project's existing (2026-08-25) dual ParT+JP-JEPA canary
receipt on EOS -- not a deliberate pin, just both being "current CPU
stable" at roughly the same time (see `STATUS.md` for the full
discovery of that prior work).

**For any future, larger-than-canary run**, this project already has a
dedicated, version-pinned production environment for exactly this
purpose: `eaf_embed_env` (python3.9, `torch==2.8.0`+cuda12.8,
`uproot==5.6.9`, `awkward==2.8.12`, `h5py==3.14.0`,
`fsspec_xrootd==0.5.5`, `pytorch-lightning==2.6.0`), frozen at
`/eos/uscms/store/user/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/eaf_gpu_validation_20260825/EAF_EMBED_ENV_FREEZE_20260825.txt`
(discovered during this task -- see `STATUS.md`). That environment
already runs JP-JEPA Mini today (see the same status doc). This
canary's throwaway venv is independent of it by design (to keep this
compatibility audit from touching any shared production asset), but
any follow-on production work should use that existing pinned
environment, not re-derive a new one.

## 2. Checkpoint used

`jpjepa_mini_pretrained.ckpt`, read **read-only** from the existing,
already-downloaded, already-hashed copy at
`/eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/jpjepa_mini_pretrained.ckpt`.
SHA256 independently re-verified by this task:

```
1484e704d9f872c17eed6ead713f9ac39c6866dd74d5ebb45c23c9af7b37e3e9
```

matching the value already recorded in that package's own receipt and
`SHA256SUMS`. No new download from HuggingFace was performed (not
necessary, and avoids creating a second, differently-provenanced copy
of the same weights).

## 3. Data used

Real events, read live via XRootD from the actual Yang-Li ntuples
(`root://cceos.ihep.ac.cn/`), the same three files audited in
`FEATURE_COMPATIBILITY_MATRIX.md`:

| Process | File | Events sampled | Jets extracted |
|---|---|---|---|
| QCD | `QCD_DelphesHH4JTrig_ntuple_mergeid166.root` | 150 | 350 |
| Signal (HH4b 2HDM H3VAR) | `HH4b_2HDM_H3VAR_H1H2_40to200_ntuple_id0-9.root` | 150 | 350 |
| ttbar | `TTbar_ntuple/selected_ntuples_0000.root` | 150 | 350 |

Jet selection: `jet_pt > 30 GeV`, `|jet_eta| < 2.5`, sorted pT-descending,
capped at 10 jets/event -- the same native SPA-Net selection contract.
Feature construction: the verified formulas from `PREPROCESSING_SPEC.md`
/ `FEATURE_COMPATIBILITY_MATRIX.md`, implemented independently in
`scripts/run_jpjepa_mini_canary.py` (not copied from the existing
`extract_shard.py`, though it agrees with it -- see that matrix
document's section 2 for the cross-check).

## 4. Results

Full machine-readable output: `work/canary_report.json`. Summary:

- **Shape**: `(1050, 128)` -- matches Mini's verified 128-d jet
  embedding (`EXTRACTION_POINT_PROOF.md`).
- **Finite**: `all_finite = true` for all 1,050 real jets.
- **Norm**: mean `30.85`, std `1.43`, range `[27.4, 38.6]` -- a tight,
  well-behaved distribution, no blow-up or collapse-to-zero.
- **Per-dimension statistics**: mean-of-per-dim-means `0.34`
  (std-across-dims of those means `2.44`, i.e. different dimensions
  sit at different baseline offsets, as expected for a LayerNorm-free
  pre-final-norm extraction point -- see `EXTRACTION_POINT_PROOF.md`
  section 2's note that this is the *pre*-`self.norm(...)` tensor);
  per-dimension std ranges `[0.30, 3.14]`.
- **Near-constant dimensions**: **zero** of 128 dimensions have
  std `< 1e-4` on this 1,050-jet, three-process sample. No dimension
  looks dead/collapsed at canary scale. (Per the task's Task 10
  instruction, if any dimension *had* come back near-constant, this
  document would describe it as "near-constant/inactive under this
  domain" rather than "noise" -- that framing turned out not to be
  needed here, but the distinction is preserved in the matrix
  document for any dimension that behaves this way at full 2M scale.)
- **Determinism**: two forward passes over the identical input tensors
  produced **bitwise-identical** output (`max_abs_diff = 0.0`). CPU,
  `eval()` mode, no dropout active -- fully reproducible as expected.
- **Padding/mask sensitivity**: see the two-part investigation below --
  net verdict is that masking is exactly correct.
- **Dependence on kinematics**: embedding norm correlates moderately
  with jet `pT` (`r = 0.47`) and weakly with constituent multiplicity
  (`r = 0.13`) and `|eta|` (`r = 0.09`) -- a plausible, not alarming,
  pattern (harder/busier jets produce a somewhat larger-norm
  representation).
- **Signal/QCD/ttbar broad behavior**: mean embedding norms are close
  across processes (`30.68` QCD / `31.17` signal / `30.70` ttbar), but
  the per-process **mean embedding vector** differs measurably from the
  global mean (L2 distance `0.91` QCD / `1.85` signal / `1.48` ttbar) --
  i.e. the frozen Mini representation is *not* process-blind even
  before any SPA-Net training sees it, which is the expected and
  desired behavior for a representation intended to help
  classification.
- **Constituent multiplicity in this sample**: min `1`, max `55`,
  mean `17.8` -- consistent with the broader `FEATURE_COMPATIBILITY_MATRIX.md`
  section 5 finding that AK4 jets are far sparser than `maxlen=128`.

### Padding/mask sensitivity -- two-part investigation

**First attempt (adversarial, flawed)**: corrupted every zero-padded
slot with unconstrained IID Gaussian noise on `(px,py,pz,E)` (mean 0,
std 50) and on the 17 standardized features (mean 0, std 5), keeping
`mask=0` there, and re-ran. **Result: non-finite (NaN) output.** This
looked like a masking failure but was not one -- see below.

**Root cause, isolated**: a minimal synthetic reproduction (a single
jet, 15 "real" constituents given literally random, unconstrained
4-vectors, **zero padding involved at all**) already produces
non-finite output. `ParticleTransformer`'s pairwise-feature computation
(`pairwise_lv_fts` -> `to_ptrapphim` -> `rapidity =
0.5*log(1 + 2*pz/(E-pz))`, `particle_transformer.py:52`) takes
`log` of a value that goes negative whenever a 4-vector is
**unphysical** (violates `E >= |p|`, i.e. spacelike or negative-energy)
-- exactly what unconstrained random noise produces with high
probability. **This is a property of feeding the architecture any
non-physical 4-vector, independent of padding or masking**, and real
particle data (real or -- critically -- exact-zero padding, which is
`(0,0,0,0)`, a degenerate but non-spacelike, non-negative-energy case)
never triggers it.

**Corrected test (physical)**: on 60 real QCD jets, each jet's padded
slots were filled with **another real jet's genuine constituent
content** (a different, physically valid 4-vector and feature set,
"donor" jets resampled to fill the padding length), with `mask`
unchanged at `0` for those slots. **Result: output identical to the
true (zero-padded) embedding to `0.0` absolute difference (bitwise),
for all 60 jets, and the repeat-run determinism check on this same
setup was also bitwise-identical.**

**Verdict**: masking is exactly, correctly implemented end-to-end
(`SequenceTrimmer` / `key_padding_mask` in `Block.forward` /
`masked_fill` after `Embed`) for any physically valid input. The
adversarial NaN was a test-construction artifact (feeding
physically-impossible 4-vectors), not a defect in JP-JEPA or in this
project's masking/padding convention. Full detail (including the
isolation script's console output) is preserved in
`work/canary_report.json`'s `padding_mask_sensitivity_check` field and
`scripts/run_jpjepa_mini_canary.py`.

## 5. Mini vs. Small comparison

**Not performed.** Per the task's explicit instruction ("Prepare Small
... as an OPTIONAL secondary sensitivity if time/resources permit. Do
not automatically launch both"), and since Mini alone already fully
answers the compatibility question this package exists to answer, this
canary used Mini only. `jpjepa_small_pretrained.ckpt` was not
downloaded. If a Small sensitivity check is wanted later, the same
canary script accepts a `part_small`-based encoder with a one-line
change (swap the `part_mini` import for `part_small` and point at a
downloaded `jpjepa_small_pretrained.ckpt`); this is left prepared, not
executed, consistent with the task's "optional secondary" framing.

## 6. Reproducing this canary

```bash
python3 -m venv /path/to/scratch/venv
source /path/to/scratch/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install uproot awkward numpy fsspec-xrootd

python3 scripts/run_jpjepa_mini_canary.py \
  --jpjepa-src-dir <path to a clone of jetparticle-jepa @ c68509eead1866c2c86714147023f5e8312634c4> \
  --checkpoint /eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/jpjepa_mini_pretrained.ckpt \
  --n-events-per-file 150 --max-jets-per-process 350 \
  --out-json work/canary_report.json
```

Requires a valid X.509 grid proxy with IHEP EOS read access (this task
used the invoking user's existing `voms-proxy`, already valid at the
time -- see `STATUS.md` for the note on a *separate* environment's
currently-unrelated auth issue against the same endpoint).
