# Upstream Provenance: JetParticle-JEPA (JP-JEPA) Inference Repository

Status: FROZEN PROVENANCE RECORD
Date: 2026-09-08

This package is a COMPATIBILITY-PREPARATION exercise only. No JP-JEPA
production embedding run and no SPA-Net training were performed. See
`STATUS.md` for the overall gate outcome.

## Repository identity

- Upstream URL: `https://github.com/Guigui14460/jetparticle-jepa`
- Cloned (read-only, this task) to a scratch directory, HEAD verified
  against the upstream `main` branch via `git ls-remote`.
- Upstream commit (frozen for this entire package):

```
c68509eead1866c2c86714147023f5e8312634c4
```

- `git ls-remote https://github.com/Guigui14460/jetparticle-jepa.git HEAD main`
  returned this exact SHA for both `HEAD` and `refs/heads/main` at the
  time of this task -- i.e. the repository's default branch tip is
  identical to the commit frozen here. There is no drift to reconcile.
- This matches the upstream commit SHA given in the task instructions
  (`c68509eead1866c2c86714147023f5e8312634c4`) exactly.
- Commit metadata (`git log -1`):
  - Author: Guillaume Letellier <guillaume.letellier@unicaen.fr>
  - Date: 2026-08-24 10:32:12 +0200
  - Subject: `initial commit`
  - This is a **single-commit** repository: the "initial commit" is the
    entire history. There is no earlier state to compare against.
- Paper reference (as given verbatim in the upstream README, not
  independently fetched or verified by this task):
  "JetParticle-JEPA (JP-JEPA): An Efficient Self-Supervised
  Representation Learning method for Jet Tagging in High-Energy
  Physics", arXiv:2606.14813.

## Repository contents (complete file listing, excluding `.git/`)

```
.gitattributes
.gitignore
README.md
__init__.py          (empty file)
dataset.py            (904 lines)
inference.py          (64 lines)
particle_transformer.py (858 lines)
```

No `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`,
`Pipfile`, or `LICENSE` file is present upstream. Dependency versions
are **not pinned by upstream anywhere** -- see "Dependency versions"
below for how this gap is handled.

`.gitattributes` declares `*.ckpt` as a Git-LFS-tracked pattern, but no
`.ckpt` files are committed to the repository itself -- pretrained
weights are distributed separately via HuggingFace (see below), not via
git-lfs in this repo.

## SHA256 of every upstream source file (frozen at commit `c68509e`)

```
1d1faa3d403452eb14148749e41619c499b904c3c4137d45983f5b1f0be14759  README.md
5c36657f89c4ecf357e8c99b1352f942d9825951b898fc241e30e455a248f9bf  inference.py
f1bac09c35e037f65db2826a2849bb68e48cdf393b7cfc4d2e0e3f35d5d83bcb  dataset.py
80ea294feb1981e3aa47f566657a10bd7a3f80720b79bf49a3f2a941b64e559e  particle_transformer.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  __init__.py
c9811b282f382c04dfa5077f73101a231bc3af0fc1dad1886bf0a568114f50a8  .gitattributes
b2580eab7825b9f22f790fb0edb7a6e239616e79907004adf36023c7ec4b9a4c  .gitignore
```

These are also reproduced in this package's `SHA256SUMS` alongside this
package's own deliverables.

## README.md (verbatim, 35 lines)

```markdown
# JetParticle-JEPA (JP-JEPA): An Efficient Self-Supervised Representation Learning method for Jet Tagging in High-Energy Physics (Inference Code)

This repository only contains the **inference code** for the paper: **[JetParticle-JEPA](https://arxiv.org/abs/2606.14813)**

## Download weights

You need first to download weights (pretrained or finetuned, Mini or Small variant) at the following link on HuggingFace: **https://huggingface.co/HadesGuigui/JetParticle-JEPA/tree/main**

## Running the inference

Run the inference script with:

\`\`\`bash
python inference.py
\`\`\`

## Dataset

Before running the inference, you need to download the required JetClass data.

At minimum, download the following file:

**[JetClass_Pythia_val_5M.tar](https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1)**

The JetClass dataset files are available on Zenodo:

https://zenodo.org/records/6619768

After downloading and extracting the dataset, **update the corresponding file paths in `inference.py`** so that they point to the location of the dataset on your machine.

Once the paths have been configured, you can run:

\`\`\`bash
python inference.py
\`\`\`
```

Key facts extracted from this README:

- The authors explicitly state this repo is **inference-only**: "This
  repository only contains the inference code." This confirms the task
  instruction's constraint that pretraining/fine-tuning code is
  unreleased -- it is not merely undocumented, it is explicitly absent
  by the authors' own description.
- Pretrained/finetuned, Mini/Small weights: distributed on HuggingFace
  at `https://huggingface.co/HadesGuigui/JetParticle-JEPA/tree/main`
  (not yet downloaded in this task -- see `STATUS.md` gating; download
  is only warranted once the feature-compatibility audit clears).
- The reference dataset for exercising this inference code is JetClass
  (Pythia, 5M validation shard on Zenodo), a **boosted large-R jet**
  tagging benchmark -- this is the domain-shift baseline referenced in
  `PREPROCESSING_SPEC.md` / `STATUS.md` domain-shift discussion, since
  our target is resolved AK4 (R=0.4) jets, not JetClass's large-R jets.

## Pretrained checkpoint filenames and SHA256

**Not yet downloaded.** Per the task's explicit gating instruction, no
full production is authorized in this session, and no HuggingFace
download was initiated during the source-code / feature-compatibility
audit phase, since the audit itself (this package's core deliverable)
does not require the weight files -- it requires only the source code,
which is now fully verified above.

Expected filenames from `inference.py` (`CKPT_TYPE = "pretrained"`
branch, `f"jpjepa_{variant}_{CKPT_TYPE}.ckpt"`):

- `jpjepa_mini_pretrained.ckpt` (primary candidate, embed_dim = 128)
- `jpjepa_small_pretrained.ckpt` (optional secondary, embed_dim = 384)

If/when the canary stage in `CANARY_RUNBOOK.md` is authorized to run,
this section (and `SHA256SUMS`) MUST be updated with the actual
downloaded filenames and their SHA256 before any embedding is trusted
or reused, per this project's standing provenance discipline (see e.g.
the ParT checkpoint SHA recorded in
`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`).

## Dependency versions necessary for exact inference

Upstream pins nothing. From `import` statements actually present in
`dataset.py`, `inference.py`, `particle_transformer.py`:

| Package | Used for | Pinned upstream? |
|---|---|---|
| `torch` | model, tensors, autocast, jit.script | No |
| `lightning.pytorch` (`lightning`) | `pl.LightningDataModule` base class only | No |
| `awkward` (`ak`) | ragged constituent arrays, padding | No |
| `numpy` | dense arrays, clipping | No |
| `uproot` | reading JetClass `.root` files (`tree.arrays`) | No |
| `tqdm` | progress bars only (no numerical effect) | No |

This project's current default `python3` environment (checked this
task) has `awkward==2.8.12`, `uproot==5.6.9`, `numpy==1.23.5`, but
**no `torch` and no `lightning`/`pytorch_lightning` installed** -- so
the JP-JEPA encoder itself cannot yet be instantiated in this
environment as-is. This is recorded as an open environment-provisioning
item in `STATUS.md`; it does not block the source-level compatibility
audit in this package, but it DOES block the canary stage
(`CANARY_RUNBOOK.md`) until resolved (see that document's prerequisites
section). No environment was modified, installed into, or upgraded in
this task.

## Non-goals honored by this record

- No pretraining/fine-tuning code was reconstructed or reimplemented
  from the paper.
- No claim is made that JP-JEPA was pretrained on R=0.4 jets by anyone
  in this project -- the paper's pretraining domain is JetClass
  (large-R boosted jets), and only released pretrained weights are in
  scope, per the task instructions.
- No repository was cloned into this project's own git tree, and no
  commit was made to this repository. The upstream clone lives only in
  a scratch working directory outside this repo's working tree.

## 2026-09-09 checkpoint provenance update

The earlier "not yet downloaded" language above records the state of the
initial 2026-09-08 source-only compatibility audit. Subsequent canary and
numerics work located and reused the project's already-existing frozen EOS copy:

`/eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/jpjepa_mini_pretrained.ckpt`

No new HuggingFace download was performed.

The checkpoint SHA256 was independently re-verified as:

`1484e704d9f872c17eed6ead713f9ac39c6866dd74d5ebb45c23c9af7b37e3e9`

This supersedes the earlier audit-phase statement for all subsequent canary,
numerics, and planned production work.
