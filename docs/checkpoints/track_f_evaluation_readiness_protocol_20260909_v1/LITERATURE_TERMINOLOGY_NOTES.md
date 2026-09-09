# Literature/terminology check — ParT and SPA-Net

Scope: enough to keep this project's terminology and comparison claims accurate. This is not a literature
review and does not attempt completeness; it exists to prevent two specific failure modes: (1) misusing
established terms, (2) claiming "state of the art" without a supporting direct comparison. Checked via web
search 2026-09-09.

## ParT — Particle Transformer

**Citation**: Huilin Qu, Congqiao Li, Sitian Qian, "Particle Transformer for Jet Tagging," *Proceedings of the
39th International Conference on Machine Learning (ICML)*, 2022. arXiv:2202.03772.
Official code: https://github.com/jet-universe/particle_transformer

**What it is**: a Transformer-based architecture for **single-jet, multi-class tagging** (e.g. top/W/Z/Higgs/QCD
jet identification) that incorporates pairwise particle-interaction features as an attention bias. Trained and
evaluated on **JetClass**, a 100-million-jet public dataset the same paper introduces.

**What its SOTA claim actually covers**: the paper reports ParT surpassing the prior state-of-the-art
**ParticleNet** on the **JetClass tagging benchmark** specifically. This is a claim about single-jet
classification accuracy on JetClass, not about event-level multi-jet reconstruction, not about any HH→4b
search, and not about any Delphes-simulated sample.

**Relevance to this project**: this project does not re-run ParT's own tagging benchmark and makes no claim
about it. It uses the **frozen, pretrained ParT checkpoint as a fixed feature extractor**, taking a
128-dimensional intermediate representation per jet (later reduced to 20 retained dimensions,
`PREPROCESSING_DECISION.md`) and feeding it as an additional per-jet input to SPA-Net. This is a transfer/
representation-reuse experiment, not a reproduction or extension of ParT's own published result. The checkpoint
in use is documented elsewhere in this project (`SPANET_PART_COMPARISON_CONTRACT_v2.md`) as "the official
JetClass-supervised Particle Transformer checkpoint, not the PKU jetfree-hh4b event-level classifier" — i.e.
explicitly the general-purpose JetClass-trained model, never fine-tuned on this project's data or task.

**Guardrail**: any writeup describing this project's own results must not say ParT (or a model using its
embeddings) is "state of the art" for HH→4b reconstruction, nor imply the JetClass SOTA finding transfers here.
If ParT's embeddings turn out to help, the correct claim is "a frozen, off-task representation transferred
usefully as an auxiliary feature here" — a transfer-learning finding, not a tagging-benchmark finding.

## SPA-Net — Symmetry Preserving Attention Networks

**Citations**:
- Alexander Shmakov, Michael James Fenton, Ta-Wei Ho, Shih-Chieh Hsu, Daniel Whiteson, Pierre Baldi, "SPANet:
  Generalized Permutationless Set Assignment for Particle Physics using Symmetry Preserving Attention,"
  *SciPost Physics* 12, 178 (2022). arXiv:2106.03898.
- Predecessor: "Permutationless Many-Jet Event Reconstruction with Symmetry Preserving Attention Networks,"
  arXiv:2010.09206.
Official code: https://github.com/Alexanders101/SPANet

**What it is**: an attention-based architecture for the **combinatorial jet-assignment problem** — mapping a
set of reconstructed jets onto the decay products of multiple unstable particles (originally top-quark pairs;
generalized to arbitrary permutation-symmetric set-assignment problems) while respecting the relevant
permutation symmetries, jointly with event-level classification.

**Relevance to this project**: SPA-Net is the reconstruction/classification backbone used throughout this
project's HH→4b work (both native-feature and ParT-augmented variants). It is not benchmarked against ParT here
— the two occupy different roles (event-level set assignment vs. single-jet representation learning); ParT's
embeddings are consumed as an *input* to SPA-Net's existing per-jet feature vector, not compared to SPA-Net as
a competing architecture.

**Guardrail**: this project's own SPA-Net numbers (any of the three arms in `PREREGISTRATION.md`) are development/
validation results on a Delphes-simulated sample, not evaluated on any published SPA-Net benchmark (e.g. the
original top-pair or `ttH`/`tttt` reconstruction tasks in the SPA-Net papers above). No comparison to those
papers' own reported accuracies is valid without matching task, sample, and jet-multiplicity/topology — none of
which hold here. No such comparison is made anywhere in this package.

## Standing rule for this project (not just this package)

**"State of the art" is never used to describe this project's own SPA2M / SPA2M+ParT / SPA10M results** unless
a specific, cited, task-and-sample-matched published benchmark supports the comparison. As of 2026-09-09, no
such benchmark is known to exist for this project's exact task (HH→4b reconstruction/classification on a
Delphes-simulated sample using SPA-Net with or without frozen ParT embeddings) — this is, as far as this
protocol's authors are aware, a novel combination, which is a reason for care in framing, not a license to
claim novelty implies superiority.

Sources:
- [Particle Transformer for Jet Tagging (PDF)](https://proceedings.mlr.press/v162/qu22b/qu22b.pdf)
- [Particle Transformer for Jet Tagging (arXiv:2202.03772)](https://arxiv.org/abs/2202.03772v2)
- [jet-universe/particle_transformer (official code)](https://github.com/jet-universe/particle_transformer)
- [SPANet: Generalized Permutationless Set Assignment (arXiv:2106.03898)](https://arxiv.org/abs/2106.03898)
- [Permutationless Many-Jet Event Reconstruction (arXiv:2010.09206)](https://arxiv.org/abs/2010.09206)
- [Alexanders101/SPANet (official code)](https://github.com/Alexanders101/SPANet)
