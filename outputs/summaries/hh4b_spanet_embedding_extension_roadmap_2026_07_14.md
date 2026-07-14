# HH→4b SPA-Net + pretrained jet-embedding extension roadmap

## Motivation

The baseline paper story should be a CMS-style AK4/AK8 Delphes HH→4b analysis with SPA-Net Higgs assignment, separate QCD/top rejection heads, and ZZ/ZH→4b validation. A natural novel ML extension is to inject pretrained jet or particle-cloud representations into SPA-Net.

This would test whether pretrained jet foundation models improve:
- HH→4b sensitivity,
- ttbar and QCD rejection,
- Higgs-pair assignment,
- low-label performance,
- robustness to detector/reconstruction variations.

## Candidate embedding sources

### 1. ParT / Particle Transformer embeddings

Priority: high.

Reason:
- Particle Transformer is a strong supervised jet-tagging baseline.
- Public JetClass-based code/models exist.
- This is likely the easiest pretrained embedding baseline to integrate first.

Possible use:
- For each AK4 or AK8 jet, compute a fixed ParT-style embedding from jet constituents.
- Concatenate/project the embedding into the SPA-Net jet token.
- Compare SPA-Net with and without embeddings.

### 2. JP-JEPA embeddings

Priority: high, after ParT feasibility is established.

Reason:
- JP-JEPA is a very new self-supervised method.
- It is directly motivated by robust, data-efficient jet representation learning.
- It may be especially interesting for low-label HH→4b studies.

Possible use:
- Use frozen JP-JEPA particle-cloud embeddings per jet.
- Test whether SPA-Net + JP-JEPA improves performance with reduced labeled training data.
- Compare robustness to missing constituents or detector-level variations.

### 3. GloParT / other global particle-transformer embeddings

Priority: optional until code/checkpoints are verified.

Reason:
- Potentially novel, but only useful if the implementation and pretrained checkpoints are available and compatible with the current data format.

## Minimal publishable embedding experiment

The first realistic embedding experiment should not try every model. It should test:

1. Baseline two-head SPA-Net using only jet kinematics and b-tag proxy.
2. SPA-Net + frozen ParT embeddings.
3. SPA-Net + frozen JP-JEPA embeddings, if available.

For each:
- full-label training,
- 50% label training,
- 25% label training,
- 10% label training.

Main metrics:
- weighted AUC vs QCD-like backgrounds,
- weighted AUC vs ttbar,
- stable S/sqrt(B),
- number of selected background rows,
- effective background statistics,
- Higgs assignment accuracy on truth-matched signal,
- ZZ/ZH mass-plane validation performance.

## Integration design

For each event:
- Keep up to N AK4 jets and M AK8 FatJets.
- For each jet, keep standard SPA-Net features:
  - pt, eta, sin(phi), cos(phi), mass, b-tag proxy.
- Add optional embedding vector:
  - AK4 embedding: shape [events, N_AK4, D]
  - AK8 embedding: shape [events, N_AK8, D]

Model variants:
- no embeddings,
- embeddings concatenated to jet tokens,
- embeddings projected separately and added to jet-token representation,
- frozen embeddings first, fine-tuning later only if feasible.

## Important caveat

The embedding study requires jet constituents or particle clouds. Before implementation, we must verify whether current Delphes ROOT files contain enough constituent information or whether we need to build embeddings from EFlow objects matched to jets.

## Paper positioning

The main paper should not be a model zoo. The clean structure is:

1. CMS-style AK4/AK8 HH→4b dataset and reconstruction.
2. Baseline model comparison.
3. Two-head SPA-Net analysis and stability.
4. ZZ/ZH→4b validation.
5. Pretrained jet-embedding extension:
   - ParT baseline,
   - JP-JEPA self-supervised extension,
   - low-label and robustness tests.

This gives a coherent physics analysis plus a genuinely novel ML contribution.
