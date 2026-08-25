# Model Freeze

Public-facing identity uses symbolic labels and SHA256 hashes. Exact internal paths are retained only in `MODEL_FREEZE.tsv` and `SOURCE_PROVENANCE.tsv`; no binary model or checkpoint is copied into this package.

| public model | SHA256 | train / validation | seed | selection rule |
|---|---|---:|---:|---|
| Step2j BDT-K | `46591e1059374aae646675a121885bd3943e1f13d5fdb6b47844cbf9a0546035` | 8,377,425 / 2,096,600 | 0 | seed-0 convergence artifact; early stopping on validation logloss, 50 rounds, minimize, save_best; 6000-round cap reached; best iteration 5998 |
| Step2j BDT-KF | `1ba187760b831e9452df537958a32828c1816715134b218d37c1d2360508df33` | 8,377,425 / 2,096,600 | 0 | seed-0 convergence artifact; early stopping on validation logloss, 50 rounds, minimize, save_best; 6000-round cap reached; best iteration 5999 |
| SPA-Net 2M primary checkpoint | `dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d` | 2,000,000 / 400,000 | 0 | primary checkpoint maximizing validation_average_jet_accuracy; epoch 49, step 48800, score 0.491227924823761 |
| SPA-Net 10M primary checkpoint | `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5` | 10,000,000 / 400,000 | 0 | primary checkpoint maximizing validation_average_jet_accuracy; epoch 47, step 234336, score 0.4928354024887085 |

## Representations and configurations

### Step2j BDT-K

- Input: 52 float32 features: 10-slot pT/eta/phi/mass, mask, pre-truncation selected-jet count, HT; jets pT>30 GeV, |eta|<2.5, pT-sorted, zero-padded.
- Configuration: XGBoost 2.1.4 gbtree binary:logistic; QuantileDMatrix/DataIter max_bin=256; max_depth=6; eta=0.05; subsample=0.8; colsample_bytree=0.8; min_child_weight=1; lambda=1; alpha=0; tree_method=hist.

### Step2j BDT-KF

- Input: 82 float32 features: BDT-K 52 plus 10-slot Sophon AK4 probB/probC/probL; same jet selection, pT ordering, truncation and zero padding.
- Configuration: XGBoost 2.1.4 gbtree binary:logistic; QuantileDMatrix/DataIter max_bin=256; max_depth=6; eta=0.05; subsample=0.8; colsample_bytree=0.8; min_child_weight=1; lambda=1; alpha=0; tree_method=hist.

### SPA-Net 2M primary checkpoint

- Input: 10 pT-sorted jets after pT>30 GeV and |eta|<2.5; float32 (pT,eta,phi,mass,probB,probC,probL), zero padding and Boolean mask; partial_events=True; score=P(signal).
- Configuration: SPA-Net commit debbdc999bfb785eb110a36c5fd3eff211ebf234; hidden=32, transformer=128, initial embedding=16, 8 encoder, 2 branch-encoder, 1 classification layer, 4 heads, dropout=0.0059; AdamW lr=0.00659, l2=0.000374, gradient clip=0.425, 50 epochs.

### SPA-Net 10M primary checkpoint

- Input: 10 pT-sorted jets after pT>30 GeV and |eta|<2.5; float32 (pT,eta,phi,mass,probB,probC,probL), zero padding and Boolean mask; partial_events=True; score=P(signal).
- Configuration: SPA-Net commit debbdc999bfb785eb110a36c5fd3eff211ebf234; hidden=32, transformer=128, initial embedding=16, 8 encoder, 2 branch-encoder, 1 classification layer, 4 heads, dropout=0.0059; AdamW lr=0.00659, l2=0.000374, gradient clip=0.425, 50 epochs.

