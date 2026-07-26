# CMS-inspired resolved HH→4b mHH category and feature contract

Status: `hh4b_bdt_v2_cms_inspired_category_and_feature_contract_pass`.

This checkpoint freezes a **CMS-inspired resolved HH→4b mHH categorization**.
It is not a reproduction of a CMS analysis.  The canonical global BDT v1 is
preserved unchanged as the uncategorized reference.

Every one of the 57326 canonical train candidates is assigned
exactly once: `low_mhh` uses $m_{HH} < 450$ GeV and `high_mhh` uses
$m_{HH} \geq 450$ GeV.  The boundary was fixed a priori and was not
optimized.  Both ggF $HH$ and VBF $HH$ remain in these reconstructed-$m_{HH}$
categories.  A VBF production-mode BDT is deferred because the candidate
schema does not contain a frozen complete representation of two additional
VBF tagging jets.

The exact 34-feature mass-aware BDT-v1 order is retained.  Eighteen
detector-level features are appended from the four candidate-jet four-vectors.
The stored pairing, generator truth, `Jet.Flavor`, source metadata, absolute
azimuths, and binary b-tag flags are not nominal features.  No continuous
b-tag score, b-jet energy regression, or per-jet resolution is claimed.

The Higgs pairs are reconstructed from the four-vectors by the frozen
125-GeV mass-ranking rule.  $H_1$ is the higher-$p_{\mathrm{T}}$ dijet,
and the leading jet in each candidate is its higher-$p_{\mathrm{T}}$
member.  Rest-frame angles use helicity axes defined by each parent candidate's
laboratory flight direction.  Cosines are audited without clipping.

The explicit ablation removes only `mbb1`, `mbb2`, `delta_mbb`, and
`r_hh_125_125`; it retains `mhh` and is therefore described as a
**categorized explicit dijet-mass-plane-blind ablation**, not as fully
mass-decorrelated.

Only train candidate Parquets were opened.  Validation candidate files opened:
0.  Test candidate files opened: 0.  Models trained: 0.  Predictions written:
0.  Hyperparameter trials: 0.

The next authorized gate is
`train_hh4b_bdt_v2_cms_inspired_categorized_grouped_cv`.
