# Two-BDT SM-normalized HH4b category baseline

This study uses two dedicated BDTs:

- BDT_QCD: trained to separate SM-normalized HH signal from QCD bbbb HT-sliced backgrounds.
- BDT_top: trained to separate SM-normalized HH signal from ttbar/top-like backgrounds.

The goal is to test a CMS-inspired two-discriminant category strategy rather than relying on a single global BDT. The outputs include:
- AUC for each BDT.
- A 2D rectangular threshold scan.
- Stable threshold scan requiring minimum background statistics.
- Fixed non-overlapping category yields and background composition.

Signal normalization follows SM HH cross sections times BR(H->bb)^2.
Backgrounds use generator cross-section weights.
