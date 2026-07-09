# Signal and background normalization convention

Current convention for the provisional SM-normalized HH4b analysis:

- Signal samples are generated with MG5/Pythia8/Delphes and used for shapes and selection efficiencies.
- Signal yields are normalized externally to approximate SM HH cross sections times BR(H→bb)^2:
  - ggF HH→4b = 10.55 fb
  - VBF HH→4b = 0.587 fb
- Background samples are normalized using their MG5 generator cross sections.
- This is a Delphes-level MC baseline. The QCD multijet background normalization is therefore a simulation-level proxy, not a final data-driven experimental estimate.

MG5 cross-section checks showed that:
- MG5 is not globally returning cross sections near 1.
- ggF HH decay handling is internally consistent with the HEFT card's effective H→bb ratio.
- VBF HHjj decay handling is internally consistent with the SM card's effective H→bb ratio.
- However, the effective branching ratios and ggF HH production normalization in the cards differ from the official SM values, so external signal normalization is used provisionally.

Harvey confirmed on 2026-07-09 that signal should be normalized to SM HH cross sections times BR\(H→bb\)^2.
