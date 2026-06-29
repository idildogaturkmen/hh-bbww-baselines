# Normalization assumptions

Current status: signal-only proxy normalization.

- Target luminosity: 138 fb^-1.
- Signal sample: HH_bbWW.
- Default signal assumption: ggF-only HH production at 13 TeV.
- Cross section used: sigma(ggF HH -> bbWW) = 7.79 fb.
- Calculation: 31.05 fb * 2 * 0.5824 * 0.2152.
- Alternative if the signal sample includes ggF+VBF: approximately 8.22 fb.
- Filter efficiency: assumed 1.0.
- k-factor: assumed 1.0.
- Generator weights: assumed unit weights.
- Normalization denominator: local skimmed HH_bbWW event count, not official sum of generated weights.
- W decays: sample treated as inclusive in WW decays; the one-lepton e/mu fraction is selected by reconstruction cuts, not multiplied separately.
- Backgrounds: not yet physics-normalized because per-sample cross sections/filter efficiencies/generated event counts are missing.
- Therefore: weighted signal yields are a sanity check; weighted S/B, S/(S+B), Asimov Z, and likelihood interpretations are not final.
