# Frozen baseline: BDT-v3 qcdplus, 2026-07-13

Purpose:
This is the frozen BDT-v3 qcdplus baseline for the HH→4b Delphes analysis.

Main change relative to BDT-v2:
The QCD iHT200–400 and iHT400–600 samples were enlarged and combined into BDT-v2-ready candidate parquets.

Samples:
- ggF HH→4b SM-normalized
- VBF HH→4b SM-normalized
- ttbar
- Zbbbb
- QCD bbbb iHT100–200
- QCD bbbb iHT200–400 combined220k
- QCD bbbb iHT400–600 combined120k
- QCD bbbb iHT600+

Important sample sizes:
- QCD iHT200–400 combined220k: 220,000 generated events and 15,038 candidate rows
- QCD iHT400–600 combined120k: 120,000 generated events and 17,761 candidate rows

Feature policy:
The BDT uses reconstructed candidate and event features only.
Truth-level jet flavor, raw indices, selected indices, event IDs, and source identifiers are excluded from the feature list.

Main BDT-v3 qcdplus result:
- QCD classifier unweighted AUC: about 0.8625
- QCD classifier physics-weighted AUC: about 0.8355
- Top classifier physics-weighted AUC: about 0.8090
- Best supported rectangle in the fixed-weight run: approximately qcd_threshold = 0.800 and top_threshold = 0.500
- Signal at 450/fb: about 92.1 events
- Background at 450/fb: about 214,886 events
- S/B: about 4.29e-4
- S/sqrt(B): about 0.199
- Effective background statistics: about 127

Caveat:
This baseline currently saves summary tables only. For categorized BDT studies, mass sculpting, and background composition, the next version should additionally save the scored test dataframe with qcd_score and top_score.
