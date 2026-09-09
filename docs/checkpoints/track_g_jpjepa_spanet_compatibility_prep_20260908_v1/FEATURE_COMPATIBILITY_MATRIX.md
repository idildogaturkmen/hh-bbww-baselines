# Feature Compatibility Matrix: Yang-Li Resolved-AK4 Ntuples -> JP-JEPA Inputs

Status: AUDITED AGAINST REAL FILES (not just documentation)
Files opened directly (read-only, via `uproot` over XRootD from LPC,
`root://cceos.ihep.ac.cn/`), this task:

| Process | File | Entries |
|---|---|---|
| QCD | `.../QCD_DelphesHH4JTrig_merged_ntuple/QCD_DelphesHH4JTrig_ntuple_mergeid166.root` | 156,976 |
| Signal (HH4b, 2HDM H3VAR H1H2 40-200) | `.../HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple/HH4b_2HDM_H3VAR_H1H2_40to200_ntuple_id0-9.root` | 140,035 |
| ttbar | `.../TTbar_ntuple/selected_ntuples_0000.root` | 57,999 |

Sampled 200-300 events per file for this audit (not the full file --
this is a compatibility audit, not the production run). All three
files were reachable and readable from this session at the time of the
audit (this LPC environment currently has a valid X.509 proxy;
contrast with the note in `STATUS.md` about a *separate*, currently
auth-failing, full-scale production attempt from the EAF environment
against the same IHEP endpoint -- these are independent facts about
different environments, not a contradiction).

**A note on relationship to prior work**: this project's EOS namespace
already contains a substantial prior effort
(`/eos/uscms/store/user/iturkmen/hh4b_delphes/part_jpjepa_dual_production_bundle_20260825_v2/`,
dated 2026-08-25, and a currently-active full-2M attempt dated
2026-09-08) that reads these exact same Yang-Li branches for a
dual ParT+JP-JEPA production. This audit was performed independently
(fresh `uproot` reads against the raw files, not by reading that
package's summaries), and is used below to **cross-check**, not
replace, this project's own conclusions -- see `STATUS.md` for the
full account of that discovery and how it was handled.

## 1. Full compatibility table

| JP-JEPA quantity | Required semantics | Yang-Li source branch(es) | Reconstruction formula | Status |
|---|---|---|---|---|
| `part_pt_log` | `log(constituent pT)` | `part_px`, `part_py` | `pt = hypot(px,py)`; `pt_log = log(pt)`; then standardize `clip((x-1.7)*0.7,-5,5)` | **EXACT** (verified: native `part_pt` branch equals `hypot(part_px,part_py)` to float32 precision on real data, max abs diff 0.0 in our sample) |
| `part_e_log` | `log(constituent E)` | `part_energy` | `log(part_energy)`; standardize `(x-2.0)*0.7`, clip | **EXACT** |
| `part_logptrel` | `log(pT_constituent / pT_jet)` | `part_px`,`part_py` + `jet_pt` | `log(pt / jet_pt[assigned AK4 jet])`; standardize | **DERIVABLE** -- needs the constituent-to-AK4-jet assignment (`part_label`), which exists and works (see ordering/assignment row below) |
| `part_logerel` | `log(E_constituent / E_jet)` | `part_energy` + `jet_energy` | `log(part_energy / jet_energy[assigned jet])`; standardize | **DERIVABLE**, same assignment dependency |
| `part_deltaR` | `hypot(deta, dphi)` of constituent from **its own AK4 jet axis** | `part_eta`,`part_phi` + `jet_eta`,`jet_phi` (assigned jet) | see deta/dphi row below; `deltaR = hypot(deta,dphi)`, standardize `(x-0.2)*4.0`, clip | **DERIVABLE**, contingent on deta/dphi row |
| `part_charge` | constituent electric charge, `{-1,0,+1}` | `part_charge` | direct passthrough, no transform | **EXACT** -- verified only `{-1,0,+1}` present in real data |
| `part_isCHad` | is charged hadron (not e/mu/photon, charge != 0) | `part_pid`, `part_charge` | `apid=abs(pid); isE=apid==11; isMu=apid==13; isPh=pid==22; rest=~(isE\|isMu\|isPh); isCHad=rest & (charge!=0)` | **DERIVABLE** -- Yang-Li has no direct `part_isChargedHadron` branch (unlike the JetClass ROOT schema JP-JEPA's `dataset.py` reads, which has this as a stored branch); verified on real data: standard PDG-code categorization (pid in {±211,±321,±2212,0,...} for hadrons) covers **100% of particles with zero uncategorized leftover** in our sample (36,683 constituents, QCD file) |
| `part_isNHad` | is neutral hadron | `part_pid`, `part_charge` | same function, `isNHad = rest & (charge==0)` | **DERIVABLE**, same verification |
| `part_isPhoton` | `pid == 22` | `part_pid` | direct | **DERIVABLE** (trivial, verified nonzero population: 13,077/36,683 in sample) |
| `part_isElectron` | `abs(pid) == 11` | `part_pid` | direct | **DERIVABLE**, verified |
| `part_isMuon` | `abs(pid) == 13` | `part_pid` | direct | **DERIVABLE**, verified |
| `part_d0` | `tanh(d0val)` | `part_d0val` | `tanh(part_d0val)`, no unit conversion (matches upstream's own no-conversion treatment) | **DERIVABLE**, semantics scrutinized separately -- see section 3 |
| `part_d0err` | `clip(d0err, 0, 1)` | `part_d0err` | direct clip | **EXACT** branch, standardization matches upstream |
| `part_dz` | `tanh(dzval)` | `part_dzval` | `tanh(part_dzval)` | **DERIVABLE**, see section 3 |
| `part_dzerr` | `clip(dzerr, 0, 1)` | `part_dzerr` | direct clip | **EXACT** branch |
| `part_deta` | eta of constituent **relative to its own AK4 jet axis** | `part_eta` + `jet_eta` (assigned jet, via `part_label`) | `part_eta[c] - jet_eta[part_label[c]]` | **DERIVABLE -- do NOT use the native `part_deta` branch** (see critical finding, section 2) |
| `part_dphi` | phi of constituent relative to its own AK4 jet axis, wrapped to `(-pi,pi]` | `part_phi` + `jet_phi` (assigned jet) | `wrap(part_phi[c] - jet_phi[part_label[c]])` | **DERIVABLE -- do NOT use the native `part_dphi` branch** (section 2) |
| Lorentz vector `(px,py,pz,E)` | 4-momentum per constituent | `part_px`,`part_py`,`part_pz`,`part_energy` | direct passthrough | **EXACT** -- all 4 branches present and directly usable, no transform |
| `part_mask` | 1 for real constituent, 0 for padding | (derived, not a branch) | `1` for every constituent assigned to a kept jet slot, `0` for padded slots beyond the real count | **EXACT construction**, same as upstream (`ak.ones_like` then pad-with-0) |
| Constituent-to-jet assignment | which AK4 jet each constituent belongs to | `part_label` | integer index into the raw (unsorted) `jet_*` arrays; `-1` = unassigned to any AK4 jet | **EXACT** -- verified: `part_label` values are `{-1, 0, 1, ..., n_jets-1}` per event, consistent with raw jet-array indexing |
| Constituent ordering | order fed into the transformer | (implicit: ROOT branch storage order) | no sort applied by either upstream JetClass `dataset.py` or this project's own extraction; see section 4 | **DERIVABLE / architecturally irrelevant** -- see section 4 |
| `maxlen` | 128 constituents/jet, zero-pad/truncate | (derived) | pad-or-truncate per AK4 jet's constituent list to 128 | **EXACT match, truncation essentially never triggers for AK4** -- see section 5 |
| Jet-level `probB`/`probC`/`probL` (native SPA-Net features, NOT a JP-JEPA input) | per-jet b/c/light tagger scores | `jet_sophonAK4_probB/C/L` | direct | **EXACT** (unrelated to JP-JEPA; recorded because the native 7-feature SPA-Net contract uses these, per `docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`) |

**No row is MISSING or unresolvably AMBIGUOUS.** Per the task's explicit
critical rule, if any row had come back MISSING, this document would
stop here and report the gap rather than proceed. Two rows required
correcting a naive branch-name-based assumption (`part_deta`/`part_dphi`,
section 2) -- these are DERIVABLE using a different, verified formula,
not silently invented.

## 2. Critical finding: the native `part_deta`/`part_dphi`/`part_dr` branches are NOT relative to the AK4 jet axis

Verified directly, three ways, on real data (QCD, signal, and ttbar
files):

1. **Self-consistency check**: `part_dr` (native) equals
   `hypot(part_deta, part_dphi)` (native) to float32 rounding
   (`2.4e-7` max abs diff) -- so `part_deta`/`part_dphi`/`part_dr` are
   an internally consistent triplet.
2. **But they do not match any AK4-jet-relative reconstruction.** For
   a representative QCD event (event 203, AK4 jet 0, `jet_pt = 131.5`
   GeV, 13 constituents assigned via `part_label`), native `part_deta`
   values are `~3.49-3.64`, while `part_eta[constituent] -
   jet_eta[jet 0]` gives `~-0.05` to `+0.09` -- a difference of `~3.54`,
   far outside any float-precision or convention (e.g. sign) explanation.
   The **same** offset (`~3.54` for this event) applies identically to
   **every constituent in the event**, including the 28 constituents
   with `part_label == -1` (not assigned to any AK4 jet at all) and
   constituents belonging to *other* AK4 jets in the same event. This
   was checked against the reconstructed fat-jet (`fj_eta`/`fj_phi`,
   via `part_fjlabel`) axis too -- no better match (`~3.55` residual).
3. **The offset is an event-level constant, uncorrelated with jet or
   fat-jet kinematics.** Scanning 15 consecutive QCD events: the
   `part_deta[c] - part_eta[c]` difference is constant across every
   constituent in an event (std `~1e-7`) but varies **event-to-event**
   from `-3.46` to `+6.60`, including events with **zero** fat jets
   (`n_fj = 0`) where a fat-jet-relative interpretation is not even
   possible, yet the offset is still present and nonzero. Reproduced
   independently on the signal (`HH4b`, offset `2.94` for event 0) and
   ttbar (offset `3.61` for event 0) files.

**Conclusion: whatever `part_deta`/`part_dphi`/`part_dr` encode in
this ntuple schema, it is not "displacement of the constituent from
its reconstructed AK4 jet axis" -- the quantity JP-JEPA's `pf_features`
and `pf_points` require.** Using these native branches directly, as
their name would naively suggest by analogy to the JetClass schema
JP-JEPA's own `dataset.py` was written against, would silently feed
the model garbage (angular displacements of several radians, an order
of magnitude larger than even a large-R jet cone, let alone an
AK4/R=0.4 cone) for every single constituent. **This is exactly the
kind of silent-corruption risk the task's critical rule warns against,
and it is now caught and documented rather than shipped.**

The correct, verified-sane reconstruction (`part_eta[c] -
jet_eta[part_label[c]]`, and wrapped `part_phi[c] - jet_phi[part_label[c]]`)
gives residuals of a few hundredths of a radian for constituents of a
131 GeV AK4 jet -- physically exactly what is expected for an R=0.4
jet. **This is the formula this project's existing (2026-08-25) ParT+
JP-JEPA extraction code (`extract_shard.py`,
`part_jpjepa_dual_production_bundle_20260825_v2`) already independently
implements** -- it never reads the native `part_deta`/`part_dphi`
branches at all (see its `PART_BRANCHES` list, which omits them
entirely) and instead derives them exactly as verified correct here.
This audit did not know that when the cross-check was designed; the
agreement is an independent corroboration of that script's choice, not
a copy of its logic. **This finding is worth surfacing explicitly to
whoever maintains that pipeline**, since its own code comments do not
record *why* the native branches are unused -- a future maintainer
tempted to "simplify" by reading `part_deta` directly (since it exists
and has a tempting name) would silently reintroduce this bug.

No hypothesis for what `part_deta`/`part_dphi` *do* represent was
confirmed (checked and ruled out: absolute eta/phi, AK4-jet-relative,
fat-jet-relative for the assigned fat jet or the event's other fat
jets; a generator-level Higgs or dihiggs-axis relation was considered
but not tested in detail since the QCD/ttbar files must define it some
other way as they have no genuine Higgs). This is left as an open,
unresolved question about the Yang-Li ntuple's internal schema -- it
does not block this project's use, since the required quantity is
independently and correctly reconstructable from `part_eta`/`part_phi`/
`jet_eta`/`jet_phi`/`part_label` (all independently verified present
and semantically as expected), but it is recorded honestly here rather
than silently ignored.

## 3. Track-displacement (`d0`/`dz`) scrutiny (Task 7)

Verified directly on real data (QCD file, 36,683 constituents sampled):

- **Neutral particles get an exact sentinel `0.0`** for both
  `part_d0val` and `part_dzval` (`frac_zero_exact = 1.0` for the
  20,506 neutral constituents in the sample) -- consistent with the
  physical expectation that impact parameters are only meaningful for
  charged tracks.
- **Charged particles have a plausible core distribution with a heavy
  tail**: median `d0val ≈ -0.0002`, 1st/99th percentile `≈ ±27-28`; but
  the full range extends to `-746` / `+708` (`d0val`) and `-872` /
  `+1099` (`dzval`). This kind of heavy tail is a known artifact of
  Delphes fast-simulation track-parameter smearing for very-low-pT or
  very-forward tracks (the smearing model's resolution formula can
  produce large values for poorly-constrained helices) -- it is not
  unique to this dataset; JetClass's own Delphes-based tracks have the
  same qualitative issue, which is precisely why upstream applies
  `tanh(...)` rather than a linear standardization to `part_d0`/`part_dz`.
- **Consequence, quantified**: `2.5%` of sampled `d0val` and `3.7%` of
  sampled `dzval` values have `|value| > 3`, i.e. are driven to
  `|tanh(...)| > 0.995` -- solidly saturated. This is a real, non-negligible
  fraction, not a rounding curiosity.
- `part_d0err`/`part_dzerr` are in a physically sane range (`[0, 0.70]`
  and `[0, 2.17]` respectively in-sample) and the upstream `clip(.,0,1)`
  is **not a no-op**: the 99th percentile of `dzerr` (`1.77`) already
  exceeds the clip ceiling, so a real, non-negligible tail of tracks is
  clipped.
- **Unit provenance**: this task did **not** independently confirm the
  physical unit (cm vs. mm vs. something else) of `part_d0val`/
  `part_dzval` against a Yang-Li/Delphes card or ntuplizer source --
  only their numerical behavior, which is internally consistent with a
  typical Delphes track-smearing output. Per this project's own
  extraction code's documented policy (`extract_shard.py`: "no unit
  conversion, exactly as both the pinned JP-JEPA repo's dataset.py and
  the official JetClass_full.yaml apply it") and per the explicit task
  instruction not to invent transformations, **this document recommends
  the same policy: apply upstream's `tanh(raw value)` with no rescaling**,
  and record the resulting saturation fraction (above) as a
  domain-characterization fact (see `STATUS.md` domain-shift section),
  not as an error to "fix" by inventing a rescaling upstream never
  specified.
- **Verdict: track-displacement inputs are DERIVABLE with correct,
  present semantics** (an actual per-track impact parameter and its
  Delphes-estimated uncertainty exist and are populated with a
  physically sane charged/neutral split) -- this is not a MISSING or
  fabricated input.

## 4. Constituent ordering audit (Task 8)

- **The Yang-Li ntuple provides constituents in ROOT-branch storage
  order** (whatever order the ntuplizer wrote `part_*` arrays in for
  each event) -- there is no explicit pT-sort applied anywhere in this
  project's extraction path, and this audit did not find one in the
  ntuplizer's output either (no evidence of monotonic pT ordering was
  required or checked, since it turns out not to matter -- next point).
- **Upstream JetClass's own `dataset.py` likewise applies no
  constituent sort** (verified in `PREPROCESSING_SPEC.md` section 4 --
  no `argsort` anywhere in `_load_awkward_array`/`_preprocess`/
  `_finalize_inputs`). So "whatever order the source file provides" is
  upstream's own convention too, not a Yang-Li-specific deviation.
- **This project does not need to prove the two orderings match**,
  because the extraction point being used (`EXTRACTION_POINT_PROOF.md`)
  is **provably invariant to constituent permutation** applied
  consistently across `pf_features`/`pf_vectors`/`pf_mask`:
  `ParticleTransformer`'s constituent-facing modules (`Embed`, the main
  self-attention `blocks`) act per-position with no positional encoding,
  and the physics-motivated pairwise attention bias (`PairEmbed`) is a
  function of each *pair's* own kinematics, not its index. The
  class-attention blocks that produce `jet_embedding` aggregate the
  full constituent sequence via attention (a weighted sum over all
  positions), which is invariant to how those positions are ordered,
  provided the same permutation is applied to `x`, `v`, and `mask`
  together (which every extraction path here does, since all three
  tensors are built from the same per-constituent boolean mask
  `part_label == oidx` in a single pass). The only operation that is
  order-sensitive is **padding**, which must (and does, verified in
  both this project's derivation and upstream's) place all padding
  **after** the real constituents, with `mask = 0` exactly there.
- **Verdict: ordering is not a compatibility risk for the frozen
  per-jet `jet_embedding`**, up to floating-point summation-order
  effects (round-off at the `~1e-6`-relative level or smaller, not a
  semantic difference). This closes Task 8 without requiring an
  unverifiable claim about upstream's internal jet-constituent storage
  convention.

## 5. Constituent multiplicity vs. `maxlen = 128` (relevant to Task 10)

Sampled QCD per-jet constituent multiplicity (50 events, 259 jets):
mean `16.8`, max observed `40`. **Zero jets exceeded `maxlen = 128`**
in this sample -- AK4 (R=0.4) jets are far sparser than the large-R,
often-boosted JetClass jets the padding budget of 128 was chosen for.
Practically: the vast majority (`>100` of 128) of every AK4 jet's
per-constituent tensor will be **padding** (`mask = 0`, zero-valued
features) when fed to JP-JEPA. This is recorded as a domain-shift fact
for `STATUS.md` (see Task 10 discussion there) -- it does not break
anything (the mask contract handles it correctly by construction), but
it does mean JP-JEPA's self-attention operates over a much sparser,
more heavily-padded sequence than it saw during pretraining, which is
exactly the kind of domain shift this package is instructed to
characterize honestly rather than paper over.

## 6. Process-level schema notes

`process_index` (native branch) distinguishes the three processes
audited: QCD `= 0`, signal (HH4b 2HDM) `= 100`, ttbar `= -1`. Branch
counts differ slightly per file (82/87/98 total branches for QCD/
signal/ttbar respectively) due to process-specific generator-level
branches (e.g. `gen_dihiggs_mass`, populated only for the signal file
in this sample), but **every branch this audit and this project's
extraction code actually require** (`jet_*`, `part_*`, per Section 1)
is present and populated in all three files checked.
