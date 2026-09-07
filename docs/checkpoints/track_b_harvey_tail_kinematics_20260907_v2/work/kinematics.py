"""Shared kinematics formulas, reused byte-identical from this project's
canonical common.py (track_b_harvey_followup_finescan_likelihood_20260829_v1/work/common.py)
so every derived quantity here (mH1/mH2/RHH/etc.) is directly comparable
to every prior package's numbers, not a re-derivation."""
import numpy as np

PAIRINGS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
BTAG_LOOSE_PROBB = 0.0243  # Yang & Li arXiv:2508.15048v2, Appendix B.2: loose SophonAK4 WP, eps_B(light)=10%


def four_vectors(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + mass ** 2)
    return px, py, pz, e


def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta ** 2 + dphi ** 2)


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def gather1d(arr2d, idx1d):
    return arr2d[np.arange(arr2d.shape[0]), idx1d]


def leading_four_topology(pt, eta, phi, mass):
    """pt/eta/phi/mass: (n,10) candidate-selected, descending-pT sorted,
    zero-padded. Slots 0..3 (leading four). argmin-R_HH over 3 pairings --
    identical convention to every prior package in this project."""
    n = pt.shape[0]
    px, py, pz, e = four_vectors(pt, eta, phi, mass)
    pairing_mbb = np.zeros((n, 3, 2), dtype=np.float64)
    pairing_dRbb = np.zeros((n, 3, 2), dtype=np.float64)
    pairing_ptbb = np.zeros((n, 3, 2), dtype=np.float64)
    pairing_dR_HH = np.zeros((n, 3), dtype=np.float64)
    pairing_dEta_HH = np.zeros((n, 3), dtype=np.float64)
    pairing_dPhi_HH = np.zeros((n, 3), dtype=np.float64)
    pairing_pt_HH = np.zeros((n, 3), dtype=np.float64)
    pairing_eta_HH = np.zeros((n, 3), dtype=np.float64)

    for k, (a, b, c, d) in enumerate(PAIRINGS):
        pxA, pyA, pzA, eA = px[:, a] + px[:, b], py[:, a] + py[:, b], pz[:, a] + pz[:, b], e[:, a] + e[:, b]
        pxB, pyB, pzB, eB = px[:, c] + px[:, d], py[:, c] + py[:, d], pz[:, c] + pz[:, d], e[:, c] + e[:, d]
        mA = np.sqrt(np.clip(eA ** 2 - (pxA ** 2 + pyA ** 2 + pzA ** 2), 0, None))
        mB = np.sqrt(np.clip(eB ** 2 - (pxB ** 2 + pyB ** 2 + pzB ** 2), 0, None))
        pairing_mbb[:, k, 0], pairing_mbb[:, k, 1] = mA, mB
        pairing_dRbb[:, k, 0] = delta_r(eta[:, a], phi[:, a], eta[:, b], phi[:, b])
        pairing_dRbb[:, k, 1] = delta_r(eta[:, c], phi[:, c], eta[:, d], phi[:, d])
        pairing_ptbb[:, k, 0] = np.sqrt(pxA ** 2 + pyA ** 2)
        pairing_ptbb[:, k, 1] = np.sqrt(pxB ** 2 + pyB ** 2)
        etaA = np.arcsinh(pzA / np.clip(np.sqrt(pxA ** 2 + pyA ** 2), 1e-9, None))
        etaB = np.arcsinh(pzB / np.clip(np.sqrt(pxB ** 2 + pyB ** 2), 1e-9, None))
        phiA, phiB = np.arctan2(pyA, pxA), np.arctan2(pyB, pxB)
        pairing_dR_HH[:, k] = delta_r(etaA, phiA, etaB, phiB)
        pairing_dEta_HH[:, k] = etaA - etaB
        pairing_dPhi_HH[:, k] = delta_phi(phiA, phiB)
        pxHH, pyHH = pxA + pxB, pyA + pyB
        pairing_pt_HH[:, k] = np.sqrt(pxHH ** 2 + pyHH ** 2)
        pzHH, eHH = pzA + pzB, eA + eB
        pairing_eta_HH[:, k] = np.arcsinh(pzHH / np.clip(np.sqrt(pxHH ** 2 + pyHH ** 2), 1e-9, None))

    pairing_RHH = np.sqrt((pairing_mbb[:, :, 0] - 125.0) ** 2 + (pairing_mbb[:, :, 1] - 125.0) ** 2)
    denom = np.clip(pairing_mbb[:, :, 0] + pairing_mbb[:, :, 1], 1e-9, None)
    pairing_mass_asym = np.abs(pairing_mbb[:, :, 0] - pairing_mbb[:, :, 1]) / denom

    pxT = px[:, 0] + px[:, 1] + px[:, 2] + px[:, 3]
    pyT = py[:, 0] + py[:, 1] + py[:, 2] + py[:, 3]
    pzT = pz[:, 0] + pz[:, 1] + pz[:, 2] + pz[:, 3]
    eT = e[:, 0] + e[:, 1] + e[:, 2] + e[:, 3]
    mHH_leading_four = np.sqrt(np.clip(eT ** 2 - (pxT ** 2 + pyT ** 2 + pzT ** 2), 0, None))

    fixed_idx = np.argmin(pairing_RHH, axis=1)
    mH1 = gather1d(pairing_mbb[:, :, 0], fixed_idx)
    mH2 = gather1d(pairing_mbb[:, :, 1], fixed_idx)
    dRbb1 = gather1d(pairing_dRbb[:, :, 0], fixed_idx)
    dRbb2 = gather1d(pairing_dRbb[:, :, 1], fixed_idx)
    ptH1 = gather1d(pairing_ptbb[:, :, 0], fixed_idx)
    ptH2 = gather1d(pairing_ptbb[:, :, 1], fixed_idx)
    RHH = gather1d(pairing_RHH, fixed_idx)
    mass_asym = gather1d(pairing_mass_asym, fixed_idx)
    deltaR_HH = gather1d(pairing_dR_HH, fixed_idx)
    deltaEta_HH = gather1d(pairing_dEta_HH, fixed_idx)
    deltaPhi_HH = gather1d(pairing_dPhi_HH, fixed_idx)
    pT_HH = gather1d(pairing_pt_HH, fixed_idx)
    eta_HH = gather1d(pairing_eta_HH, fixed_idx)
    min_dRbb = np.minimum(dRbb1, dRbb2)
    max_dRbb = np.maximum(dRbb1, dRbb2)
    pt_asym = np.abs(ptH1 - ptH2) / np.clip(ptH1 + ptH2, 1e-9, None)

    return dict(mH1=mH1, mH2=mH2, RHH=RHH, mass_asym=mass_asym, dRbb1=dRbb1, dRbb2=dRbb2,
                min_dRbb=min_dRbb, max_dRbb=max_dRbb, ptH1=ptH1, ptH2=ptH2,
                mHH_leading_four=mHH_leading_four, deltaR_HH=deltaR_HH,
                deltaEta_HH=deltaEta_HH, deltaPhi_HH=deltaPhi_HH,
                pT_HH=pT_HH, eta_HH=eta_HH, pt_asym=pt_asym,
                fixed_pairing_index=fixed_idx)


def native_select_and_order(jet_pt, jet_eta, jet_phi, jet_mass, jet_probB, jet_probC, jet_probL,
                             pt_min=30.0, abs_eta_max=2.5, n_slots=10):
    """Byte-identical selection/ordering convention to native_inputs() in
    run_spanet_physical_scan.py: pt>30, |eta|<2.5, descending pT, zero-pad
    to n_slots. Inputs are awkward arrays (jagged, one row per event)."""
    import awkward as ak
    sel = (jet_pt > pt_min) & (np.abs(jet_eta) < abs_eta_max)
    arrays = [jet_pt, jet_eta, jet_phi, jet_mass, jet_probB, jet_probC, jet_probL]
    selected = [a[sel] for a in arrays]
    order = ak.argsort(selected[0], axis=1, ascending=False)
    selected = [a[order] for a in selected]
    n_selected = ak.to_numpy(ak.num(selected[0], axis=1))
    padded = [ak.to_numpy(ak.fill_none(ak.pad_none(a[:, :n_slots], n_slots, clip=True), 0.0)).astype(np.float64)
              for a in selected]
    return (*padded, n_selected)
