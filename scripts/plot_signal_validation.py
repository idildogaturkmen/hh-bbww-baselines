import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("outputs/plots/signal_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

NPZ_DIR = Path("outputs/hbb_npz")
M_H = 125.0

def load_all_splits():
    jets_list, mask_list, labels_list = [], [], []

    for split in ["train", "val", "test"]:
        data = np.load(NPZ_DIR / f"{split}.npz")
        jets_list.append(data["jets"])
        mask_list.append(data["mask"])
        labels_list.append(data["labels"])

    jets = np.concatenate(jets_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return jets, mask, labels

def invariant_mass(j1, j2):
    pt1, eta1, phi1, m1 = j1[:4]
    pt2, eta2, phi2, m2 = j2[:4]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(max(m2_tot, 0.0))

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi <= -np.pi:
        dphi += 2 * np.pi
    return dphi

def delta_r(j1, j2):
    eta1, phi1 = j1[1], j1[2]
    eta2, phi2 = j2[1], j2[2]
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

def save_hist(values, bins, xlabel, ylabel, title, filename):
    plt.figure()
    plt.hist(values, bins=bins, histtype="step", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=200)
    plt.close()

def main():
    jets, mask, labels = load_all_splits()

    n_events = jets.shape[0]
    n_jets = mask.sum(axis=1)

    all_real_jets = jets[mask > 0]

    leading_pt = []
    subleading_pt = []
    matched_btags = []
    matched_pts = []
    mbb_matched = []
    dr_matched = []

    for ev in range(n_events):
        real = np.where(mask[ev] > 0)[0]
        pts = jets[ev, real, 0]
        sorted_pts = np.sort(pts)[::-1]

        leading_pt.append(sorted_pts[0])
        if len(sorted_pts) > 1:
            subleading_pt.append(sorted_pts[1])

        i, j = labels[ev]
        ji = jets[ev, i]
        jj = jets[ev, j]

        matched_btags.extend([ji[4], jj[4]])
        matched_pts.extend([ji[0], jj[0]])
        mbb_matched.append(invariant_mass(ji, jj))
        dr_matched.append(delta_r(ji, jj))

    save_hist(
        n_jets,
        bins=np.arange(0.5, 13.5, 1),
        xlabel="Number of AK4 jets",
        ylabel="Events",
        title="AK4 jet multiplicity in usable HH→bbWW events",
        filename="jet_multiplicity.png",
    )

    save_hist(
        leading_pt,
        bins=50,
        xlabel="Leading AK4 jet $p_T$",
        ylabel="Events",
        title="Leading AK4 jet $p_T$",
        filename="leading_jet_pt.png",
    )

    save_hist(
        subleading_pt,
        bins=50,
        xlabel="Subleading AK4 jet $p_T$",
        ylabel="Events",
        title="Subleading AK4 jet $p_T$",
        filename="subleading_jet_pt.png",
    )

    save_hist(
        all_real_jets[:, 4],
        bins=50,
        xlabel="AK4 jet b-tag score",
        ylabel="Jets",
        title="B-tag scores of all real AK4 jets",
        filename="all_jet_btag.png",
    )

    save_hist(
        matched_btags,
        bins=50,
        xlabel="Matched H→bb jet b-tag score",
        ylabel="Jets",
        title="B-tag scores of matched H→bb AK4 jets",
        filename="matched_hbb_jet_btag.png",
    )

    save_hist(
        matched_pts,
        bins=50,
        xlabel="Matched H→bb jet $p_T$",
        ylabel="Jets",
        title="Matched H→bb AK4 jet $p_T$",
        filename="matched_hbb_jet_pt.png",
    )

    save_hist(
        mbb_matched,
        bins=60,
        xlabel="$m_{bb}$ of matched AK4 jets",
        ylabel="Events",
        title="Reconstructed $m_{bb}$ using matched H→bb jets",
        filename="matched_mbb.png",
    )

    save_hist(
        dr_matched,
        bins=50,
        xlabel="$\\Delta R$ between matched H→bb jets",
        ylabel="Events",
        title="$\\Delta R$ between matched H→bb AK4 jets",
        filename="matched_deltaR_bb.png",
    )

    print(f"Saved plots to {OUTDIR}")
    print(f"Number of usable events plotted: {n_events}")
    print(f"Mean jet multiplicity: {n_jets.mean():.3f}")
    print(f"Mean matched m_bb: {np.mean(mbb_matched):.3f}")
    print(f"Median matched m_bb: {np.median(mbb_matched):.3f}")
    print(f"Mean matched deltaR: {np.mean(dr_matched):.3f}")

if __name__ == "__main__":
    main()
