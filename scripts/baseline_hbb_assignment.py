import itertools
import numpy as np
from pathlib import Path

M_H = 125.0

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

def pair_correct(pred, truth):
    return set(pred) == set(truth)

def top2_btag(jets, mask):
    real = np.where(mask > 0)[0]
    btags = jets[real, 4]
    return tuple(real[np.argsort(btags)[-2:]])

def closest_mass(jets, mask):
    real = np.where(mask > 0)[0]
    best_pair = None
    best_score = float("inf")

    for i, j in itertools.combinations(real, 2):
        mbb = invariant_mass(jets[i], jets[j])
        score = abs(mbb - M_H)
        if score < best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair

def combined_btag_mass(jets, mask):
    real = np.where(mask > 0)[0]
    best_pair = None
    best_score = float("inf")

    for i, j in itertools.combinations(real, 2):
        mbb = invariant_mass(jets[i], jets[j])
        btag_sum = jets[i, 4] + jets[j, 4]

        # Lower is better: close to Higgs mass, with reward for high b-tag.
        score = abs(mbb - M_H) - 20.0 * btag_sum

        if score < best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair

def evaluate(path):
    data = np.load(path)
    jets = data["jets"]
    mask = data["mask"]
    labels = data["labels"]

    methods = {
        "top2_btag": top2_btag,
        "closest_mass": closest_mass,
        "combined_btag_mass": combined_btag_mass,
    }

    correct = {name: 0 for name in methods}
    n = len(labels)

    for ev in range(n):
        truth = tuple(labels[ev])
        for name, fn in methods.items():
            pred = fn(jets[ev], mask[ev])
            if pair_correct(pred, truth):
                correct[name] += 1

    return n, {name: correct[name] / n for name in methods}

def main():
    outdir = Path("outputs/hbb_npz")

    print("H → bb AK4 jet-assignment baselines")
    print("----------------------------------")

    for split in ["train", "val", "test"]:
        n, acc = evaluate(outdir / f"{split}.npz")
        print(f"\n{split}: {n} events")
        for name, value in acc.items():
            print(f"  {name:20s}: {value:.4f}")

if __name__ == "__main__":
    main()
