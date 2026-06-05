import math
import numpy as np
from itertools import islice
from datasets import load_dataset

data_files = {
    "train": [
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000001.parquet",
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000002.parquet",
    ]
}

MAX_JETS = 12
N_EVENTS_TO_SCAN = None # Set to None to scan all events in the dataset

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi

def delta_r(eta1, phi1, eta2, phi2):
    return math.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)

def find_hbb_bquarks(event):
    candidates = []
    for i, pid in enumerate(event["FullReco_GenPart_PID"]):
        if abs(pid) == 5 and event["FullReco_GenPart_Status"][i] == 23:
            candidates.append(i)

    candidates = sorted(
        candidates,
        key=lambda i: event["FullReco_GenPart_PT"][i],
        reverse=True,
    )
    return candidates[:2]

def match_bquarks_to_jets(event, b_indices, max_dr=0.4):
    matches = []

    for bidx in b_indices:
        b_eta = event["FullReco_GenPart_Eta"][bidx]
        b_phi = event["FullReco_GenPart_Phi"][bidx]

        best_j = None
        best_dr = 999.0

        for j in range(len(event["FullReco_JetAK4_PT"])):
            dr = delta_r(
                b_eta,
                b_phi,
                event["FullReco_JetAK4_Eta"][j],
                event["FullReco_JetAK4_Phi"][j],
            )
            if dr < best_dr:
                best_dr = dr
                best_j = j

        if best_dr < max_dr:
            matches.append(best_j)
        else:
            matches.append(-1)

    return matches

def build_event_arrays(event):
    n_jets = min(len(event["FullReco_JetAK4_PT"]), MAX_JETS)

    jets = np.zeros((MAX_JETS, 6), dtype=np.float32)
    mask = np.zeros((MAX_JETS,), dtype=np.float32)

    for j in range(n_jets):
        jets[j, 0] = event["FullReco_JetAK4_PT"][j]
        jets[j, 1] = event["FullReco_JetAK4_Eta"][j]
        jets[j, 2] = event["FullReco_JetAK4_Phi"][j]
        jets[j, 3] = event["FullReco_JetAK4_Mass"][j]
        jets[j, 4] = event["FullReco_JetAK4_BTag"][j]
        jets[j, 5] = event["FullReco_JetAK4_BTagPhys"][j]
        mask[j] = 1.0

    return jets, mask

ds = load_dataset(
    "fastmachinelearning/collide-1m",
    data_files=data_files,
    split="train",
    streaming=True,
)

all_jets = []
all_masks = []
all_labels = []

n_scanned = 0
n_usable = 0

for event in islice(ds, N_EVENTS_TO_SCAN):
    n_scanned += 1

    if n_scanned % 1000 == 0:
        print(f"Scanned {n_scanned}; usable so far: {n_usable}", flush=True)

    b_indices = find_hbb_bquarks(event)
    if len(b_indices) != 2:
        continue

    matches = match_bquarks_to_jets(event, b_indices)

    if len(matches) != 2:
        continue

    if matches[0] < 0 or matches[1] < 0:
        continue

    if matches[0] == matches[1]:
        continue

    if matches[0] >= MAX_JETS or matches[1] >= MAX_JETS:
        continue

    jets, mask = build_event_arrays(event)

    all_jets.append(jets)
    all_masks.append(mask)
    all_labels.append(np.array(matches, dtype=np.int64))

    n_usable += 1

all_jets = np.stack(all_jets)
all_masks = np.stack(all_masks)
all_labels = np.stack(all_labels)

np.random.seed(42)
idx = np.random.permutation(len(all_jets))

all_jets = all_jets[idx]
all_masks = all_masks[idx]
all_labels = all_labels[idx]

n_total = len(all_jets)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

splits = {
    "train": slice(0, n_train),
    "val": slice(n_train, n_train + n_val),
    "test": slice(n_train + n_val, n_total),
}

import os
os.makedirs("outputs/hbb_npz", exist_ok=True)

for name, sl in splits.items():
    np.savez(
        f"outputs/hbb_npz/{name}.npz",
        jets=all_jets[sl],
        mask=all_masks[sl],
        labels=all_labels[sl],
    )

print("Scanned events:", n_scanned)
print("Usable matched events:", n_usable)
print("Saved total events:", n_total)
print("Train:", n_train)
print("Val:", n_val)
print("Test:", n_total - n_train - n_val)
print("Output directory: outputs/hbb_npz")
print("Jet features: pt, eta, phi, mass, btag, btagPhys")
print("Labels: two AK4 jet indices matched to H->bb b quarks")
