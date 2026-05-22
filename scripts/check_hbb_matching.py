import math
from itertools import islice
from datasets import load_dataset

data_files = {
    "train": [
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000001.parquet",
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000002.parquet",
    ]
}

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
    """
    Find candidate b quarks directly from the H->bb decay.
    In this dataset, useful prompt b quarks often have status 23 and PID ±5.
    We select the two highest-pT status-23 b quarks as a first practical label definition.
    """
    candidates = []
    pids = event["FullReco_GenPart_PID"]
    statuses = event["FullReco_GenPart_Status"]

    for i, pid in enumerate(pids):
        if abs(pid) == 5 and statuses[i] == 23:
            candidates.append(i)

    # Usually should be exactly two for H->bb in HH_bbWW.
    candidates = sorted(
        candidates,
        key=lambda i: event["FullReco_GenPart_PT"][i],
        reverse=True,
    )

    return candidates[:2]

def match_bquarks_to_jets(event, b_indices, max_dr=0.4):
    jet_matches = []

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
            jet_matches.append((bidx, best_j, best_dr))
        else:
            jet_matches.append((bidx, None, best_dr))

    return jet_matches

ds = load_dataset(
    "fastmachinelearning/collide-1m",
    data_files=data_files,
    split="train",
    streaming=True,
)

n = 0
n_two_b = 0
n_two_matched = 0
n_distinct_matched = 0

for iev, event in enumerate(islice(ds, 1000)):
    n += 1

    b_indices = find_hbb_bquarks(event)
    if len(b_indices) == 2:
        n_two_b += 1

    matches = match_bquarks_to_jets(event, b_indices)

    matched_jets = [m[1] for m in matches if m[1] is not None]
    if len(matched_jets) == 2:
        n_two_matched += 1
        if len(set(matched_jets)) == 2:
            n_distinct_matched += 1

    if iev < 5:
        print("=" * 80)
        print("Event", iev)
        print("b quark gen indices:", b_indices)
        for bidx in b_indices:
            print(
                "  b gen", bidx,
                "PID=", event["FullReco_GenPart_PID"][bidx],
                "pt=", round(event["FullReco_GenPart_PT"][bidx], 2),
                "eta=", round(event["FullReco_GenPart_Eta"][bidx], 2),
                "phi=", round(event["FullReco_GenPart_Phi"][bidx], 2),
                "status=", event["FullReco_GenPart_Status"][bidx],
                "M1=", event["FullReco_GenPart_M1"][bidx],
            )

        print("matches:")
        for bidx, jidx, dr in matches:
            print("  gen b", bidx, "-> jet", jidx, "dR=", round(dr, 3))
            if jidx is not None:
                print(
                    "     jet pt=", round(event["FullReco_JetAK4_PT"][jidx], 2),
                    "eta=", round(event["FullReco_JetAK4_Eta"][jidx], 2),
                    "phi=", round(event["FullReco_JetAK4_Phi"][jidx], 2),
                    "btag=", event["FullReco_JetAK4_BTag"][jidx],
                    "btagPhys=", event["FullReco_JetAK4_BTagPhys"][jidx],
                )

print("\nSummary")
print("events scanned:", n)
print("events with two status-23 b quarks:", n_two_b)
print("events with two matched b quarks:", n_two_matched)
print("events with two distinct matched AK4 jets:", n_distinct_matched)
