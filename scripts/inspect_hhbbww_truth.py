from datasets import load_dataset
from itertools import islice

data_files = {
    "train": [
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000001.parquet",
        "HH_bbWW/HH_bbWW-NEVENT10000-RS23000002.parquet",
    ]
}

ds = load_dataset(
    "fastmachinelearning/collide-1m",
    data_files=data_files,
    split="train",
    streaming=True,
)

interesting = {25, 24, 5}
for iev, event in enumerate(islice(ds, 5)):
    print("=" * 100)
    print(f"Event {iev}")
    print("n AK4 jets:", len(event["FullReco_JetAK4_PT"]))
    print("n GenPart:", len(event["FullReco_GenPart_PID"]))

    print("\nAK4 jets:")
    for j in range(min(len(event["FullReco_JetAK4_PT"]), 10)):
        print(
            f"  jet {j:2d}:",
            "pt=", round(event["FullReco_JetAK4_PT"][j], 2),
            "eta=", round(event["FullReco_JetAK4_Eta"][j], 2),
            "phi=", round(event["FullReco_JetAK4_Phi"][j], 2),
            "m=", round(event["FullReco_JetAK4_Mass"][j], 2),
            "btag=", round(event["FullReco_JetAK4_BTag"][j], 3),
            "btagPhys=", round(event["FullReco_JetAK4_BTagPhys"][j], 3),
)

    print("\nGen particles with |PID| in {25 Higgs, 24 W, 5 b}:")
    for i, pid in enumerate(event["FullReco_GenPart_PID"]):
        if abs(pid) in interesting:
            print(
                f"  gen {i:3d}:",
                "PID=", pid,
                "pt=", round(event["FullReco_GenPart_PT"][i], 2),
                "eta=", round(event["FullReco_GenPart_Eta"][i], 2),
                "phi=", round(event["FullReco_GenPart_Phi"][i], 2),
                "M1=", event["FullReco_GenPart_M1"][i],
                "M2=", event["FullReco_GenPart_M2"][i],
                "D1=", event["FullReco_GenPart_D1"][i],
                "D2=", event["FullReco_GenPart_D2"][i],
                "status=", event["FullReco_GenPart_Status"][i],
            )
