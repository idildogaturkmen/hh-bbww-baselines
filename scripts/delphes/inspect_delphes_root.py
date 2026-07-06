#!/usr/bin/env python3
import argparse
import uproot

def main():
    parser = argparse.ArgumentParser(description="Inspect a Delphes ROOT file.")
    parser.add_argument("root_file")
    args = parser.parse_args()

    with uproot.open(args.root_file) as f:
        print(f"FILE: {args.root_file}")

        print("\nKEYS:")
        for key in f.keys():
            print(f"  {key}")

        if "Delphes" not in f:
            raise SystemExit("ERROR: no Delphes tree found")

        tree = f["Delphes"]
        print(f"\nTREE: Delphes")
        print(f"ENTRIES: {tree.num_entries}")

        print("\nBRANCHES:")
        for name in tree.keys():
            print(f"  {name}")

        print("\nREQUIRED PREFIX CHECK:")
        for prefix in ["Event", "Jet", "GenParticle", "MissingET", "Electron", "Muon"]:
            ok = any(name == prefix or name.startswith(prefix + ".") for name in tree.keys())
            print(f"  {prefix}: {'OK' if ok else 'MISSING'}")

if __name__ == "__main__":
    main()
