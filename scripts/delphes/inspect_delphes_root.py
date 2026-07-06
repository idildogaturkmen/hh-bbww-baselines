#!/usr/bin/env python3
import argparse
import uproot

def has_prefix(tree, prefix):
    return any(name == prefix or name.startswith(prefix + "/") or name.startswith(prefix + ".") for name in tree.keys())

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

        print("\nOBJECT CHECK:")
        for prefix in ["Event", "Particle", "Jet", "GenJet", "MissingET", "GenMissingET", "Electron", "Muon"]:
            print(f"  {prefix}: {'OK' if has_prefix(tree, prefix) else 'MISSING'}")

        print("\nSIZE BRANCHES:")
        for name in tree.keys():
            if name.endswith("_size"):
                print(f"  {name}")

if __name__ == "__main__":
    main()
