#!/usr/bin/env python3
import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


KINDS = ("root", "parquet", "metadata", "logs")


def default_store() -> Path:
    return Path(os.environ.get("HH4B_STORE", f"/uscms_data/d3/{os.environ['USER']}/hh4b_delphes"))


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    dest = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"Refusing unsafe tar member path: {member.name}")
    tar.extractall(destination)


def copy_without_overwrite(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"SKIP existing: {dst}")
        return False
    shutil.copy2(src, dst)
    print(f"COPIED: {src} -> {dst}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect transferred HH4b Condor shard tarballs into the HH4B store."
    )
    parser.add_argument("--store", default=str(default_store()), help="Submit-side HH4B store")
    parser.add_argument("--return-dir", default=None, help="Directory containing returned shard tarballs")
    parser.add_argument("--campaign", default=None, help="Campaign prefix in returned tarball names")
    parser.add_argument("--cluster", default=None, help="Condor cluster id to collect")
    parser.add_argument("--pattern", default=None, help="Explicit tarball glob, relative to return dir")
    args = parser.parse_args()

    store = Path(args.store)
    return_dir = Path(args.return_dir) if args.return_dir else store / "condor_return"

    if args.pattern:
        pattern = args.pattern
    elif args.campaign and args.cluster:
        pattern = f"{args.campaign}_{args.cluster}_*.tar.gz"
    elif args.campaign:
        pattern = f"{args.campaign}_*.tar.gz"
    else:
        pattern = "*.tar.gz"

    tarballs = sorted(return_dir.glob(pattern))
    if not tarballs:
        raise SystemExit(f"ERROR: no tarballs matched {return_dir / pattern}")

    counts = {kind: 0 for kind in KINDS}
    copied = {kind: [] for kind in KINDS}

    for tarball in tarballs:
        print(f"Collecting: {tarball}")
        with tempfile.TemporaryDirectory(prefix="extract_", dir=return_dir) as tmp_name:
            tmp = Path(tmp_name)
            with tarfile.open(tarball, "r:gz") as tf:
                safe_extract(tf, tmp)

            for kind in KINDS:
                source_dir = tmp / kind
                if not source_dir.is_dir():
                    continue
                for src in sorted(path for path in source_dir.rglob("*") if path.is_file()):
                    dst = store / kind / src.name
                    if copy_without_overwrite(src, dst):
                        counts[kind] += 1
                        copied[kind].append(dst)

    print()
    print("Collection summary:")
    print(f"  tarballs: {len(tarballs)}")
    for kind in KINDS:
        print(f"  {kind}: {counts[kind]} new files")
        for path in copied[kind]:
            print(f"    {path}")


if __name__ == "__main__":
    main()
