#!/usr/bin/env python3
packages = ["uproot", "awkward", "vector", "hist", "matplotlib", "numpy", "pandas", "pyarrow"]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK {pkg}: {version}")
    except Exception as e:
        print(f"MISSING {pkg}: {e}")
        raise SystemExit(1)

print("Python stack is ready.")
