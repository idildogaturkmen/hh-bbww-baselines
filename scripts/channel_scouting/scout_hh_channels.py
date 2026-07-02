'''
Pivot direction:
Search for interesting channels in HH final states, using a simple scouting approach.
look for bbbb, bb gamma gamma, bbWW, and other final states.
'''

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ----------------------------------------------------------------------
# Basic math helpers
# ----------------------------------------------------------------------

def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b and b > 0 else 0.0


def neff(sumw: float, sumw2: float) -> float:
    return float(sumw * sumw / sumw2) if sumw2 > 0 else 0.0


def fourvec(pt: float, eta: float, phi: float, mass: float = 0.0) -> tuple[float, float, float, float]:
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return e, px, py, pz


def inv_mass(objs: list[tuple[float, float, float, float]]) -> float:
    e = sum(o[0] for o in objs)
    px = sum(o[1] for o in objs)
    py = sum(o[2] for o in objs)
    pz = sum(o[3] for o in objs)
    m2 = e * e - px * px - py * py - pz * pz
    return math.sqrt(max(m2, 0.0))


def delta_phi(phi1: float, phi2: float) -> float:
    d = phi1 - phi2
    while d > math.pi:
        d -= 2 * math.pi
    while d <= -math.pi:
        d += 2 * math.pi
    return d


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


# ----------------------------------------------------------------------
# Column and metadata helpers
# ----------------------------------------------------------------------

def norm_name(x: Any) -> str:
    return str(x).strip().lower().replace("/", "_").replace("-", "_")


def col_key(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def first_existing(cols: set[str], candidates: list[str]) -> str | None:
    lookup = {col_key(c): c for c in cols}
    for cand in candidates:
        key = col_key(cand)
        if key in lookup:
            return lookup[key]
    return None


def regex_col(cols: set[str], patterns: list[str]) -> str | None:
    ordered = sorted(cols)
    for pat in patterns:
        r = re.compile(pat, re.IGNORECASE)
        for c in ordered:
            if r.search(c):
                return c
    return None


def as_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return []


def load_metadata(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Metadata file not found: {p}")
        return None

    df = pd.read_csv(p)
    lower = {c.lower(): c for c in df.columns}

    sample_col = None
    for key in ["sample", "sample_name", "dataset", "name"]:
        if key in lower:
            sample_col = lower[key]
            break

    xsec_col = None
    for key in ["sigma_proxy", "xsec_fb", "cross_section", "cross_section_fb", "cross_section_pb", "xsec", "xs", "sigma_fb", "sigma_pb"]:
        if key in lower:
            xsec_col = lower[key]
            break

    group_col = None
    for key in ["process_group", "group", "process", "category"]:
        if key in lower:
            group_col = lower[key]
            break

    n_col = None
    for key in ["n_local_skim", "n_local", "n_events", "num_events", "n_generated", "ngen", "sumw"]:
        if key in lower:
            n_col = lower[key]
            break

    df.attrs["sample_col"] = sample_col
    df.attrs["xsec_col"] = xsec_col
    df.attrs["group_col"] = group_col
    df.attrs["n_col"] = n_col

    print("[INFO] Metadata columns:")
    print("  sample_col:", sample_col)
    print("  xsec_col:  ", xsec_col)
    print("  group_col: ", group_col)
    print("  n_col:     ", n_col)

    return df


def sample_from_path(path: Path) -> str:
    # Expected: outputs/collide_selected_backgrounds/<sample>/*.parquet
    if path.parent.name:
        return path.parent.name
    return path.stem


def infer_group(sample: str, meta_group: str | None = None) -> str:
    if meta_group and str(meta_group).lower() != "nan":
        return str(meta_group)

    s = sample.lower()

    if "hh" in s or "doubleh" in s or "di_higgs" in s or "dihiggs" in s:
        if any(k in s for k in ["4b", "bbbb", "to4b"]):
            return "HH4b_signal"
        if any(k in s for k in ["bbgg", "bbgammagamma", "bbaa", "bbgam", "bb_gamma"]):
            return "HHbbgg_signal"
        if "bbww" in s or ("bb" in s and "ww" in s):
            return "HHbbWW_signal"
        return "HH_signal"

    if "qcd" in s:
        return "qcd"
    if "ttbar" in s or "ttto" in s or "tt_" in s or s.startswith("tt"):
        return "ttbar"
    if "wjets" in s or "wto" in s:
        return "wjets"
    if "zjets" in s or "dy" in s or "drell" in s:
        return "dy_zjets"
    if "diboson" in s or "ww" in s or "wz" in s or "zz" in s:
        return "diboson"
    if "singletop" in s or "single_top" in s or "st_" in s:
        return "single_top"
    if "tth" in s or "vh" in s or "ggh" in s or "vbfh" in s:
        return "single_higgs"
    if "gamma" in s or "gjets" in s or "diphoton" in s:
        return "photon_background"

    return "other_background"


def build_sample_info(files: list[Path], metadata: pd.DataFrame | None, lumi_fb: float) -> dict[str, dict[str, Any]]:
    sample_counts = defaultdict(int)
    for f in files:
        try:
            sample_counts[sample_from_path(f)] += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            pass

    meta_by_key = {}
    if metadata is not None and metadata.attrs.get("sample_col"):
        sample_col = metadata.attrs["sample_col"]
        for _, row in metadata.iterrows():
            meta_by_key[norm_name(row[sample_col])] = row

    info = {}
    for sample, n_local in sample_counts.items():
        row = None
        key = norm_name(sample)

        if key in meta_by_key:
            row = meta_by_key[key]
        else:
            # substring fallback
            for k, candidate in meta_by_key.items():
                if key in k or k in key:
                    row = candidate
                    break

        xsec = None
        group = None
        n_norm = n_local

        if row is not None:
            xsec_col = metadata.attrs.get("xsec_col")
            group_col = metadata.attrs.get("group_col")
            n_col = metadata.attrs.get("n_col")

            if xsec_col:
                try:
                    xsec = float(row[xsec_col])
                except Exception:
                    xsec = None

            if group_col:
                group = str(row[group_col])

            if n_col:
                try:
                    tmp = float(row[n_col])
                    if tmp > 0:
                        n_norm = tmp
                except Exception:
                    pass

        # Current project convention: rough proxy weight ~ L * sigma_proxy / N_local.
        # If xsec is missing/NaN, keep the sample for raw counts but exclude it
        # from weighted B/S sums. This avoids poisoning B_w with NaN.
        if xsec is not None:
            try:
                if not np.isfinite(float(xsec)):
                    xsec = None
            except Exception:
                xsec = None

        if xsec is not None and n_norm > 0:
            weight = lumi_fb * xsec / n_norm
            has_weight = True
        else:
            weight = float("nan")
            has_weight = False

        info[sample] = {
            "n_local": n_local,
            "n_norm": n_norm,
            "xsec_proxy": xsec,
            "event_weight": weight,
            "has_weight": has_weight,
            "process_group": infer_group(sample, group),
        }

    return info


def is_signal(sample: str, group: str, signal_key: str) -> bool:
    text = f"{sample} {group}".lower()

    patterns = {
        "hh_any": r"(hh|doubleh|dihiggs|di_higgs)",
        "hh4b": r"(hh.*(4b|bbbb|to4b|bbbarbbbar)|4b.*hh|bbbb)",
        "hhbbgg": r"(hh.*(bbgg|bbgam|bb_gamma|bbgammagamma|bbaa|gamma.*gamma)|bbgg|bbaa)",
        "hhbbww": r"(hh.*bb.*ww|bbww)",
    }

    pat = patterns.get(signal_key, patterns["hh_any"])
    return bool(re.search(pat, text))


# ----------------------------------------------------------------------
# Branch discovery
# ----------------------------------------------------------------------

def discover_columns(cols: set[str]) -> dict[str, str | None]:
    m = {}

    m["ak4_pt"] = first_existing(cols, ["FullReco_JetAK4_PT", "FullReco_JetPuppiAK4_PT"])
    m["ak4_eta"] = first_existing(cols, ["FullReco_JetAK4_Eta", "FullReco_JetPuppiAK4_Eta"])
    m["ak4_phi"] = first_existing(cols, ["FullReco_JetAK4_Phi", "FullReco_JetPuppiAK4_Phi"])
    m["ak4_mass"] = first_existing(cols, ["FullReco_JetAK4_Mass", "FullReco_JetPuppiAK4_Mass"])
    m["ak4_btag"] = first_existing(cols, ["FullReco_JetAK4_BTag", "FullReco_JetPuppiAK4_BTag"])
    m["ak4_btagphys"] = first_existing(cols, ["FullReco_JetAK4_BTagPhys", "FullReco_JetPuppiAK4_BTagPhys"])

    m["ak8_pt"] = first_existing(cols, ["FullReco_JetAK8_PT", "FullReco_FatJet_PT", "FullReco_JetPuppiAK8_PT"])
    m["ak8_eta"] = first_existing(cols, ["FullReco_JetAK8_Eta", "FullReco_FatJet_Eta", "FullReco_JetPuppiAK8_Eta"])
    m["ak8_phi"] = first_existing(cols, ["FullReco_JetAK8_Phi", "FullReco_FatJet_Phi", "FullReco_JetPuppiAK8_Phi"])
    m["ak8_mass"] = first_existing(cols, ["FullReco_JetAK8_Mass", "FullReco_FatJet_Mass", "FullReco_JetPuppiAK8_Mass"])

    m["electron_pt"] = first_existing(cols, ["FullReco_Electron_PT"])
    m["electron_eta"] = first_existing(cols, ["FullReco_Electron_Eta"])
    m["muon_pt"] = first_existing(cols, ["FullReco_MuonTight_PT", "FullReco_Muon_PT"])
    m["muon_eta"] = first_existing(cols, ["FullReco_MuonTight_Eta", "FullReco_Muon_Eta"])

    m["met"] = first_existing(cols, ["FullReco_MET_MET", "FullReco_PUPPIMET_MET"])
    m["met_phi"] = first_existing(cols, ["FullReco_MET_Phi", "FullReco_PUPPIMET_Phi"])

    m["photon_pt"] = regex_col(cols, [
        r"FullReco_.*Photon.*_PT",
        r"FullReco_.*Gamma.*_PT",
        r"Photon.*PT",
    ])
    m["photon_eta"] = regex_col(cols, [
        r"FullReco_.*Photon.*_Eta",
        r"FullReco_.*Gamma.*_Eta",
        r"Photon.*Eta",
    ])
    m["photon_phi"] = regex_col(cols, [
        r"FullReco_.*Photon.*_Phi",
        r"FullReco_.*Gamma.*_Phi",
        r"Photon.*Phi",
    ])

    return m


# ----------------------------------------------------------------------
# Event feature extraction
# ----------------------------------------------------------------------

def get(row: pd.Series, col: str | None) -> list[float]:
    if col is None or col not in row.index:
        return []
    return as_list(row[col])


def count_pt_eta(pt: list[float], eta: list[float], pt_min: float, eta_max: float | None) -> int:
    n = 0
    for i, p in enumerate(pt):
        e = eta[i] if i < len(eta) else 0.0
        if p >= pt_min and (eta_max is None or abs(e) <= eta_max):
            n += 1
    return n


def btag_mask_and_scores(row: pd.Series, m: dict[str, str | None], pt_min: float, eta_max: float, btag_threshold: float):
    pt = get(row, m["ak4_pt"])
    eta = get(row, m["ak4_eta"])
    btag = get(row, m["ak4_btag"])
    bphys = get(row, m["ak4_btagphys"])

    out = []
    for i, p in enumerate(pt):
        e = eta[i] if i < len(eta) else 0.0
        if p < pt_min or abs(e) > eta_max:
            continue

        score = 0.0
        is_b = False

        if i < len(btag):
            try:
                score = max(score, float(btag[i]))
            except Exception:
                pass

        if i < len(bphys):
            try:
                val = float(bphys[i])
                # If BTagPhys is truth/flavor-like, abs==5 indicates b-like.
                # If it is score-like, threshold works.
                if abs(val) == 5:
                    is_b = True
                    score = max(score, 1.0)
                elif 0.0 <= val <= 1.0:
                    score = max(score, val)
            except Exception:
                pass

        if score >= btag_threshold:
            is_b = True

        out.append((i, is_b, score, p))

    return out


def selected_ak4_indices(row: pd.Series, m: dict[str, str | None], pt_min: float, eta_max: float) -> list[int]:
    pt = get(row, m["ak4_pt"])
    eta = get(row, m["ak4_eta"])
    out = []
    for i, p in enumerate(pt):
        e = eta[i] if i < len(eta) else 0.0
        if p >= pt_min and abs(e) <= eta_max:
            out.append(i)
    return out


def best_hh4b_masses(row: pd.Series, m: dict[str, str | None], pt_min: float, eta_max: float, btag_threshold: float):
    pt = get(row, m["ak4_pt"])
    eta = get(row, m["ak4_eta"])
    phi = get(row, m["ak4_phi"])
    mass = get(row, m["ak4_mass"])

    binfo = btag_mask_and_scores(row, m, pt_min, eta_max, btag_threshold)
    bjets = [(i, score, p) for i, is_b, score, p in binfo if is_b]

    if len(bjets) < 4:
        return None, None, None

    bjets = sorted(bjets, key=lambda x: (x[1], x[2]), reverse=True)[:4]
    idx = [x[0] for x in bjets]

    def jet4(i):
        return fourvec(
            float(pt[i]),
            float(eta[i]) if i < len(eta) else 0.0,
            float(phi[i]) if i < len(phi) else 0.0,
            float(mass[i]) if i < len(mass) else 0.0,
        )

    pairings = [
        ((idx[0], idx[1]), (idx[2], idx[3])),
        ((idx[0], idx[2]), (idx[1], idx[3])),
        ((idx[0], idx[3]), (idx[1], idx[2])),
    ]

    best = None
    for p1, p2 in pairings:
        m1 = inv_mass([jet4(p1[0]), jet4(p1[1])])
        m2 = inv_mass([jet4(p2[0]), jet4(p2[1])])
        score = abs(m1 - 125.0) + abs(m2 - 125.0)
        if best is None or score < best[0]:
            best = (score, m1, m2)

    return best[1], best[2], best[0]


def diphoton_mass(row: pd.Series, m: dict[str, str | None], pt_min: float, eta_max: float):
    pt = get(row, m["photon_pt"])
    eta = get(row, m["photon_eta"])
    phi = get(row, m["photon_phi"])

    photons = []
    for i, p in enumerate(pt):
        e = eta[i] if i < len(eta) else 0.0
        if p >= pt_min and abs(e) <= eta_max:
            photons.append((i, p))
    if len(photons) < 2:
        return None

    photons = sorted(photons, key=lambda x: x[1], reverse=True)[:2]
    objs = []
    for i, _ in photons:
        objs.append(fourvec(
            float(pt[i]),
            float(eta[i]) if i < len(eta) else 0.0,
            float(phi[i]) if i < len(phi) else 0.0,
            0.0,
        ))
    return inv_mass(objs)


def vbf_pair_features(row: pd.Series, m: dict[str, str | None], pt_min: float, eta_max: float, btag_threshold: float):
    pt = get(row, m["ak4_pt"])
    eta = get(row, m["ak4_eta"])
    phi = get(row, m["ak4_phi"])
    mass = get(row, m["ak4_mass"])

    selected = selected_ak4_indices(row, m, pt_min, eta_max)
    binfo = btag_mask_and_scores(row, m, pt_min, eta_max, btag_threshold)
    bset = {i for i, is_b, _, _ in binfo if is_b}

    # VBF candidates: non-b or lower-btag AK4 jets.
    cand = [i for i in selected if i not in bset]
    if len(cand) < 2:
        return 0.0, 0.0

    best = (0.0, 0.0)
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            i, j = cand[a], cand[b]
            deta = abs((eta[i] if i < len(eta) else 0.0) - (eta[j] if j < len(eta) else 0.0))
            objs = [
                fourvec(float(pt[i]), float(eta[i]), float(phi[i]), float(mass[i]) if i < len(mass) else 0.0),
                fourvec(float(pt[j]), float(eta[j]), float(phi[j]), float(mass[j]) if j < len(mass) else 0.0),
            ]
            mjj = inv_mass(objs)
            if mjj > best[0]:
                best = (mjj, deta)
    return best


def event_features(row: pd.Series, m: dict[str, str | None], args) -> dict[str, float | bool | None]:
    ak4_pt = get(row, m["ak4_pt"])
    ak4_eta = get(row, m["ak4_eta"])
    ak8_pt = get(row, m["ak8_pt"])
    ak8_eta = get(row, m["ak8_eta"])
    el_pt = get(row, m["electron_pt"])
    el_eta = get(row, m["electron_eta"])
    mu_pt = get(row, m["muon_pt"])
    mu_eta = get(row, m["muon_eta"])
    pho_pt = get(row, m["photon_pt"])
    pho_eta = get(row, m["photon_eta"])

    n_ak4 = count_pt_eta(ak4_pt, ak4_eta, args.ak4_pt_min, args.ak4_eta_max)
    n_ak8_200 = count_pt_eta(ak8_pt, ak8_eta, 200.0, 2.5)
    n_ak8_300 = count_pt_eta(ak8_pt, ak8_eta, 300.0, 2.5)
    n_e = count_pt_eta(el_pt, el_eta, 10.0, 2.5)
    n_mu = count_pt_eta(mu_pt, mu_eta, 10.0, 2.5)
    n_lep = n_e + n_mu
    n_pho = count_pt_eta(pho_pt, pho_eta, 25.0, 2.5)

    binfo = btag_mask_and_scores(row, m, args.ak4_pt_min, args.ak4_eta_max, args.btag_threshold)
    n_btag = sum(1 for _, is_b, _, _ in binfo if is_b)

    h1, h2, hh_score = best_hh4b_masses(row, m, args.ak4_pt_min, args.ak4_eta_max, args.btag_threshold)
    mgg = diphoton_mass(row, m, 25.0, 2.5)
    vbf_mjj, vbf_deta = vbf_pair_features(row, m, 30.0, 5.0, args.btag_threshold)

    return {
        "n_ak4": n_ak4,
        "n_btag": n_btag,
        "n_ak8_200": n_ak8_200,
        "n_ak8_300": n_ak8_300,
        "n_lep": n_lep,
        "n_photon": n_pho,
        "has_met": bool(m["met"] is not None),
        "h1_mass": h1,
        "h2_mass": h2,
        "hh4b_pair_score": hh_score,
        "mgg": mgg,
        "vbf_mjj": vbf_mjj,
        "vbf_deta": vbf_deta,
    }


# ----------------------------------------------------------------------
# Region definitions
# ----------------------------------------------------------------------

def region_definitions():
    return [
        {
            "region": "bbWW_onelep_reference_like",
            "signal_key": "hhbbww",
            "selection": lambda f: f["n_lep"] >= 1 and f["n_btag"] >= 2 and f["has_met"],
            "selection_sketch": ">=1 e/mu, >=2 b-tag AK4 jets, MET available",
        },
        {
            "region": "HH4b_resolved_4b_basic",
            "signal_key": "hh4b",
            "selection": lambda f: f["n_btag"] >= 4,
            "selection_sketch": ">=4 b-tag AK4 jets",
        },
        {
            "region": "HH4b_resolved_4b_Hmass",
            "signal_key": "hh4b",
            "selection": lambda f: f["n_btag"] >= 4
            and f["h1_mass"] is not None
            and f["h2_mass"] is not None
            and 70.0 <= f["h1_mass"] <= 190.0
            and 70.0 <= f["h2_mass"] <= 190.0,
            "selection_sketch": ">=4 b-tag AK4 jets and both reconstructed H candidates in 70-190 GeV",
        },
        {
            "region": "HH4b_boosted_2AK8",
            "signal_key": "hh4b",
            "selection": lambda f: f["n_ak8_200"] >= 2,
            "selection_sketch": ">=2 AK8/fat jets with pT>200 GeV",
        },
        {
            "region": "HH4b_boosted_1AK8_2b",
            "signal_key": "hh4b",
            "selection": lambda f: f["n_ak8_300"] >= 1 and f["n_btag"] >= 2,
            "selection_sketch": ">=1 AK8/fat jet with pT>300 GeV and >=2 b-tag AK4 jets",
        },
        {
            "region": "HH4b_resolved_VBF_like",
            "signal_key": "hh4b",
            "selection": lambda f: f["n_btag"] >= 4 and f["vbf_mjj"] >= 500.0 and f["vbf_deta"] >= 3.0,
            "selection_sketch": ">=4 b-tag AK4 jets plus VBF-like non-b jet pair with mjj>500 and |deta|>3",
        },
        {
            "region": "bbgamma_gamma_basic",
            "signal_key": "hhbbgg",
            "selection": lambda f: f["n_photon"] >= 2 and f["n_btag"] >= 2,
            "selection_sketch": ">=2 photons and >=2 b-tag AK4 jets",
        },
        {
            "region": "bbgamma_gamma_mgg_window",
            "signal_key": "hhbbgg",
            "selection": lambda f: f["n_photon"] >= 2
            and f["n_btag"] >= 2
            and f["mgg"] is not None
            and 115.0 <= f["mgg"] <= 135.0,
            "selection_sketch": ">=2 photons, >=2 b-tag AK4 jets, 115<mgg<135 GeV",
        },
    ]


# ----------------------------------------------------------------------
# Main scouting
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="outputs/collide_selected_backgrounds/**/*.parquet")
    parser.add_argument("--metadata", default="config/collide_sample_metadata_rough.csv")
    parser.add_argument("--outdir", default="outputs/channel_scouting")
    parser.add_argument("--lumi-fb", type=float, default=138.0)
    parser.add_argument("--ak4-pt-min", type=float, default=30.0)
    parser.add_argument("--ak4-eta-max", type=float, default=2.5)
    parser.add_argument("--btag-threshold", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events-per-file", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(".").glob(args.input_glob))
    if args.max_files:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No parquet files matched: {args.input_glob}")

    print(f"[INFO] Found {len(files)} parquet files")

    metadata = load_metadata(args.metadata)
    sample_info = build_sample_info(files, metadata, args.lumi_fb)

    pd.DataFrame([
        {"sample": s, **info}
        for s, info in sorted(sample_info.items())
    ]).to_csv(outdir / "sample_weight_inventory.csv", index=False)

    regions = region_definitions()
    region_acc = {
        r["region"]: {
            "region": r["region"],
            "signal_key": r["signal_key"],
            "selection_sketch": r["selection_sketch"],
            "raw_signal": 0,
            "raw_background": 0,
            "raw_signal_missing_weight": 0,
            "raw_background_missing_weight": 0,
            "S_w": 0.0,
            "B_w": 0.0,
            "B_w2": 0.0,
            "hh_any_raw": 0,
            "selected_total_raw": 0,
            "sum_n_ak4": 0.0,
            "sum_n_btag": 0.0,
            "sum_n_ak8_200": 0.0,
            "sum_n_photon": 0.0,
        }
        for r in regions
    }
    bkg_acc = defaultdict(lambda: {"raw": 0, "weighted": 0.0, "w2": 0.0})
    object_rows = []

    required_col_union = set()

    for file_idx, f in enumerate(files, start=1):
        sample = sample_from_path(f)
        sinfo = sample_info[sample]
        weight = float(sinfo["event_weight"])
        group = str(sinfo["process_group"])

        try:
            pf = pq.ParquetFile(f)
            cols = set(pf.schema_arrow.names)
        except Exception as e:
            print(f"[WARN] Could not read schema for {f}: {e}")
            continue

        m = discover_columns(cols)
        needed = sorted({c for c in m.values() if c is not None})
        required_col_union.update(needed)

        if not needed:
            print(f"[WARN] No useful columns found in {f}")
            continue

        try:
            table = pq.read_table(f, columns=needed)
            df = table.to_pandas()
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue

        if args.max_events_per_file:
            df = df.iloc[: args.max_events_per_file]

        sample_counts = {
            "sample": sample,
            "process_group": group,
            "file": str(f),
            "raw_events": len(df),
            "has_weight": sinfo["has_weight"],
            "event_weight": weight,
            "has_ak4": m["ak4_pt"] is not None,
            "has_ak8": m["ak8_pt"] is not None,
            "has_photon": m["photon_pt"] is not None,
            "has_lepton": (m["electron_pt"] is not None or m["muon_pt"] is not None),
            "has_met": m["met"] is not None,
        }

        n_ge4b = 0
        n_ge2pho = 0
        n_ge2ak8 = 0

        print(f"[INFO] [{file_idx}/{len(files)}] {sample}: {len(df)} rows")

        for _, row in df.iterrows():
            feats = event_features(row, m, args)

            if feats["n_btag"] >= 4:
                n_ge4b += 1
            if feats["n_photon"] >= 2:
                n_ge2pho += 1
            if feats["n_ak8_200"] >= 2:
                n_ge2ak8 += 1

            for r in regions:
                passed = bool(r["selection"](feats))
                if not passed:
                    continue

                rname = r["region"]
                acc = region_acc[rname]
                sig = is_signal(sample, group, r["signal_key"])
                hh_any = is_signal(sample, group, "hh_any")

                acc["selected_total_raw"] += 1
                acc["sum_n_ak4"] += feats["n_ak4"]
                acc["sum_n_btag"] += feats["n_btag"]
                acc["sum_n_ak8_200"] += feats["n_ak8_200"]
                acc["sum_n_photon"] += feats["n_photon"]

                if hh_any:
                    acc["hh_any_raw"] += 1

                if sig:
                    acc["raw_signal"] += 1
                    if np.isfinite(weight):
                        acc["S_w"] += weight
                    else:
                        acc["raw_signal_missing_weight"] += 1
                else:
                    acc["raw_background"] += 1
                    if np.isfinite(weight):
                        acc["B_w"] += weight
                        acc["B_w2"] += weight * weight

                        key = (rname, group)
                        bkg_acc[key]["raw"] += 1
                        bkg_acc[key]["weighted"] += weight
                        bkg_acc[key]["w2"] += weight * weight
                    else:
                        acc["raw_background_missing_weight"] += 1

        sample_counts["frac_ge4_btag"] = safe_div(n_ge4b, len(df))
        sample_counts["frac_ge2_photon"] = safe_div(n_ge2pho, len(df))
        sample_counts["frac_ge2_ak8"] = safe_div(n_ge2ak8, len(df))
        object_rows.append(sample_counts)

    # Summary table
    summary_rows = []
    for r in regions:
        acc = region_acc[r["region"]]
        s = acc["S_w"]
        b = acc["B_w"]
        total_sel = acc["selected_total_raw"]
        n_eff = neff(acc["B_w"], acc["B_w2"])

        row = {
            "channel_region": r["region"],
            "signal_definition": r["signal_key"],
            "selection_sketch": r["selection_sketch"],
            "raw_signal": acc["raw_signal"],
            "raw_background": acc["raw_background"],
            "raw_signal_missing_weight": acc["raw_signal_missing_weight"],
            "raw_background_missing_weight": acc["raw_background_missing_weight"],
            "raw_selected_total": total_sel,
            "raw_any_HH_selected": acc["hh_any_raw"],
            "S_w": s,
            "B_w": b,
            "S_over_B": safe_div(s, b),
            "S_over_SplusB": safe_div(s, s + b),
            "S_over_sqrtB": safe_div(s, math.sqrt(b)) if b > 0 else 0.0,
            "S_over_sqrtSplusB": safe_div(s, math.sqrt(s + b)) if (s + b) > 0 else 0.0,
            "asimov_Z_A": asimov_z(s, b),
            "N_eff_bkg": n_eff,
            "avg_n_ak4_selected": safe_div(acc["sum_n_ak4"], total_sel),
            "avg_n_btag_selected": safe_div(acc["sum_n_btag"], total_sel),
            "avg_n_ak8_200_selected": safe_div(acc["sum_n_ak8_200"], total_sel),
            "avg_n_photon_selected": safe_div(acc["sum_n_photon"], total_sel),
        }

        comments = []
        if acc["raw_signal"] == 0:
            comments.append("No channel-specific signal matched; check sample naming/regex.")
        if acc["raw_signal_missing_weight"] > 0:
            comments.append("Some selected signal events have missing weights.")
        if acc["raw_background_missing_weight"] > 0:
            comments.append("Some selected background events have missing weights; weighted B is incomplete.")
        if acc["raw_signal"] > 0 and s < 1.0:
            comments.append("Very small weighted signal yield.")
        if b > 0 and n_eff < 10:
            comments.append("Background estimate likely unstable.")
        elif b > 0 and n_eff < 25:
            comments.append("Background effective statistics are modest.")
        if row["S_over_sqrtSplusB"] > 0 and row["S_over_B"] > 0:
            comments.append("Use S/sqrt(S+B) especially if S and B are comparable.")
        row["comment"] = " ".join(comments)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        by=["S_over_sqrtSplusB", "asimov_Z_A", "S_over_B"],
        ascending=False,
    )
    summary.to_csv(outdir / "channel_scouting_summary.csv", index=False)

    # Dominant backgrounds
    bkg_rows = []
    b_by_region = {r: region_acc[r]["B_w"] for r in region_acc}
    for (region, group), vals in bkg_acc.items():
        bw = vals["weighted"]
        bkg_rows.append({
            "channel_region": region,
            "process_group": group,
            "raw_background": vals["raw"],
            "B_w": bw,
            "fraction_of_B_w": safe_div(bw, b_by_region.get(region, 0.0)),
            "N_eff_group": neff(bw, vals["w2"]),
        })

    bkg_df = pd.DataFrame(bkg_rows)
    if len(bkg_df):
        bkg_df = bkg_df.sort_values(["channel_region", "B_w"], ascending=[True, False])
    bkg_df.to_csv(outdir / "channel_scouting_dominant_backgrounds.csv", index=False)

    # Object counts
    obj_df = pd.DataFrame(object_rows)
    obj_df.to_csv(outdir / "channel_scouting_object_counts.csv", index=False)

    # Discovered branches
    pd.DataFrame({"column_used": sorted(required_col_union)}).to_csv(outdir / "channel_scouting_columns_used.csv", index=False)

    # Draft decision note
    def df_md(df: pd.DataFrame, n: int | None = None) -> str:
        x = df if n is None else df.head(n)
        try:
            return x.to_markdown(index=False)
        except Exception:
            return x.to_csv(index=False)

    md = []
    md.append("# Channel scouting draft\n")
    md.append("This is an automatically generated first-pass scouting summary using rough proxy MC weights.\n")
    md.append("These numbers are diagnostic only. The purpose is to identify promising channels/regions before training new models.\n")
    md.append("\n## Ranked scouting summary\n")
    md.append(df_md(summary))
    md.append("\n## Dominant backgrounds\n")
    md.append(df_md(bkg_df, 80))
    md.append("\n## Object availability by file/sample\n")
    md.append(df_md(obj_df, 80))
    md.append("\n## Interpretation checklist\n")
    md.append("""
- Prefer regions with nonzero channel-specific signal.
- Prefer regions with improved S/B and non-negligible S/sqrt(S+B).
- Check N_eff_bkg before trusting any high-purity region.
- For HH4b, compare resolved 4b, boosted AK8, and VBF-like categories.
- If S and B are comparable, prioritize S/sqrt(S+B) over S/sqrt(B).
- Do not move to SPA-Net/ParT/JP-JEPA/RINO until a promising region is identified.
""")
    (outdir / "channel_choice_draft.md").write_text("\n".join(md))

    print("\n[INFO] Wrote:")
    for name in [
        "sample_weight_inventory.csv",
        "channel_scouting_summary.csv",
        "channel_scouting_dominant_backgrounds.csv",
        "channel_scouting_object_counts.csv",
        "channel_scouting_columns_used.csv",
        "channel_choice_draft.md",
    ]:
        print(" -", outdir / name)

    print("\n[INFO] Top scouting rows:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
