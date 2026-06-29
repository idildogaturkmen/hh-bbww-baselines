'''
Make a rough COLLIDE-1M metadata CSV for use in rough weighted score study.
This is a rough proxy for the full metadata, and is not intended to be used for any other purpose. The cross sections are rough inclusive 13 TeV proxies, and the filter efficiencies are assumed to be 1.0 for all samples. The purpose of this file is to provide a rough estimate of the relative importance of different samples in the COLLIDE-1M dataset, for use in the rough weighted score study. 
The cross sections are taken from various sources, and are not intended to be used for any other purpose. The filter efficiencies are assumed to be 1.0 for all samples, and are not intended to be used for any other purpose. The purpose of this file is to provide a rough estimate of the relative importance of different samples in the COLLIDE-1M dataset, for use in the rough weighted score study.
Assumptions:
- The cross sections are rough inclusive 13 TeV proxies, and are not intended to be used for any other purpose.
- The filter efficiencies are assumed to be 1.0 for all samples, and are not intended to be used for any other purpose.
- The purpose of this file is to provide a rough estimate of the relative importance of different samples in the COLLIDE-1M dataset, for use in the rough weighted score study.

'''
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PB_TO_FB = 1000.0

# Higgs branching fractions, rough LHCHXSWG-style values for mH ~125 GeV.
BR_H = {
    "bb": 0.5824,
    "WW": 0.2152,
    "ZZ": 0.02641,
    "tautau": 0.06272,
    "gammagamma": 0.00227,
    "cc": 0.0289,
    "gluglu": 0.0818,
}

# W/Z rough branching fractions.
BR_W_HAD = 0.6741
BR_W_LEP = 1.0 - BR_W_HAD  # e, mu, tau combined

BR_Z_HAD = 0.6991
BR_Z_NONHAD = 1.0 - BR_Z_HAD
BR_Z_CHARGED_LEPTONS = 0.101  # e, mu, tau combined
BR_Z_NUNU = 0.200
BR_Z_BB = 0.1512
BR_Z_CC = 0.1203

rows = []


def add(sample, group, xsec_fb, status, source, note=""):
    rows.append(
        {
            "sample": sample,
            "process_group": group,
            "xsec_fb": xsec_fb,
            "filter_eff": 1.0 if np.isfinite(xsec_fb) else np.nan,
            "k_factor": 1.0 if np.isfinite(xsec_fb) else np.nan,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def add_missing(sample, group, status, source, note):
    add(sample, group, np.nan, status, source, note)


# ----------------------------------------------------------------------
# HH samples: ggF-only proxy at 13 TeV.
# ----------------------------------------------------------------------
sigma_hh_ggf_fb = 31.05

add(
    "HH_bbWW",
    "signal",
    sigma_hh_ggf_fb * 2 * BR_H["bb"] * BR_H["WW"],
    "rough_proxy_signal",
    "ggF HH 31.05 fb * 2*BR(H->bb)*BR(H->WW)",
    "Assumes HH_bbWW is ggF-only and inclusive in W decays. Alternative ggF+VBF is slightly higher; confirm with COLLIDE-1M authors.",
)

add(
    "HH_4b",
    "HH_other",
    sigma_hh_ggf_fb * BR_H["bb"] * BR_H["bb"],
    "rough_proxy_background",
    "ggF HH 31.05 fb * BR(H->bb)^2",
)

add(
    "HH_bbZZ",
    "HH_other",
    sigma_hh_ggf_fb * 2 * BR_H["bb"] * BR_H["ZZ"],
    "rough_proxy_background",
    "ggF HH 31.05 fb * 2*BR(H->bb)*BR(H->ZZ)",
)

add(
    "HH_bbgammagamma",
    "HH_other",
    sigma_hh_ggf_fb * 2 * BR_H["bb"] * BR_H["gammagamma"],
    "rough_proxy_background",
    "ggF HH 31.05 fb * 2*BR(H->bb)*BR(H->gammagamma)",
)

add(
    "HH_bbtautau",
    "HH_other",
    sigma_hh_ggf_fb * 2 * BR_H["bb"] * BR_H["tautau"],
    "rough_proxy_background",
    "ggF HH 31.05 fb * 2*BR(H->bb)*BR(H->tautau)",
)

# ----------------------------------------------------------------------
# Diboson samples.
# Cross sections are rough inclusive 13 TeV proxies, then split by W/Z decay modes.
# ----------------------------------------------------------------------
sigma_WW_pb = 118.7
sigma_WZ_pb = 50.6
sigma_ZZ_pb = 17.2

add("WW_hadronic", "diboson", sigma_WW_pb * PB_TO_FB * BR_W_HAD**2, "rough_proxy_background", "WW inclusive rough proxy * W_had^2")
add("WW_semileptonic", "diboson", sigma_WW_pb * PB_TO_FB * 2 * BR_W_HAD * BR_W_LEP, "rough_proxy_background", "WW inclusive rough proxy * 2*W_had*W_lep")
add("WW_leptonic", "diboson", sigma_WW_pb * PB_TO_FB * BR_W_LEP**2, "rough_proxy_background", "WW inclusive rough proxy * W_lep^2")

add("WZ_hadronic", "diboson", sigma_WZ_pb * PB_TO_FB * BR_W_HAD * BR_Z_HAD, "rough_proxy_background", "WZ inclusive * W_had*Z_had")
add("WZ_semileptonic", "diboson", sigma_WZ_pb * PB_TO_FB * (BR_W_LEP * BR_Z_HAD + BR_W_HAD * BR_Z_NONHAD), "rough_proxy_background", "WZ inclusive rough split")
add("WZ_leptonic", "diboson", sigma_WZ_pb * PB_TO_FB * BR_W_LEP * BR_Z_NONHAD, "rough_proxy_background", "WZ inclusive rough split")

add("ZZ_hadronic", "diboson", sigma_ZZ_pb * PB_TO_FB * BR_Z_HAD**2, "rough_proxy_background", "ZZ inclusive * Z_had^2")
add("ZZ_semileptonic", "diboson", sigma_ZZ_pb * PB_TO_FB * 2 * BR_Z_HAD * BR_Z_NONHAD, "rough_proxy_background", "ZZ inclusive rough split")
add("ZZ_leptonic", "diboson", sigma_ZZ_pb * PB_TO_FB * BR_Z_NONHAD**2, "rough_proxy_background", "ZZ inclusive rough split")

# ----------------------------------------------------------------------
# W/Z+jets rough inclusive proxies.
# These are large and will be statistically unstable if only a few local events pass.
# ----------------------------------------------------------------------
sigma_W_lnu_one_flavor_pb = 20480.0
sigma_W_total_pb = sigma_W_lnu_one_flavor_pb / 0.1086

add(
    "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    "wjets",
    sigma_W_total_pb * BR_W_LEP * PB_TO_FB,
    "rough_proxy_background",
    "Inclusive W proxy scaled to W->e,mu,tau",
    "Large-weight low-local-stat sample; use with caution.",
)

add(
    "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    "wjets",
    sigma_W_total_pb * BR_W_HAD * PB_TO_FB,
    "rough_proxy_background",
    "Inclusive W proxy scaled to W->qq",
    "Large-weight low-local-stat sample; use with caution.",
)

sigma_Z_ll_one_flavor_pb = 1952.0
sigma_Z_total_pb = sigma_Z_ll_one_flavor_pb / 0.03366

add("DYJetsToLL_13TeV-madgraphMLM-pythia8", "dy_zjets", sigma_Z_total_pb * BR_Z_CHARGED_LEPTONS * PB_TO_FB, "rough_proxy_background", "Inclusive Z proxy scaled to charged leptons e,mu,tau")
add("ZJetsToQQ_13TeV-madgraphMLM-pythia8", "dy_zjets", sigma_Z_total_pb * BR_Z_HAD * PB_TO_FB, "rough_proxy_background", "Inclusive Z proxy scaled to hadrons")
add("ZJetsTobb_13TeV-madgraphMLM-pythia8", "dy_zjets", sigma_Z_total_pb * BR_Z_BB * PB_TO_FB, "rough_proxy_background", "Inclusive Z proxy scaled to Z->bb")
add("ZJetsTocc_13TeV-madgraphMLM-pythia8", "dy_zjets", sigma_Z_total_pb * BR_Z_CC * PB_TO_FB, "rough_proxy_background", "Inclusive Z proxy scaled to Z->cc")
add("ZJetsTovv_13TeV-madgraphMLM-pythia8", "dy_zjets", sigma_Z_total_pb * BR_Z_NUNU * PB_TO_FB, "rough_proxy_background", "Inclusive Z proxy scaled to invisible Z decays")

# ----------------------------------------------------------------------
# ttbar, split by W decays.
# ----------------------------------------------------------------------
sigma_ttbar_pb = 831.8

add("tt0123j_5f_ckm_LO_MLM_hadronic", "ttbar", sigma_ttbar_pb * PB_TO_FB * BR_W_HAD**2, "rough_proxy_background", "ttbar inclusive * W_had^2")
add("tt0123j_5f_ckm_LO_MLM_semiLeptonic", "ttbar", sigma_ttbar_pb * PB_TO_FB * 2 * BR_W_HAD * BR_W_LEP, "rough_proxy_background", "ttbar inclusive * 2*W_had*W_lep")
add("tt0123j_5f_ckm_LO_MLM_leptonic", "ttbar", sigma_ttbar_pb * PB_TO_FB * BR_W_LEP**2, "rough_proxy_background", "ttbar inclusive * W_lep^2")

# ----------------------------------------------------------------------
# ttV / ttH / tttt.
# ----------------------------------------------------------------------
add("ttH_incl", "ttV_ttH_tttt", 507.1, "rough_proxy_background", "ttH inclusive rough 13 TeV proxy")
add("ttW_incl", "ttV_ttH_tttt", 868.0, "rough_proxy_background", "CMS ttW measured full-phase-space proxy")
add("ttZ_incl", "ttV_ttH_tttt", 990.0, "rough_proxy_background", "CMS ttZ measured full-phase-space proxy")
add("tttt_incl", "ttV_ttH_tttt", 17.7, "rough_proxy_background", "CMS tttt measured full-phase-space proxy")

# ----------------------------------------------------------------------
# Single Higgs samples.
# ----------------------------------------------------------------------
sigma_ggH_pb = 48.58
sigma_VBFH_pb = 3.782
sigma_VH_pb = 2.257

for decay, br in [
    ("WW", BR_H["WW"]),
    ("ZZ", BR_H["ZZ"]),
    ("bb", BR_H["bb"]),
    ("cc", BR_H["cc"]),
    ("gammagamma", BR_H["gammagamma"]),
    ("gluglu", BR_H["gluglu"]),
    ("tautau", BR_H["tautau"]),
]:
    add(f"ggH{decay}", "single_higgs", sigma_ggH_pb * PB_TO_FB * br, "rough_proxy_background", f"ggH inclusive * BR(H->{decay})")
    add(f"VBFH{decay}", "single_higgs", sigma_VBFH_pb * PB_TO_FB * br, "rough_proxy_background", f"VBFH inclusive * BR(H->{decay})")

add("VH_incl", "single_higgs", sigma_VH_pb * PB_TO_FB, "rough_proxy_background", "WH+ZH inclusive rough proxy, H decays inclusive")

# ----------------------------------------------------------------------
# Triboson.
# ----------------------------------------------------------------------
add("VVV_incl", "triboson", 600.0, "rough_proxy_background", "Very rough inclusive VVV proxy", "Order-of-magnitude proxy only.")

# ----------------------------------------------------------------------
# Special samples excluded from rough weighted score until filter definitions are clear.
# ----------------------------------------------------------------------
for sample, group in [
    ("QCD_HT50toInf", "qcd"),
    ("QCD_HT50tobb", "qcd"),
    ("gamma", "gamma"),
    ("gamma_V", "gamma"),
    ("minbias", "minbias"),
    ("upsilon_to_leptons", "upsilon"),
]:
    add_missing(
        sample,
        group,
        "excluded_pending_metadata",
        "No reliable rough cross section/filter-efficiency mapping yet",
        "Excluded from rough weighted score study until COLLIDE-1M metadata/filter definitions are clarified.",
    )

out = Path("config/collide_sample_metadata_rough.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {out} with {len(rows)} rows")
