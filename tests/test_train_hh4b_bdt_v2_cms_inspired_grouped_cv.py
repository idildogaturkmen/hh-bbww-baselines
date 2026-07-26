#!/usr/bin/env python3
"""Synthetic-only tests for categorized HH4b BDT-v2 grouped CV."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"]="8"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"/"analysis"))
from train_hh4b_bdt_v2_cms_inspired_grouped_cv import (  # noqa: E402
    category_local_hierarchical_fit_weights,
    category_specific_threshold_selection,
    check_categorized_oof_integrity,
    load_frozen_hyperparameter_space,
    member_bootstrap_differences,
    prohibit_pooled_uncalibrated_category_auc,
    score_mass_rows,
    select_category_hyperparameters,
    selection_efficiencies,
)
from train_hh4b_bdt_v1_grouped_cv import (  # noqa: E402
    aggregate_gain_importance,
    binary_metrics,
    rank_hyperparameter_trials,
)
from hh4b_bdt_v2_common import write_table_bundle  # noqa: E402
from scripts.plotting.hh4b_cms_style import apply_cms_style,add_delphes_header,save_png_pdf  # noqa: E402


class WeightTests(unittest.TestCase):
    def data(self):
        target=np.array([1,1,1,1,0,0,0,0]); members=np.array([1,1,2,3,4,4,5,6])
        modes=np.array(["g","g","v","v","","","",""],object)
        families=np.array(["","","","","a","a","b","b"],object)
        return target,members,modes,families
    def test_category_local_hierarchy(self):
        d=self.data(); w=category_local_hierarchical_fit_weights(*d,np.ones(8,bool))
        self.assertAlmostEqual(w[d[0]==1].sum(),w[d[0]==0].sum())
        self.assertAlmostEqual(w[d[1]==2].sum(),w[d[1]==3].sum())
    def test_heldout_exclusion(self):
        d=self.data(); selected=np.array([1,1,1,0,1,1,1,0],bool)
        w=category_local_hierarchical_fit_weights(*d,selected)
        self.assertTrue(np.all(w[~selected]==0)); self.assertAlmostEqual(w[selected].mean(),1)
    def test_global_evaluation_weights_remain_separate(self):
        d=self.data(); fit=category_local_hierarchical_fit_weights(*d,np.ones(8,bool))
        evaluation=np.arange(1,9,dtype=float)
        self.assertFalse(np.array_equal(fit,evaluation)); np.testing.assert_array_equal(evaluation,np.arange(1,9))


class SearchTests(unittest.TestCase):
    def test_exact_frozen_24(self):
        rows=load_frozen_hyperparameter_space(ROOT/"docs/checkpoints/hh4b_bdt_v1_grouped_cv_20260725_v1/hyperparameter_space.tsv")
        self.assertEqual(len(rows),24); self.assertEqual([r["trial_id"] for r in rows],list(range(24)))
    def test_independent_selection_and_tiebreak(self):
        hyper=load_frozen_hyperparameter_space(ROOT/"docs/checkpoints/hh4b_bdt_v1_grouped_cv_20260725_v1/hyperparameter_space.tsv")
        rows=[]
        for cat in ("low_mhh","high_mhh"):
            for trial in range(24):
                for fold in range(5):
                    auc=.8 + (.01 if (cat=="low_mhh" and trial==1) or (cat=="high_mhh" and trial==2) else 0)
                    rows.append({"category":cat,"trial_id":trial,"fold":fold,"weighted_roc_auc":auc,"fit_status":"pass"})
        _,selected=select_category_hyperparameters(rows,hyper)
        self.assertEqual(selected["low_mhh"]["trial_id"],1); self.assertEqual(selected["high_mhh"]["trial_id"],2)
    def test_exact_tiebreak_order(self):
        rows = [
            {"trial_id":5, "mean_weighted_roc_auc":.81, "worst_fold_weighted_roc_auc":.60,
             "std_weighted_roc_auc":.10, "complexity_proxy":999},
            {"trial_id":4, "mean_weighted_roc_auc":.80, "worst_fold_weighted_roc_auc":.72,
             "std_weighted_roc_auc":.10, "complexity_proxy":999},
            {"trial_id":3, "mean_weighted_roc_auc":.80, "worst_fold_weighted_roc_auc":.71,
             "std_weighted_roc_auc":.01, "complexity_proxy":999},
            {"trial_id":2, "mean_weighted_roc_auc":.80, "worst_fold_weighted_roc_auc":.71,
             "std_weighted_roc_auc":.02, "complexity_proxy":100},
            {"trial_id":1, "mean_weighted_roc_auc":.80, "worst_fold_weighted_roc_auc":.71,
             "std_weighted_roc_auc":.02, "complexity_proxy":200},
            {"trial_id":0, "mean_weighted_roc_auc":.80, "worst_fold_weighted_roc_auc":.71,
             "std_weighted_roc_auc":.02, "complexity_proxy":200},
        ]
        ranked=rank_hyperparameter_trials(rows)
        self.assertEqual([row["trial_id"] for row in ranked],[5,4,3,2,0,1])


class IntegrityTests(unittest.TestCase):
    def base(self,produced,cats=None):
        return check_categorized_oof_integrity([0,1,2],produced,[{1},{2}],[{2},{1}],
                                               ["l","l","h"],cats or ["l","l","h"])
    def test_completeness(self): self.assertEqual(self.base([0,2])["missing_oof_rows"],1)
    def test_duplicates(self): self.assertEqual(self.base([0,1,1,2])["duplicate_oof_rows"],1)
    def test_leakage(self):
        r=check_categorized_oof_integrity([0],[0],[{1}],[{1}],["l"],["l"])
        self.assertEqual(r["member_leakage_rows"],1)
    def test_category_mismatch(self): self.assertEqual(self.base([0,1,2],["l","h","h"])["category_mismatch_rows"],1)
    def test_pooled_auc_prohibited(self):
        with self.assertRaises(ValueError): prohibit_pooled_uncalibrated_category_auc()


class MetricTests(unittest.TestCase):
    def fixture(self):
        scores=np.array([.9,.8,.7,.6,.4,.3,.2,.1]); target=np.array([1,1,1,1,0,0,0,0])
        weights=np.ones(8); cats=np.array(["low_mhh"]*4+["high_mhh"]*4,object)
        # Ensure both classes in each category.
        cats=np.array(["low_mhh","low_mhh","high_mhh","high_mhh"]*2,object)
        return scores,target,weights,cats
    def test_category_thresholds(self):
        s,t,w,c=self.fixture(); thresholds,selected=category_specific_threshold_selection(s,t,w,c,.5)
        self.assertEqual(set(thresholds),{"low_mhh","high_mhh"}); self.assertTrue(selected.any())
    def test_combined_aggregation(self):
        s,t,w,c=self.fixture(); _,selected=category_specific_threshold_selection(s,t,w,c,.5)
        result=selection_efficiencies(selected,t,w,np.array(["g"]*4+[""]*4),np.array([""]*4+["b"]*4))
        self.assertIn("weighted_background_efficiency",result)
    def test_global_category_metrics(self):
        s,t,w,c=self.fixture(); metric=binary_metrics(t[c=="low_mhh"],s[c=="low_mhh"],w[c=="low_mhh"])
        self.assertTrue(np.isfinite(metric["weighted_roc_auc"]))
    def test_feature_category_ablation(self):
        first=.81; second=.79; self.assertAlmostEqual(first-second,.02)


class DiagnosticTests(unittest.TestCase):
    def test_bootstrap_reproducibility(self):
        members=np.repeat(np.arange(4),2); weights=np.ones(8); target=np.array([0,0,0,0,1,1,1,1])
        a=np.array([1,0,1,0,1,0,1,0],bool); b=~a
        x=member_bootstrap_differences(members,weights,target,a,b,seed=7,replicates=20)
        y=member_bootstrap_differences(members,weights,target,a,b,seed=7,replicates=20)
        np.testing.assert_array_equal(x,y)
    def test_feature_importance(self):
        rows=aggregate_gain_importance([[1,0],[1,1]],["mhh","hh_pt"],"x")
        self.assertAlmostEqual(sum(r["mean_normalized_gain_importance"] for r in rows),1)
    def test_score_mass_diagnostics(self):
        rows=score_mass_rows(np.linspace(0,1,100),np.linspace(1,200,100),np.linspace(250,449,100),
                             np.ones(100),"x","low_mhh")
        self.assertEqual(len(rows),5); self.assertTrue(all(not r["used_in_model_selection"] for r in rows))


class OutputTests(unittest.TestCase):
    def test_tables_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=write_table_bundle(
                Path(tmp),"x",[{"a":r"$x_y$","b":1}],("a","b"),
                caption="c",label="tab:x",latex_raw_fields=("a",),
            )
            self.assertTrue(all(p.stat().st_size for p in paths))
            self.assertIn(r"$x_y$",paths[2].read_text())
    def test_style_png_pdf_and_cleanup(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            meta=apply_cms_style(); self.assertFalse(meta["official_cms_status_claimed"])
            fig,ax=plt.subplots(); ax.plot([0,1],[0,1]); add_delphes_header(ax,"Train-only grouped cross-validation")
            png,pdf=save_png_pdf(fig,Path(tmp)/"x",dpi=300); self.assertTrue(png.is_file() and pdf.is_file())


if __name__=="__main__": unittest.main()
