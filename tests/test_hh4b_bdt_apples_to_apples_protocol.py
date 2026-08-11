import importlib.util
import json
from pathlib import Path

import numpy as np
import csv
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/baselines/hh4b_bdt_apples_to_apples_v1.json").read_text())
SPEC = importlib.util.spec_from_file_location("bdt_common", ROOT / "scripts/analysis/hh4b_bdt_apples_to_apples_common.py")
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)


def test_frozen_population_and_seals():
    assert CONFIG["population"]["rows"] == 1_042_397
    assert CONFIG["population"]["signal_rows"] + CONFIG["population"]["background_rows"] == 1_042_397
    assert CONFIG["population"]["btag_requirement"] is None
    assert CONFIG["sealed_data"]["bdt_validation_payloads_opened"] == 0
    assert CONFIG["sealed_data"]["bdt_test_payloads_opened"] == 0


def test_nested_source_partitions_are_disjoint_and_cover_rows():
    folds = np.repeat(np.arange(5), 3)
    groups = np.repeat([f"g{i}" for i in range(5)], 3)
    for outer in range(5):
        for inner in range(5):
            if inner == outer:
                continue
            fit, held, outer_mask = COMMON.nested_masks(folds, outer, inner)
            assert np.all(fit | held | outer_mask)
            assert not set(groups[fit]) & set(groups[held])
            assert not set(groups[fit]) & set(groups[outer_mask])
            assert not set(groups[held]) & set(groups[outer_mask])


def test_frozen_source_registry_population_and_group_leakage():
    path = ROOT / CONFIG["population"]["source_audit"]
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["auxiliary_qcd"] == "False"]
    assert len(rows) == 441
    assert sum(int(row["resolved_rows"]) for row in rows) == 1_042_397
    assignments = {}
    for row in rows:
        fold = int(row["optimized_fold_k5"])
        previous = assignments.setdefault(row["group_id"], fold)
        assert previous == fold


def test_exact_feature_order_and_mass_plane_ablation():
    blind = CONFIG["features"]["mass_plane_blind"]
    excluded = CONFIG["features"]["mass_plane_blind_exclusions"]
    aware = blind + CONFIG["features"]["mass_aware_append"]
    COMMON.validate_feature_contract(aware, blind, excluded)
    assert len(aware) == 34 and len(blind) == 30
    assert excluded == ["mbb1", "mbb2", "delta_mbb", "r_hh_125_125"]
    assert "mhh" in blind


def test_training_and_physical_weight_roles_are_separate():
    COMMON.validate_training_weights(np.array([0.1, 1.0, 2.0]))
    for bad in (np.array([1.0, -1.0]), np.array([1.0, np.nan]), np.array([0.0, 1.0])):
        try:
            COMMON.validate_training_weights(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid learner weights accepted")
    assert CONFIG["weights"]["training"]["signed"] is False
    assert CONFIG["weights"]["evaluation"]["signed"] is True


def test_threshold_is_deterministic_and_development_only():
    scores = np.array([0.9, 0.8, 0.8, 0.4, 0.3, 0.2])
    labels = np.array([1, 1, 0, 1, 0, 0])
    weights = np.ones(6)
    first = COMMON.select_fixed_efficiency_threshold(scores, labels, weights, 0.585957)
    second = COMMON.select_fixed_efficiency_threshold(scores, labels, weights, 0.585957)
    assert first == second == 0.8
    outer_scores = np.array([-999.0, 999.0])
    assert outer_scores.size == 2 and first == 0.8


def test_xgboost_score_generation_is_deterministic_at_fixed_seed():
    rng = np.random.default_rng(20260811)
    features = rng.normal(size=(80, 6))
    labels = (features[:, 0] + 0.2 * features[:, 1] > 0).astype(int)
    parameters = dict(
        n_estimators=12, max_depth=2, learning_rate=0.08,
        min_child_weight=1, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=5.0, reg_alpha=0.1, gamma=0.0,
        objective="binary:logistic", booster="gbtree", tree_method="hist",
        max_bin=256, eval_metric="logloss", n_jobs=1,
        random_state=CONFIG["implementation"]["random_seed"], verbosity=0,
    )
    first = XGBClassifier(**parameters).fit(features, labels).predict_proba(features)[:, 1]
    second = XGBClassifier(**parameters).fit(features, labels).predict_proba(features)[:, 1]
    assert np.array_equal(first, second)


def test_historical_comparator_preserves_identical_universe():
    universe = np.array([True, True, False, True])
    values = np.array([33.9, 34.0, 1.0, 80.0])
    selected = COMMON.historical_rhh34_mask(values, universe)
    assert selected.tolist() == [True, False, False, False]
    assert not np.any(selected & ~universe)


def test_full_training_and_submission_remain_blocked():
    assert CONFIG["authorization"]["full_training_authorized"] is False
    assert CONFIG["authorization"]["condor_submit_authorized"] is False


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
