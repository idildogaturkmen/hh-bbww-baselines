from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from plot_hh4b_cut_baseline_publication import (  # noqa: E402
    add_header,
    pd,
    plot_conditional_thresholds,
    plot_modal_tie,
    plot_yields,
    plt,
    short_structure,
)


def test_shared_header_uses_figure_coordinates_for_multi_panel_layouts() -> None:
    figure, axes = plt.subplots(1, 2)
    add_header(axes[0], "Multi-panel subtitle")
    assert len(figure.texts) == 3
    assert [text.get_text() for text in figure.texts] == [
        "Delphes simulation",
        r"$\sqrt{s}=13$ TeV, 138 fb$^{-1}$ equivalent",
        "Multi-panel subtitle",
    ]
    assert not axes[0].texts
    plt.close(figure)


def test_close_yield_comparisons_use_zero_based_linear_axes() -> None:
    frame = pd.DataFrame([
        {
            "outer_fold": "pooled",
            "scope": "combined",
            "selection_id": selection,
            "signal_selected_signed_yield": signal,
            "background_selected_signed_yield": background,
        }
        for selection, signal, background in (
            ("nested_outer_oof", 190.0, 50.0e6),
            ("historical_rhh125125_lt34", 185.0, 49.0e6),
            ("fixed_nominal_deployment_cut", 191.0, 52.0e6),
        )
    ])
    figure, _ = plot_yields(frame)
    assert all(axes.get_yscale() == "linear" for axes in figure.axes)
    assert all(axes.get_ylim()[0] == 0.0 for axes in figure.axes)
    plt.close(figure)


def test_conditional_threshold_coordinates_have_publication_labels() -> None:
    frame = pd.DataFrame([
        {
            "is_nominal_structure": True,
            "category_id": "exact3tag",
            "threshold_variable": "ht_candidate_jets",
            "median": 176.0,
            "p16_linear": 170.0,
            "p84_linear": 180.0,
        },
        {
            "is_nominal_structure": True,
            "category_id": "ge4tag",
            "threshold_variable": "abs_h_delta_eta",
            "median": 6.9,
            "p16_linear": 6.0,
            "p84_linear": 7.5,
        },
    ])
    figure, _ = plot_conditional_thresholds(frame)
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert any(r"$H_T^{\mathrm{cand.}}$" in label for label in labels)
    assert any(r"$|\Delta\eta(H_1,H_2)|$" in label for label in labels)
    plt.close(figure)


def test_heterogeneous_modal_sidecar_has_no_trailing_empty_fields() -> None:
    modal = pd.DataFrame([
        {"category_id": "exact3tag", "maximum_winner_count": 1, "frequency": 0.2},
        {"category_id": "ge4tag", "maximum_winner_count": 1, "frequency": 0.3},
    ])
    category = pd.DataFrame([
        {
            "category_id": category_id,
            "unique_modal_replica_category_fraction": 0.6,
            "tie_resolution_replica_category_fraction": 0.4,
        }
        for category_id in ("exact3tag", "ge4tag")
    ])
    figure, sidecar = plot_modal_tie(modal, category)
    serialized = sidecar.to_csv(sep="\t", index=False, lineterminator="\n")
    assert sidecar.columns[-1] == "record_type"
    assert all(not line.endswith("\t") for line in serialized.splitlines())
    plt.close(figure)


def test_required_publication_labels_are_exact() -> None:
    ht = short_structure("radial_mass__category__plus_ht_candidate_jets")
    deta = short_structure("radial_mass__category__plus_abs_h_delta_eta")
    assert r"$H_T^{\mathrm{cand.}}$" in ht
    assert r"$|\Delta\eta(H_1,H_2)|$" in deta


def test_no_official_cms_branding() -> None:
    source = (SCRIPT_DIR / "plot_hh4b_cut_baseline_publication.py").read_text(
        encoding="utf-8"
    )
    assert "CMS Preliminary" not in source
    assert "Delphes simulation" in source


def test_repository_gate_allows_unrelated_prepared_worktree() -> None:
    source = (SCRIPT_DIR / "plot_hh4b_cut_baseline_publication.py").read_text(
        encoding="utf-8"
    )
    verify_body = source.split("def verify_repository", 1)[1].split("def add_header", 1)[0]
    assert "status --porcelain" not in verify_body
    assert "head == remote and branch == BRANCH" in verify_body


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
