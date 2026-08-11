from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from plot_hh4b_cut_baseline_publication import add_header, plt, short_structure  # noqa: E402


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
