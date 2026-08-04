from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import tarfile

import pandas as pd
import pytest

from scripts.analysis import extract_pn_c7x_spanet_signal_truth as extractor
from scripts.analysis.pn_c7x_spanet_common import adler32, sha256


REPO = Path(__file__).resolve().parents[1]


def test_extractor_import_is_uproot_lazy_and_memmap_explicit():
    path = REPO / "scripts/analysis/extract_pn_c7x_spanet_signal_truth.py"
    source = path.read_text()
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "uproot" not in top_level_imports
    assert "from uproot.source.file import MemmapSource" in source
    assert "uproot.open(root_path, handler=MemmapSource)" in source


def test_frozen_registry_contains_exact_train_signal_scope_only():
    registry = pd.read_csv(extractor.ROOT_REGISTRY, sep="\t")
    signal = registry[registry["sample_class"].eq("signal")]
    assert len(registry) == 441
    assert len(signal) == 87
    assert signal["source_access_mode"].eq("remote_bundle_extraction_required").all()
    assert set(signal["source_checksum_kind"]) == {"bundle_adler32_size", "bundle_sha256"}
    assert not registry["validation_or_test_payload_opened"].astype(bool).any()
    assert sha256(extractor.ROOT_REGISTRY) == extractor.ROOT_REGISTRY_SHA256


def test_checksum_verification_supports_both_frozen_kinds(tmp_path):
    payload = tmp_path / "bundle.tar.gz"
    payload.write_bytes(b"checksum-bound-train-signal")
    for kind, expected in (
        ("bundle_adler32_size", adler32(payload)),
        ("bundle_sha256", hashlib.sha256(payload.read_bytes()).hexdigest()),
    ):
        row = pd.Series({
            "transport_id": "signal_source",
            "source_size_bytes": payload.stat().st_size,
            "source_checksum_kind": kind,
            "source_checksum": expected,
        })
        evidence = extractor.verify_bundle(payload, row)
        assert evidence["checksum_observed"] == expected
        assert evidence["size_observed"] == payload.stat().st_size


def test_checksum_mismatch_fails_closed(tmp_path):
    payload = tmp_path / "bundle.tar.gz"
    payload.write_bytes(b"wrong")
    row = pd.Series({
        "transport_id": "signal_source",
        "source_size_bytes": payload.stat().st_size,
        "source_checksum_kind": "bundle_sha256",
        "source_checksum": "0" * 64,
    })
    with pytest.raises(RuntimeError, match="checksum drift"):
        extractor.verify_bundle(payload, row)


def test_exact_nested_root_member_is_extracted_without_extractall(tmp_path):
    inner = tmp_path / "reconstruction.tar.gz"
    root_payload = tmp_path / "expected.root"
    root_payload.write_bytes(b"synthetic-root-payload")
    with tarfile.open(inner, "w:gz") as archive:
        archive.add(root_payload, arcname="root/expected.root")
    outer = tmp_path / "bundle.tar.gz"
    with tarfile.open(outer, "w:gz") as archive:
        archive.add(inner, arcname="./products/source_reconstruction.tar.gz")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    row = pd.Series({
        "root_archive_member": "root/expected.root",
        "root_basename": "expected.root",
        "layout_rule": "nested_reconstruction_bundle_source_root_basename",
    })
    extracted, digest = extractor.extract_exact_root(outer, row, scratch)
    assert extracted.read_bytes() == b"synthetic-root-payload"
    assert digest == sha256(extracted)
    assert "extractall" not in (REPO / "scripts/analysis/extract_pn_c7x_spanet_signal_truth.py").read_text()


def test_exact_direct_root_member_uses_registry_layout_rule(tmp_path):
    root_payload = tmp_path / "expected.root"
    root_payload.write_bytes(b"direct-root-payload")
    outer = tmp_path / "bundle.tar.gz"
    with tarfile.open(outer, "w:gz") as archive:
        archive.add(root_payload, arcname="root/expected.root")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    row = pd.Series({
        "root_archive_member": "root/expected.root",
        "root_basename": "expected.root",
        "layout_rule": "standard_bundle_root_target_tag",
    })
    extracted, digest = extractor.extract_exact_root(outer, row, scratch)
    assert extracted.read_bytes() == b"direct-root-payload"
    assert digest == sha256(extracted)


def test_no_validation_test_observed_or_remote_mutation_commands_are_present():
    paths = [
        REPO / "scripts/analysis/extract_pn_c7x_spanet_signal_truth.py",
        REPO / "scripts/analysis/pn_c7x_spanet_common.py",
        REPO / "docs/paper/jhep_hh4b_ml/HIG_24_015_SINGLE_HEAD_SPANET_CONTRACT.md",
    ]
    combined = "\n".join(path.read_text() for path in paths).lower()
    assert "git push" not in combined
    assert "git force-push" not in combined
    assert "observed data opened" not in combined
    assert "tables/train_fourb_model_development.parquet" in combined
    assert "validation_payload_files_opened\": 0" in combined
    assert "test_or_evaluation_payload_files_opened\": 0" in combined


def test_reference_pdfs_are_not_tracked():
    completed = subprocess.run(
        ["git", "ls-files", "*.pdf"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE
    )
    names = {Path(line).name for line in completed.stdout.splitlines()}
    assert "HIG-24-015-paper-v6 y.pdf" not in names
    assert "HIG-24-015-pas-v13 (1).pdf" not in names
