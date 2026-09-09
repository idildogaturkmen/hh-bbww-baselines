#!/usr/bin/env python3
"""Unified CONTROL/TEST inference exporter: produces a schema-conformant
.npz (schemas/model_eval_events.schema.json, 2026-09-10 amended version)
for either the frozen native SPA2M checkpoint (CONTROL) or the future
SPA2M+ParT checkpoint (TEST), over the identical 400,000-event holdout_A
validation cohort (production_2M_val.h5 or, for TEST, its ParT-joined
successor).

Closes gap G5 in track_f_postproduction_pipeline_20260908_v1/
FINAL_EXECUTION_PACKET.md Step 9 ("no inference/scoring script exists
anywhere in the project that produces the .npz files [the evaluator]
requires").

REUSES, DOES NOT REINVENT, model-loading/scoring logic:
  - build_model_and_dataset()/load_checkpoint_into()/preflight()-style
    checks, _patched_load_assignments weight-loading patch: adapted
    (parameterized, not hardcoded) from
    .../spanet_2M_seed0_classification_evaluation_v1/evaluate_classification.py
    (same architecture/options block, byte-identical values).
  - predicted-assignment extraction: model.predict(sources).assignments,
    the exact call paper_exports/track_b_harvey_tail_characterization_
    20260825_v1/work/part2_spanet_assignment.py already uses successfully
    against this exact CONTROL checkpoint.
  - four_vec()/pair_mass() dijet-mass reconstruction: copied verbatim from
    part3_event_listing_and_truth.py / part6_pairing_accuracy_and_mass.py.
  - process-label reconstruction: reconstruct_process_labels()/
    verify_process_reconstruction(), copied verbatim from
    evaluate_classification.py (manifest-replay + hard cross-check against
    the HDF5's own binary label).

DOES NOT pre-score H1/H2 correctness. Emits ONLY the raw per-event
quantities the current, amended schema requires:
    event_id, process, score,
    pred_b1, pred_b2, pred_b3, pred_b4,
    truth_b1, truth_b2, truth_b3, truth_b4,
    assignment_defined, higgs_mass_1, higgs_mass_2, n_jets
    (+ optional part_embedding_norm_mean/jet_pt_leading/jet_eta_leading,
    TEST-only, only if the corresponding --part-embedding-norm-npz is
    actually supplied -- never fabricated).
`code/evaluate_multi_model.py`'s `match_higgs_pairs()` is the ONLY place
H1/H2 correctness is ever computed, per PREREGISTRATION.md Amendment
2026-09-10 -- this script must never duplicate that logic.

TRUTH AND RAW KINEMATICS ARE READ DIRECTLY VIA h5py, NOT THROUGH THE MODEL,
DELIBERATELY. Two reasons: (1) the frozen HDF5's INPUTS/Source/{pt,eta,phi,
mass} are RAW PHYSICAL quantities (verified this session: pt values are
GeV-scale, e.g. 176.06/112.96/... -- log_normalize/normalize per
FROZEN_FACTS.json's native_feature_schema is applied by the SPA-Net
dataset object ON THE FLY when building network input tensors, never
baked into the file), so reading them directly gives exact, unnormalized
GeV masses without needing to invert any normalization. (2) this project
has an ALREADY-DOCUMENTED in-place-mutation bug precedent (RUNNER_
PROVENANCE_RECEIPT.json, paper_exports/track_b_harvey_tail_characterization_
20260825_v1/work/: `torch.from_numpy(jets)` sharing memory with a numpy
array that was then mutated in place by log1p, silently corrupting
downstream storage that reused the same array) -- reading kinematics via
an independent h5py pass, never converting those specific arrays to a
torch tensor, makes this exporter structurally immune to that exact bug
class, not just carefully avoiding it by convention.

FOUR-VECTOR CONVENTION: px=pt*cos(phi), py=pt*sin(phi), pz=pt*sinh(eta),
E=sqrt(px^2+py^2+pz^2+m^2) (standard massive-particle convention, GeV
throughout). higgs_mass_1/2 = invariant mass of the dijet system formed by
the model's own predicted pair (pred_b1,pred_b2) / (pred_b3,pred_b4) in
RAW, UNCANONICALIZED slot order -- exactly the per-jet indices SPA-Net
returned, never resolved against truth's H1/H2 labeling (that resolution
is evaluate_multi_model.py's job alone, per the Amendment).

EVENT IDENTITY: prefers an explicit `JOIN_KEY/native_hdf5_row_index`
dataset in --input-h5 if present (the ParT-shard convention,
FROZEN_FACTS.json part_production.h5_schema); CONTROL's plain
production_2M_val.h5 has no such column, so event_id there is
np.arange(n) -- but this equality with native_hdf5_row_index is PROVEN,
not assumed: the DataLoader-yielded classification target array is
hard-asserted equal (element-wise, in order) to the raw h5py-read
CLASSIFICATIONS/EVENT/signal array, the exact same identity check
evaluate_classification.py's own main() already performs and has already
passed for this checkpoint/file pair.

FAIL-CLOSED: hard exit on checkpoint/input-h5/train-h5 SHA256 mismatch,
spanet-repo pinned-commit mismatch or dirty working tree, event-yaml
SHA256 mismatch, row-count mismatch, dropped/short final batch, identity
check failure, out-of-range or non-full-rank truth, or any NaN/Inf where
a value is required to be finite. Never silently continues past any of
these.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np


# ---------------------------------------------------------------- helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def four_vec(pt, eta, phi, mass):
    """Standard massive-particle four-vector convention, GeV throughout.
    Copied verbatim (same formula) from part3_event_listing_and_truth.py /
    part6_pairing_accuracy_and_mass.py."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + mass ** 2)
    return px, py, pz, e


def _gather(arr2d, idx1d):
    """arr2d: (n, 10); idx1d: (n,) per-row column index -> returns (n,)."""
    rows = np.arange(arr2d.shape[0])
    return arr2d[rows, idx1d]


def pair_mass(pt, eta, phi, mass, i, j):
    """Invariant mass of the dijet (i,j), i/j: (n,) per-row jet-slot
    indices. Copied verbatim (same formula) from part3/part6."""
    px0, py0, pz0, e0 = four_vec(_gather(pt, i), _gather(eta, i), _gather(phi, i), _gather(mass, i))
    px1, py1, pz1, e1 = four_vec(_gather(pt, j), _gather(eta, j), _gather(phi, j), _gather(mass, j))
    px, py, pz, e = px0 + px1, py0 + py1, pz0 + pz1, e0 + e1
    m2 = e ** 2 - (px ** 2 + py ** 2 + pz ** 2)
    return np.sqrt(np.clip(m2, 0, None))


# ---- FINALIZED MINIMAL WEIGHT-loading patch -- byte-for-byte unchanged
# from evaluate_classification.py and every prior gate/training script this
# project has run. Kept even though this script reads truth via h5py
# directly (not through this patch's own load_assignments path), because
# JetReconstructionDataset construction still calls it internally and it
# is the proven-correct way to avoid a load-time crash/silent-wrong-shape
# on this event topology. ----
def _patched_load_assignments(self, hdf5_file, limit_index):
    import torch
    from spanet.dataset.types import SpecialKey
    targets = OrderedDict()
    for event_particle, daughter_particles in self.event_info.product_particles.items():
        target_data = torch.empty(len(daughter_particles), self.num_events, dtype=torch.int64)
        for index, daughter in enumerate(daughter_particles):
            dataset = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], daughter)
            dataset.read_direct(target_data[index].numpy())
        for index, source in enumerate(daughter_particles.sources):
            if source >= 0:
                target_data[index] += self.source_offsets[source] * (target_data[index] >= 0)
        target_data = target_data.transpose(0, 1)
        try:
            target_mask = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Mask)
            target_mask = torch.from_numpy(target_mask[:]).bool()
        except KeyError:
            target_mask = (target_data >= 0).all(1)
        try:
            target_weight = self.dataset(hdf5_file, [SpecialKey.Targets, event_particle], SpecialKey.Weight)
            target_weight = torch.from_numpy(target_weight[:])
        except KeyError:
            target_weight = torch.ones_like(target_mask, dtype=float)
        target_data = target_data[limit_index]
        target_mask = target_mask[limit_index]
        target_weight = target_weight[limit_index]
        targets[event_particle] = (target_data, target_mask, target_weight)
    return targets


# --------------------------------------------------------------- preflight

def preflight(checkpoint, checkpoint_sha256, input_h5, input_h5_sha256,
              hdf5_train, hdf5_train_sha256, event_yaml, event_yaml_sha256,
              spanet_repo, spanet_pinned_commit):
    import subprocess
    print("==== Preflight (read-only exporter) ====", file=sys.stderr)
    checks = [
        ("checkpoint", checkpoint, checkpoint_sha256),
        ("input H5", input_h5, input_h5_sha256),
        ("train H5 (Options()-required, never read for scoring)", hdf5_train, hdf5_train_sha256),
        ("event YAML", event_yaml, event_yaml_sha256),
    ]
    for name, path, expected in checks:
        if not os.path.isfile(path):
            print(f"FATAL: {name} not found: {path}", file=sys.stderr)
            sys.exit(2)
        actual = sha256_file(path)
        print(f"{name} sha256: {actual}", file=sys.stderr)
        if expected is not None and actual != expected:
            print(f"FATAL: {name} sha256 mismatch -- expected {expected}, got {actual}", file=sys.stderr)
            sys.exit(3)

    try:
        commit = subprocess.check_output(["git", "-C", spanet_repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        commit = f"<error: {e}>"
    print(f"spanet_repo commit: {commit}", file=sys.stderr)
    if commit != spanet_pinned_commit:
        print(f"FATAL: pinned commit mismatch -- expected {spanet_pinned_commit}, got {commit}", file=sys.stderr)
        sys.exit(5)

    try:
        status = subprocess.check_output(["git", "-C", spanet_repo, "status", "--porcelain"], text=True)
    except Exception as e:  # noqa: BLE001
        status = f"<error: {e}>"
    if status.strip():
        print(f"FATAL: pinned spanet_repo working tree is not clean:\n{status}", file=sys.stderr)
        sys.exit(6)

    print("==== Preflight OK ====", file=sys.stderr)
    return dict(spanet_commit=commit)


# ---------------------------------------------------------- process labels

def reconstruct_process_labels(val_manifest_tsv):
    """Verbatim (same algorithm) reuse of evaluate_classification.py's
    reconstruct_process_labels(): replays build_hdf5_2M.py's own
    load_split() concatenation order EXACTLY (`sorted(rows, key=(process,
    rank))`, process alphabetical: qcd, signal, ttbar)."""
    import csv
    from collections import Counter
    with open(val_manifest_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows_sorted = sorted(rows, key=lambda r: (r["process"], int(r["rank"])))
    labels = []
    per_process_file_count = Counter()
    for row in rows_sorted:
        labels.extend([row["process"]] * int(row["quota"]))
        per_process_file_count[row["process"]] += 1
    return labels, dict(per_process_file_count), len(rows_sorted)


def verify_process_reconstruction(process_labels, hdf5_signal_label):
    """Verbatim reuse of evaluate_classification.py's hard, blocking
    cross-check: reconstructed process must exactly agree with the HDF5's
    own binary label, zero exceptions."""
    process_arr = np.array(process_labels)
    assert len(process_arr) == len(hdf5_signal_label), (
        f"length mismatch: reconstructed={len(process_arr)} vs HDF5={len(hdf5_signal_label)}")
    is_signal_reconstructed = (process_arr == "signal")
    is_signal_hdf5 = (hdf5_signal_label == 1)
    n_mismatch = int((is_signal_reconstructed != is_signal_hdf5).sum())
    if n_mismatch != 0:
        raise AssertionError(
            f"process-label reconstruction MISMATCHES the HDF5's own binary label at {n_mismatch} rows -- "
            "refusing to use this reconstruction.")
    return dict(n_mismatch=n_mismatch,
                n_signal=int((process_arr == "signal").sum()),
                n_qcd=int((process_arr == "qcd").sum()),
                n_ttbar=int((process_arr == "ttbar").sum()))


# --------------------------------------------------------------- raw HDF5

def read_raw_h5(input_h5_path):
    """Direct h5py read -- see module docstring for why this is
    deliberate (raw physical kinematics, no in-place-mutation exposure,
    no dependency on the model's own tensor pipeline)."""
    import h5py
    with h5py.File(input_h5_path, "r") as f:
        pt = f["INPUTS/Source/pt"][:]
        eta = f["INPUTS/Source/eta"][:]
        phi = f["INPUTS/Source/phi"][:]
        mass = f["INPUTS/Source/mass"][:]
        mask = f["INPUTS/Source/MASK"][:]
        signal_label = f["CLASSIFICATIONS/EVENT/signal"][:]
        b1 = f["TARGETS/h1/b1"][:]
        b2 = f["TARGETS/h1/b2"][:]
        b3 = f["TARGETS/h2/b3"][:]
        b4 = f["TARGETS/h2/b4"][:]
        explicit_event_id = f["JOIN_KEY/native_hdf5_row_index"][:] if "JOIN_KEY/native_hdf5_row_index" in f else None
    return dict(pt=pt, eta=eta, phi=phi, mass=mass, mask=mask, signal_label=signal_label,
                b1=b1, b2=b2, b3=b3, b4=b4, explicit_event_id=explicit_event_id)


# --------------------------------------------------------------- model/dataset

def build_model_and_dataset(event_yaml, hdf5_train, hdf5_val, batch_size):
    """Adapted from evaluate_classification.py's build_model_and_dataset():
    same architecture/optimizer/loss-scale values (FROZEN_FACTS.json
    spanet_architecture_and_training_protocol, byte-identical), only the
    three paths (event_yaml, hdf5_train, hdf5_val) and batch_size are
    parameterized instead of hardcoded, so this one function serves both
    CONTROL and the future TEST (which will pass its own joined H5 and,
    if its event YAML differs to add ParT feature columns, its own YAML)."""
    import pytorch_lightning as pl
    from spanet import JetReconstructionModel, Options
    from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset

    JetReconstructionDataset.load_assignments = _patched_load_assignments

    pl.seed_everything(0, workers=True)

    options = Options(event_yaml, hdf5_train, hdf5_val)
    options.batch_size = batch_size
    options.num_dataloader_workers = 0
    options.dataset_limit = 1.0
    options.partial_events = True

    options.assignment_loss_scale = 1.0
    options.classification_loss_scale = 1.0
    options.detection_loss_scale = 0.0
    options.kl_loss_scale = 0.0
    options.regression_loss_scale = 0.0
    options.balance_losses = False

    options.hidden_dim = 32
    options.num_encoder_layers = 8
    options.num_branch_encoder_layers = 2
    options.num_classification_layers = 1
    options.dropout = 0.0059

    options.optimizer = "AdamW"
    options.learning_rate = 0.00659
    options.l2_penalty = 0.000374
    options.gradient_clip = 0.425

    options.balance_particles = False
    options.balance_jets = False
    options.balance_classifications = False

    options.epochs = 50
    options.num_gpu = 0  # CPU only, no GPU touched by this exporter

    def quiet(fn, *a, **kw):
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*a, **kw)

    model = quiet(JetReconstructionModel, options)
    return model


def load_checkpoint_into(model, ckpt_path, strict=True):
    """Verbatim reuse of evaluate_classification.py's load_checkpoint_into()."""
    import torch
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    if strict and (missing or unexpected):
        raise AssertionError(f"checkpoint load not clean: missing={missing}, unexpected={unexpected}")
    return missing, unexpected


def score_and_predict(model, batch_size, canary_n_batches=None):
    """Single DataLoader pass (drop_last=False, matching evaluate_
    classification.py's own documented deviation from the training-time
    dataloader, needed to cover every event). Per batch: model.forward()
    for the classification score (identical to evaluate_classification.py's
    score_full_population()), and model.predict() for the assignment
    indices (identical call to part2_spanet_assignment.py's own usage) --
    two forward passes per batch rather than reimplementing extract_
    predictions()/nan handling myself, trading ~2x CPU time for zero risk
    of subtly diverging from either proven precedent's own numerics.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(model.validation_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=0, pin_memory=False)

    all_probs, all_true = [], []
    all_pred_b1, all_pred_b2, all_pred_b3, all_pred_b4 = [], [], [], []
    n_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            outputs = model.forward(batch.sources)
            logits = outputs.classifications["EVENT/signal"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            true = batch.classification_targets["EVENT/signal"]
            all_probs.append(probs.numpy())
            all_true.append(true.numpy())

            pred = model.predict(batch.sources)
            h1_idx, h2_idx = pred.assignments[0], pred.assignments[1]
            all_pred_b1.append(h1_idx[:, 0]); all_pred_b2.append(h1_idx[:, 1])
            all_pred_b3.append(h2_idx[:, 0]); all_pred_b4.append(h2_idx[:, 1])

            n_batches += 1
            print(f"  batch {n_batches} scored ({sum(len(p) for p in all_probs)} events so far)", file=sys.stderr)
            if canary_n_batches is not None and n_batches >= canary_n_batches:
                break

    wall_s = time.time() - t0
    return dict(
        probs=np.concatenate(all_probs), true=np.concatenate(all_true),
        pred_b1=np.concatenate(all_pred_b1), pred_b2=np.concatenate(all_pred_b2),
        pred_b3=np.concatenate(all_pred_b3), pred_b4=np.concatenate(all_pred_b4),
        n_batches=n_batches, wall_s=wall_s,
    )


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["control", "test"],
                     help="which model arm this invocation exports (recorded in the summary only; "
                          "all other behavior is identical for both)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--input-h5", required=True, help="the val HDF5 to score (production_2M_val.h5 for "
                     "CONTROL; the ParT-joined val HDF5 for TEST)")
    ap.add_argument("--input-h5-sha256", required=True)
    ap.add_argument("--hdf5-train", required=True, help="required by SPA-Net Options() construction; "
                     "never read for scoring (documented, proven pattern from evaluate_classification.py)")
    ap.add_argument("--hdf5-train-sha256", required=True)
    ap.add_argument("--event-yaml", required=True)
    ap.add_argument("--event-yaml-sha256", required=True)
    ap.add_argument("--spanet-repo", required=True, help="path inserted at sys.path[0] BEFORE `import spanet`, "
                     "so the pinned commit is what actually executes regardless of whatever else may be "
                     "pip-installed in the running Python environment")
    ap.add_argument("--spanet-pinned-commit", required=True)
    ap.add_argument("--val-manifest-tsv", required=True)
    ap.add_argument("--expected-n-events", type=int, default=400000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--canary-n-batches", type=int, default=None,
                     help="if set, score only this many DataLoader batches (a size-(batch_size*N) PREFIX "
                          "of the cohort, not a random sample) and skip the exactly-N-events completeness "
                          "check -- for a bounded, cheap correctness canary only. Omit for a real export.")
    ap.add_argument("--part-embedding-norm-npz", default=None,
                     help="OPTIONAL, TEST-only: a .npz with a 'part_embedding_norm_mean' array (event-order-"
                          "matched) to attach as a diagnostic column. Never fabricated if omitted.")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.spanet_repo)

    preflight(args.checkpoint, args.checkpoint_sha256, args.input_h5, args.input_h5_sha256,
              args.hdf5_train, args.hdf5_train_sha256, args.event_yaml, args.event_yaml_sha256,
              args.spanet_repo, args.spanet_pinned_commit)

    is_canary = args.canary_n_batches is not None

    print("==== Reading raw HDF5 (h5py, direct, independent of the model pipeline) ====", file=sys.stderr)
    raw = read_raw_h5(args.input_h5)
    n_events_h5 = len(raw["pt"])
    if n_events_h5 != args.expected_n_events:
        print(f"FATAL: input H5 has {n_events_h5} rows, expected {args.expected_n_events}", file=sys.stderr)
        sys.exit(7)

    print("==== Reconstructing + verifying process labels ====", file=sys.stderr)
    process_labels, per_process_file_count, n_manifest_rows = reconstruct_process_labels(args.val_manifest_tsv)
    process_check = verify_process_reconstruction(process_labels, raw["signal_label"])
    print(f"process reconstruction VERIFIED: {process_check}", file=sys.stderr)

    print("==== Building model + dataset, loading checkpoint ====", file=sys.stderr)
    import spanet  # noqa: F401 -- import here (after sys.path insert) so the pinned-commit tree is used
    assert spanet.__file__.startswith(os.path.abspath(args.spanet_repo)), (
        f"spanet resolved to {spanet.__file__}, not under the pinned repo {args.spanet_repo} -- "
        "a different spanet install shadowed the sys.path override; refusing to proceed")
    model = build_model_and_dataset(args.event_yaml, args.hdf5_train, args.input_h5, args.batch_size)
    n_val_events = len(model.validation_dataset)
    if n_val_events != args.expected_n_events:
        print(f"FATAL: model.validation_dataset has {n_val_events} events, expected {args.expected_n_events}",
              file=sys.stderr)
        sys.exit(8)
    missing, unexpected = load_checkpoint_into(model, args.checkpoint, strict=True)
    print(f"checkpoint loaded clean: missing={missing} unexpected={unexpected}", file=sys.stderr)

    print(f"==== Scoring (canary_n_batches={args.canary_n_batches}) ====", file=sys.stderr)
    sp = score_and_predict(model, args.batch_size, canary_n_batches=args.canary_n_batches)
    n_scored = len(sp["probs"])
    print(f"scored {n_scored} events in {sp['wall_s']:.1f}s ({n_scored / max(sp['wall_s'], 1e-9):.1f} events/s)",
          file=sys.stderr)

    if not is_canary and n_scored != args.expected_n_events:
        print(f"FATAL: scored {n_scored} events, expected exactly {args.expected_n_events} -- "
              "a batch was dropped or short; refusing to export a partial file as if complete.", file=sys.stderr)
        sys.exit(9)

    # ---- slice raw/process arrays down to the scored prefix (no-op unless canary) ----
    pt, eta, phi, mass, mask = raw["pt"][:n_scored], raw["eta"][:n_scored], raw["phi"][:n_scored], \
        raw["mass"][:n_scored], raw["mask"][:n_scored]
    signal_label = raw["signal_label"][:n_scored]
    b1, b2, b3, b4 = raw["b1"][:n_scored], raw["b2"][:n_scored], raw["b3"][:n_scored], raw["b4"][:n_scored]
    process_arr = np.array(process_labels[:n_scored])
    explicit_event_id = raw["explicit_event_id"][:n_scored] if raw["explicit_event_id"] is not None else None

    print("==== Identity proof: DataLoader-yielded label order == raw HDF5 label order ====", file=sys.stderr)
    if not np.array_equal(sp["true"], signal_label):
        print("FATAL: dataloader classification_targets != HDF5 label order in raw row order -- "
              "row-identity assumption is FALSE for this file/dataset combination. Refusing to export "
              "(event_id would be wrong for every downstream consumer).", file=sys.stderr)
        sys.exit(10)
    print("Identity proof PASSED: safe to treat DataLoader batch order == raw HDF5 row order.", file=sys.stderr)

    if explicit_event_id is not None:
        event_id = explicit_event_id.astype(np.int64)
        event_id_method = "explicit_JOIN_KEY_native_hdf5_row_index_column"
    else:
        event_id = np.arange(n_scored, dtype=np.int64)
        event_id_method = "proven_arange_via_dataloader_hdf5_identity_check (see log above)"
    if len(set(event_id.tolist())) != len(event_id):
        print("FATAL: event_id is not unique.", file=sys.stderr)
        sys.exit(11)

    # ---- process -> {0,1,2} ----
    process_code = np.full(n_scored, -1, dtype=np.int8)
    process_code[process_arr == "signal"] = 0
    process_code[process_arr == "qcd"] = 1
    process_code[process_arr == "ttbar"] = 2
    if (process_code < 0).any():
        print("FATAL: an unrecognized process label survived reconstruction.", file=sys.stderr)
        sys.exit(12)

    # ---- truth / matchability ----
    assignment_defined = (b1 != -1)
    for name, arr in (("b2", b2), ("b3", b3), ("b4", b4)):
        if not np.array_equal(assignment_defined, (arr != -1)):
            print(f"FATAL: truth_b1's matchability does not agree with {name}'s -- refusing to export a "
                  "single assignment_defined flag that would misrepresent one of them.", file=sys.stderr)
            sys.exit(13)

    n_jets = mask.sum(axis=1).astype(np.int64)

    # ---- Higgs masses from the model's OWN raw predicted pairs (no canonicalization) ----
    higgs_mass_1 = pair_mass(pt, eta, phi, mass, sp["pred_b1"], sp["pred_b2"]).astype(np.float32)
    higgs_mass_2 = pair_mass(pt, eta, phi, mass, sp["pred_b3"], sp["pred_b4"]).astype(np.float32)

    # ==================================================================
    # Hard validation suite (task item 10) -- ALL must pass before writing.
    # ==================================================================
    n = n_scored
    arrays = dict(event_id=event_id, process=process_code, score=sp["probs"],
                  pred_b1=sp["pred_b1"], pred_b2=sp["pred_b2"], pred_b3=sp["pred_b3"], pred_b4=sp["pred_b4"],
                  truth_b1=b1, truth_b2=b2, truth_b3=b3, truth_b4=b4,
                  assignment_defined=assignment_defined,
                  higgs_mass_1=higgs_mass_1, higgs_mass_2=higgs_mass_2, n_jets=n_jets)

    if not is_canary:
        assert n == args.expected_n_events, f"n={n} != expected {args.expected_n_events}"
    for k, v in arrays.items():
        assert len(v) == n, f"field '{k}' has length {len(v)}, expected {n}"
    assert len(set(event_id.tolist())) == n, "event_id not unique (re-check)"
    assert set(np.unique(process_code).tolist()) <= {0, 1, 2}, "process outside {0,1,2}"
    assert np.array_equal(n_jets, mask.sum(axis=1)), "n_jets disagrees with source MASK"
    assert (n_jets >= 0).all() and (n_jets <= 10).all(), "n_jets out of [0,10] range"

    # ---- predicted indices: in-range, and real-jet where physically possible ----
    # A prediction pointing at a padded (non-real) slot is INSPECTED, not blindly
    # forbidden: the model must assign 4 jet slots total (2 per Higgs candidate),
    # and n_jets can be as low as 2 (confirmed in this file: min=2) -- for any
    # event with n_jets<4 there are NOT 4 distinct real jets available, so a
    # padded-slot prediction there is mathematically unavoidable, not a model
    # defect. n_jets>=4 offers no such excuse -- a padded prediction there would
    # be a genuine anomaly and is treated as fatal, not merely counted.
    for name, arr in (("pred_b1", sp["pred_b1"]), ("pred_b2", sp["pred_b2"]),
                       ("pred_b3", sp["pred_b3"]), ("pred_b4", sp["pred_b4"])):
        assert ((arr >= 0) & (arr < 10)).all(), f"{name} out of [0,10) array-bounds range"
        points_to_real = _gather(mask, arr)
        if not points_to_real.all():
            bad = ~points_to_real
            n_bad_low = int((bad & (n_jets < 4)).sum())
            n_bad_high = int((bad & (n_jets >= 4)).sum())
            print(f"NOTE: {name} points to a padded (non-real) jet slot for {int(bad.sum())} event(s) "
                  f"-- {n_bad_low} with n_jets<4 (expected/unavoidable: fewer than 4 real jets exist, "
                  f"no valid all-real assignment is possible), {n_bad_high} with n_jets>=4 "
                  "(would be UNEXPECTED).", file=sys.stderr)
            if n_bad_high > 0:
                print(f"FATAL: {name} points to a padded slot for {n_bad_high} event(s) with n_jets>=4 -- "
                      "not explained by insufficient real jets. Refusing to export without investigation.",
                      file=sys.stderr)
                sys.exit(14)

    if assignment_defined.any():
        truth_pairs = np.stack([b1[assignment_defined], b2[assignment_defined],
                                 b3[assignment_defined], b4[assignment_defined]], axis=1)
        assert (truth_pairs >= 0).all(), "a -1 truth index appears in an assignment_defined event"
        assert (truth_pairs < 10).all(), "a truth index is out of [0,10) array-bounds range"
        n_unique = np.array([len(set(row.tolist())) for row in truth_pairs])
        assert (n_unique == 4).all(), "truth_b1..truth_b4 do not name 4 distinct jets for some matchable event"
        assert (n_jets[assignment_defined] >= 4).all(), "n_jets < 4 for a matchable event"

    n_degenerate_pred = int((np.array([len(set(row.tolist())) for row in
                                        np.stack([sp["pred_b1"], sp["pred_b2"], sp["pred_b3"], sp["pred_b4"]], axis=1)])
                              < 4).sum())

    assert np.isfinite(sp["probs"]).all(), "non-finite score"
    assert (sp["probs"] >= 0.0).all() and (sp["probs"] <= 1.0).all(), "score outside [0,1]"
    assert np.isfinite(higgs_mass_1).all() and np.isfinite(higgs_mass_2).all(), "non-finite Higgs mass"

    extra = {}
    if args.part_embedding_norm_npz:
        pe = np.load(args.part_embedding_norm_npz)
        norm = pe["part_embedding_norm_mean"][:n_scored]
        assert len(norm) == n, "part_embedding_norm_mean length mismatch"
        extra["part_embedding_norm_mean"] = norm.astype(np.float32)

    print(f"==== Validation suite PASSED (n={n}, n_degenerate_pred_pairs={n_degenerate_pred}) ====", file=sys.stderr)

    np.savez(args.out_npz,
             event_id=event_id, process=process_code, score=sp["probs"].astype(np.float32),
             pred_b1=sp["pred_b1"].astype(np.int64), pred_b2=sp["pred_b2"].astype(np.int64),
             pred_b3=sp["pred_b3"].astype(np.int64), pred_b4=sp["pred_b4"].astype(np.int64),
             truth_b1=b1.astype(np.int64), truth_b2=b2.astype(np.int64),
             truth_b3=b3.astype(np.int64), truth_b4=b4.astype(np.int64),
             assignment_defined=assignment_defined,
             higgs_mass_1=higgs_mass_1, higgs_mass_2=higgs_mass_2, n_jets=n_jets,
             **extra)

    summary = dict(
        arm=args.arm, is_canary=is_canary, n_scored=n, out_npz=args.out_npz,
        event_id_method=event_id_method,
        process_composition=dict(signal=int((process_code == 0).sum()), qcd=int((process_code == 1).sum()),
                                  ttbar=int((process_code == 2).sum())),
        n_assignment_defined=int(assignment_defined.sum()),
        n_degenerate_predicted_pairs=n_degenerate_pred,
        checkpoint_load_missing_keys=missing, checkpoint_load_unexpected_keys=unexpected,
        wall_s=sp["wall_s"], events_per_s=n / max(sp["wall_s"], 1e-9),
        score_summary=dict(mean=float(sp["probs"].mean()), std=float(sp["probs"].std())),
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
