"""Exact maximum-cardinality, minimum-total-DeltaR bipartite matching
between HH-origin truth b-hadrons and selected reco jets, subject to
DeltaR < dr_max.

Primary objective: maximize the number of matched truth objects.
Secondary objective: among all matchings achieving that maximum, minimize
the summed DeltaR of the matched pairs.

Implemented as a bitmask DP over (truth_index, bitmask_of_used_jets):
n_truth <= 8 and n_jets <= 10 in this dataset, so the full state space is
at most 8 * 2**10 = 8192 states per event -- exact by construction (full
enumeration via memoized recursion, no heuristic/greedy step), and fast
enough to run on all ~27k events without any external package.
"""
import numpy as np


def exact_match(dR, n_truth, n_jets, dr_max=0.4):
    """dR: (n_truth, n_jets) float matrix. Returns (assignment, n_matched,
    total_dr) where assignment[t] = matched jet index or -1."""
    if n_truth == 0 or n_jets == 0:
        return np.full(n_truth, -1, dtype=np.int64), 0, 0.0

    # candidate edges per truth object (jet index, dR) with dR < dr_max
    edges = [[(j, dR[t, j]) for j in range(n_jets) if dR[t, j] < dr_max] for t in range(n_truth)]

    memo = {}

    def solve(t_idx, used_mask):
        key = (t_idx, used_mask)
        if key in memo:
            return memo[key]
        if t_idx == n_truth:
            memo[key] = (0, 0.0, ())
            return memo[key]
        skip_res = solve(t_idx + 1, used_mask)
        best = (skip_res[0], skip_res[1], (-1,) + skip_res[2])
        for j, d in edges[t_idx]:
            if used_mask & (1 << j):
                continue
            cnt2, tot2, rest2 = solve(t_idx + 1, used_mask | (1 << j))
            cand_count = cnt2 + 1
            cand_tot = tot2 + d
            if (cand_count > best[0]) or (cand_count == best[0] and cand_tot < best[1]):
                best = (cand_count, cand_tot, (j,) + rest2)
        memo[key] = best
        return best

    n_matched, total_dr, choice_seq = solve(0, 0)
    assignment = np.full(n_truth, -1, dtype=np.int64)
    for t, j in enumerate(choice_seq):
        assignment[t] = j
    return assignment, n_matched, total_dr


def greedy_match(dR, n_truth, n_jets, dr_max=0.4):
    """Global edge-sorted greedy one-to-one matching (bijective, but NOT
    proven globally minimal in total DeltaR) -- kept for the explicit
    GREEDY vs EXACT comparison required by Correction 3a."""
    assignment = np.full(n_truth, -1, dtype=np.int64)
    if n_truth == 0 or n_jets == 0:
        return assignment, 0, 0.0
    cands = [(dR[t, j], t, j) for t in range(n_truth) for j in range(n_jets) if dR[t, j] < dr_max]
    cands.sort(key=lambda x: x[0])
    used_t, used_j = set(), set()
    total_dr = 0.0
    n_matched = 0
    for d, t, j in cands:
        if t in used_t or j in used_j:
            continue
        used_t.add(t)
        used_j.add(j)
        assignment[t] = j
        total_dr += d
        n_matched += 1
    return assignment, n_matched, total_dr
