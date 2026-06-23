"""Offline synthetic unit tests for the oracle-ceiling analysis math.

These tests exercise ONLY the pure analysis functions in scripts/oracle_ceiling.py.
They make NO live API calls and need no GPU/network — they validate that the
diagnostic is proof against both false positives (routing looks good when it isn't)
and false negatives (routing looks hopeless when it isn't).

Coverage mandated by the plan (docs/ORACLE_CEILING_DIAGNOSTIC.md §7 self-validation):
  (a) 3 disjoint specialists -> routing_oracle = 1.0, headroom ~ 0.667
  (b) 3 identical models     -> headroom = 0
  (c) pure-noise matrix (p=0.5 everywhere) -> headroom CI includes 0 (no FP on noise)
  (d) cross-fit debiasing reduces the max-selection bias vs naive max_m p_hat
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracle_ceiling.py"
_spec = importlib.util.spec_from_file_location("oracle_ceiling", _SCRIPT)
oc = importlib.util.module_from_spec(_spec)
sys.modules["oracle_ceiling"] = oc
_spec.loader.exec_module(oc)


# --------------------------------------------------------------------------- #
# (a) 3 disjoint specialists: each query solved by exactly one (different) model.
#     -> routing_oracle = 1.0, best_single ~ 1/3, headroom ~ 2/3.
# --------------------------------------------------------------------------- #
def _disjoint_specialists(Q=30, K=5):
    S = np.zeros((Q, 3, K))
    for q in range(Q):
        S[q, q % 3, :] = 1.0
    return S


def test_a_disjoint_specialists_oracle_is_one():
    S = _disjoint_specialists()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.routing_oracle == pytest.approx(1.0, abs=1e-9)
    assert st.routing_oracle_naive == pytest.approx(1.0, abs=1e-9)


def test_a_disjoint_specialists_best_single_is_third():
    S = _disjoint_specialists()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.best_single == pytest.approx(1.0 / 3.0, abs=0.05)


def test_a_disjoint_specialists_headroom_two_thirds():
    S = _disjoint_specialists()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.routing_headroom == pytest.approx(2.0 / 3.0, abs=0.05)


def test_a_disjoint_specialists_full_disagreement():
    # Every query: one model right, two wrong -> always a disagreement.
    S = _disjoint_specialists()
    assert oc.disagreement_rate(S) == pytest.approx(1.0, abs=1e-9)


def test_a_disjoint_verdict_router_bound():
    # Huge real headroom, no TRINITY data -> ROUTER_BOUND (routing clearly can help).
    S = _disjoint_specialists()
    rep = oc.analyze_matrix(oc._tensor_to_matrix(S, "disjoint"),
                            n_boot=500, seed=0, crossfit_splits=60)
    assert rep["verdict"]["label"] == "ROUTER_BOUND"


# --------------------------------------------------------------------------- #
# (b) 3 identical models: any per-query outcome, but all three models agree.
#     -> headroom == 0 exactly (cross-fit), and its CI includes 0.
# --------------------------------------------------------------------------- #
def _identical_models(Q=40, K=5, seed=1):
    rng = np.random.default_rng(seed)
    base = (rng.random((Q, 1, K)) < 0.6).astype(float)
    return np.repeat(base, 3, axis=1)


def test_b_identical_models_zero_headroom():
    S = _identical_models()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.routing_headroom == pytest.approx(0.0, abs=1e-9)


def test_b_identical_models_zero_disagreement():
    S = _identical_models()
    assert oc.disagreement_rate(S) == pytest.approx(0.0, abs=1e-9)


def test_b_identical_models_headroom_ci_includes_zero():
    S = _identical_models()
    cis = oc.bootstrap_all(S, n_boot=500, seed=0, crossfit_splits=40)
    h = cis["routing_headroom"]
    assert h["ci_lo"] <= 0.0 <= h["ci_hi"]


def test_b_identical_models_clairvoyant_inflates_but_headroom_zero():
    # clairvoyant_any uses the independence formula 1-prod(1-p), so identical (but
    # noisy) models still inflate it above best_single — that inflation is pure
    # unroutable_noise, NOT routable headroom. The routing headroom stays ~0.
    S = _identical_models()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.clairvoyant_any > st.best_single          # union-over-noise inflation
    assert st.unroutable_noise > 0.0                    # the inflation is noise...
    assert st.routing_headroom == pytest.approx(0.0, abs=1e-9)  # ...not headroom


# --------------------------------------------------------------------------- #
# (c) Pure noise: every (q,m) cell is an independent fair coin (true p=0.5).
#     There is NO routable structure, so the headroom CI must include 0
#     (otherwise the diagnostic fires a false positive on noise).
# --------------------------------------------------------------------------- #
def _pure_noise(Q=120, M=3, K=5, seed=2):
    rng = np.random.default_rng(seed)
    return (rng.random((Q, M, K)) < 0.5).astype(float)


def test_c_pure_noise_headroom_ci_includes_zero():
    S = _pure_noise()
    cis = oc.bootstrap_all(S, n_boot=2000, seed=0, crossfit_splits=60)
    h = cis["routing_headroom"]
    assert h["ci_lo"] <= 0.0 <= h["ci_hi"], (
        f"false positive on noise: headroom CI = [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]"
    )


def test_c_pure_noise_verdict_not_router_bound():
    S = _pure_noise()
    rep = oc.analyze_matrix(oc._tensor_to_matrix(S, "noise"),
                            n_boot=2000, seed=0, crossfit_splits=60)
    # On pure noise the diagnostic must NOT claim routing can help.
    assert rep["verdict"]["label"] != "ROUTER_BOUND"


def test_c_pure_noise_clairvoyant_exceeds_oracle():
    # Union-over-noise inflation: clairvoyant_any (~0.875) >> oracle (~0.5).
    # This is exactly the false-positive trap the plan warns about; we report the
    # gap as unroutable_noise rather than headroom.
    S = _pure_noise()
    st = oc.compute_stats(S, crossfit_splits=100, seed=0)
    assert st.clairvoyant_any > st.routing_oracle + 0.2
    assert st.unroutable_noise > 0.2


# --------------------------------------------------------------------------- #
# (d) Cross-fit debiasing: 3 statistically identical p=0.5 models (drawn
#     independently). The TRUE routing oracle is 0.5 (no model is better on any
#     query in expectation), but naive max_m p_hat is biased UP by the winner's
#     curse. Cross-fit must land closer to 0.5.
# --------------------------------------------------------------------------- #
def _iid_half(Q=200, M=3, K=6, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.random((Q, M, K)) < 0.5).astype(float)


def test_d_naive_max_is_upward_biased():
    S = _iid_half()
    naive = oc.routing_oracle_naive(oc.p_hat(S))
    assert naive > 0.5 + 0.02, f"expected naive max biased above 0.5, got {naive:.3f}"


def test_d_crossfit_reduces_bias():
    S = _iid_half()
    naive = oc.routing_oracle_naive(oc.p_hat(S))
    cf = oc.routing_oracle_crossfit(S, n_splits=300, seed=0)
    # Cross-fit estimate is closer to the true 0.5 than the naive max.
    assert abs(cf - 0.5) < abs(naive - 0.5)
    # And the bias reduction is a clear margin, not a coin-flip.
    assert (naive - 0.5) - abs(cf - 0.5) > 0.02


def test_d_crossfit_falls_back_to_naive_when_k_one():
    # K==1 has no split to cross-fit; the function must fall back to the naive max
    # (and the caller is responsible for noting it is not debiased).
    rng = np.random.default_rng(7)
    S = (rng.random((50, 3, 1)) < 0.5).astype(float)
    cf = oc.routing_oracle_crossfit(S, n_splits=100, seed=0)
    naive = oc.routing_oracle_naive(oc.p_hat(S))
    assert cf == pytest.approx(naive, abs=1e-9)


# --------------------------------------------------------------------------- #
# Supporting math: McNemar, router_gap_closed, threshold sensitivity, I/O.
# --------------------------------------------------------------------------- #
def test_mcnemar_identical_is_p_one():
    mc = oc.mcnemar(np.array([1, 0, 1, 1, 0]), np.array([1, 0, 1, 1, 0]))
    assert mc["n_discordant"] == 0
    assert mc["p_exact"] == pytest.approx(1.0)


def test_mcnemar_fully_discordant_is_significant():
    mc = oc.mcnemar(np.ones(20, int), np.zeros(20, int))
    assert mc["n_discordant"] == 20
    assert mc["p_exact"] < 0.01


def test_mcnemar_known_counts_exact_pvalue():
    # b=8 (A only), c=2 (B only) -> two-sided exact binomial on n=10, k=2.
    a = np.array([1] * 8 + [0] * 2 + [1] * 5)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5)
    mc = oc.mcnemar(a, b)
    assert mc["b_only_a"] == 8 and mc["c_only_b"] == 2 and mc["n_discordant"] == 10
    # 2 * P(X <= 2) for X~Bin(10, 0.5) = 2 * (1+10+45)/1024 = 0.109375.
    assert mc["p_exact"] == pytest.approx(0.109375, abs=1e-6)


def test_router_gap_closed_basic():
    # best=0.6, oracle=0.8, trinity=0.7 -> closed half the 0.2 headroom.
    assert oc.router_gap_closed(0.7, 0.6, 0.8) == pytest.approx(0.5)


def test_router_gap_closed_nan_when_no_headroom():
    g = oc.router_gap_closed(0.6, 0.6, 0.6)
    assert np.isnan(g)


def test_threshold_oracle_matches_hard_definition():
    # Two models, two queries. p>=0.5 => model "solves" the query.
    # q0: m0 solves (p=0.8), m1 not (0.2); q1: m1 solves (0.6), m0 not (0.4).
    p = np.array([[0.8, 0.2], [0.4, 0.6]])
    oracle, bs, head = oc._threshold_oracle(p, thr=0.5)
    assert oracle == pytest.approx(1.0)          # router solves both
    assert bs == pytest.approx(0.5)              # each fixed model solves one
    assert head == pytest.approx(0.5)


def test_matrix_tensor_roundtrip():
    S = _disjoint_specialists(Q=9, K=3)
    matrix = oc._tensor_to_matrix(S, "rt")
    S2, qids, models = oc.matrix_to_tensor(matrix)
    assert S2.shape == S.shape
    assert np.array_equal(S2, S)
    assert len(qids) == 9 and len(models) == 3


def test_matrix_to_tensor_rejects_ragged_k():
    bad = {
        "benchmark": "x", "k": 2, "level": "L0", "seed": 0,
        "tasks": [{"id": "q0", "answer": "a",
                   "per_model": {"m0": [1, 0], "m1": [1]}}],
    }
    with pytest.raises(ValueError):
        oc.matrix_to_tensor(bad)


def test_compute_stats_rejects_non_binary():
    with pytest.raises(ValueError):
        oc.compute_stats(np.full((3, 2, 2), 0.5))


def test_analyze_with_trinity_reports_gap_and_mcnemar():
    # Disjoint specialists (oracle 1.0, best ~1/3). A TRINITY that solves ~half
    # the queries should report a partial gap_closed and a McNemar comparison.
    S = _disjoint_specialists(Q=30, K=5)
    matrix = oc._tensor_to_matrix(S, "disjoint")
    qids = [t["id"] for t in matrix["tasks"]]
    trinity = {q: (1 if i % 2 == 0 else 0) for i, q in enumerate(qids)}
    rep = oc.analyze_matrix(matrix, trinity_per_query=trinity,
                            n_boot=300, seed=0, crossfit_splits=60)
    assert "trinity" in rep
    assert "router_gap_closed" in rep["trinity"]
    assert "mcnemar_vs_best_single" in rep["trinity"]
    assert 0.0 <= rep["trinity"]["accuracy"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    S = _disjoint_specialists(Q=30, K=5)
    pt, lo, hi = oc.bootstrap_ci(S, lambda s: oc.best_single(oc.p_hat(s))[0],
                                 n_boot=500, seed=0)
    assert lo <= pt <= hi
