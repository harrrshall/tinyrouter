#!/usr/bin/env python3
"""Oracle-ceiling diagnostic + pool-complementarity audit.

Implements recommendation #1 of IMPROVEMENTS.md (see docs/ORACLE_CEILING_DIAGNOSTIC.md).
Answers ONE question without lying: on the current 3-model pool, is there enough
routable complementarity that a better router could beat the best single model, or is
the ceiling so close to best-single that the only real lever is the pool itself?

Three modes:

  --collect   live: draw K samples per (query, model), grade with the FIXED grader,
              and persist every API call to JSONL plus a compact per-(query,model)
              matrix. Sets the cost ledger and enforces a --max-cost-usd cap.

  --analyze   offline: read a matrix JSON and compute best_single, routing_oracle
              (cross-fit / winner's-curse-debiased), clairvoyant_any (reported
              separately, NOT routing-achievable), routing_headroom, unroutable_noise,
              router_gap_closed, bootstrap CIs, paired McNemar, and a
              threshold-sensitivity report. Writes an oracle report JSON.

  --selftest  offline: synthetic unit tests of the analysis math (no API calls).

Design contract: the analysis math (this module's pure functions) has NO network and
NO torch/GPU dependency, so it is unit-testable on the dev box. Only --collect touches
the network, and it reuses FireworksPool (retry/concurrency/cost-ledger) + reward.score_text
(the FIXED grader) — it does NOT reinvent extraction.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
# Make the package importable when run as a plain script (python scripts/oracle_ceiling.py).
sys.path.insert(0, str(_REPO / "src"))


# =============================================================================
# Analysis core (pure, offline, unit-tested). Everything below operates on a
# "solves" tensor S of shape (Q, M, K) with S[q, m, k] in {0, 1}: did model m
# solve query q on independent sample k? p_hat[q, m] = mean_k S[q, m, k].
# =============================================================================
@dataclass
class CeilingStats:
    """Point estimates of the ceiling quantities. CIs are computed separately."""

    n_queries: int
    n_models: int
    k: int
    per_model: list[float]                 # mean_q p_hat[q, m], one per model
    best_single: float                     # max_m per_model
    best_single_model: int                 # argmax
    routing_oracle: float                  # winner's-curse-debiased (cross-fit) when K>=2
    routing_oracle_naive: float            # mean_q max_m p_hat[q, m] (upward-biased)
    clairvoyant_any: float                 # mean_q (1 - prod_m (1 - p_hat)) — NOT achievable
    routing_headroom: float                # routing_oracle - best_single
    unroutable_noise: float                # clairvoyant_any - routing_oracle
    disagreement_rate: float               # fraction of q where models don't all agree
    routing_oracle_thresh: float           # hard p>=0.5 oracle (threshold sensitivity)
    best_single_thresh: float              # hard p>=0.5 best single
    routing_headroom_thresh: float         # threshold-version headroom


def _validate_solves(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 3:
        raise ValueError(f"solves tensor must be 3-D (Q, M, K), got shape {S.shape}")
    if S.size and not np.all(np.isin(S, (0.0, 1.0))):
        raise ValueError("solves tensor must contain only 0/1 entries")
    return S


def p_hat(S: np.ndarray) -> np.ndarray:
    """Per-(query, model) solve probability, averaged over the K samples."""
    S = _validate_solves(S)
    return S.mean(axis=2)  # (Q, M)


def best_single(p: np.ndarray) -> tuple[float, int]:
    """best_single = max_m mean_q p[q, m]; returns (value, argmax model index)."""
    per_model = p.mean(axis=0)
    m = int(np.argmax(per_model))
    return float(per_model[m]), m


def routing_oracle_naive(p: np.ndarray) -> float:
    """mean_q max_m p[q, m] — the achievable router ceiling, but upward-biased.

    `max` of noisy per-model estimates is >= max of the true values (Jensen), so this
    over-states the ceiling when p is a noisy estimate. Use `routing_oracle_crossfit`
    for the debiased number; this is reported alongside it to show the bias gap.
    """
    if p.size == 0:
        return 0.0
    return float(p.max(axis=1).mean())


def clairvoyant_any(p: np.ndarray) -> float:
    """mean_q (1 - prod_m (1 - p[q, m])): probability SOME model solves q.

    An optimistic UPPER bound a single-pick router can NEVER reach (it counts
    independent lucky draws across models). Reported separately and labelled
    "not routing-achievable" — it measures noise, not routable opportunity.
    """
    if p.size == 0:
        return 0.0
    return float((1.0 - np.prod(1.0 - p, axis=1)).mean())


def routing_oracle_crossfit(
    S: np.ndarray, *, n_splits: int = 200, seed: int = 0
) -> float:
    """Winner's-curse-debiased routing oracle via split-half select / evaluate.

    Thin wrapper returning only the oracle; see :func:`crossfit_oracle_and_best`
    for the paired (oracle, best_single) needed for an unbiased headroom.
    """
    return crossfit_oracle_and_best(S, n_splits=n_splits, seed=seed)[0]


def crossfit_oracle_and_best(
    S: np.ndarray, *, n_splits: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Cross-fit (routing_oracle, best_single), both evaluated on the SAME held-out half.

    The winner's-curse fix (plan §2.1) is to SELECT on one set of samples and EVALUATE
    on a disjoint set, so the selected-model accuracy is unbiased. For the *headroom*
    (oracle - best_single) to also be unbiased, both terms must be evaluated under the
    identical cross-fit, otherwise their estimation biases differ and the difference is
    contaminated (e.g. an oracle built from half-K samples compared against a best_single
    built from full-K samples yields a spurious negative headroom on identical models).

    For each split:
      * Per query, split its K samples into selection-half A (n_a = K//2) and
        evaluation-half B (the rest).
      * Router oracle: per query pick argmax model on A, score that model on B; average.
      * best_single: pick the single best FIXED model by mean accuracy on A (aggregated
        over all queries), score THAT one model on B; average. Selection and evaluation
        are disjoint, so this is the unbiased held-out accuracy of the best fixed model.

    Averaged over `n_splits` random A/B partitions. Requires K>=2; for K==1 there is no
    split, so we fall back to the naive (biased) estimates and the caller should note it.

    Argmax ties are broken by a tiny deterministic per-model jitter (model 0 favoured),
    too small to ever override a real probability difference (>= 1/n_a).
    """
    S = _validate_solves(S)
    Q, M, K = S.shape
    if Q == 0 or M == 0:
        return 0.0, 0.0
    if K < 2:
        p = p_hat(S)
        return routing_oracle_naive(p), best_single(p)[0]

    rng = np.random.default_rng(seed)
    n_a = K // 2  # selection-half size
    jitter = np.linspace(0.0, 1e-9, M)

    oracle_tot = np.zeros(n_splits)
    best_tot = np.zeros(n_splits)
    for s in range(n_splits):
        # The A/B sample split is PER QUERY, SHARED across models. If the split were
        # shuffled independently per model, two identical models would get different
        # A-half estimates by luck and the argmax would re-introduce the winner's curse
        # (picking whichever identical copy happened to look better on A). A shared split
        # keeps identical models tied, so identical pools correctly yield zero headroom.
        perm = rng.permuted(
            np.broadcast_to(np.arange(K), (Q, K)), axis=1
        )  # per-query shuffle of sample indices, applied to every model
        idx_a = np.broadcast_to(perm[:, None, :n_a], (Q, M, n_a))
        idx_b = np.broadcast_to(perm[:, None, n_a:], (Q, M, K - n_a))
        sel = np.take_along_axis(S, idx_a, axis=2).mean(axis=2)   # (Q, M) on half A
        evl = np.take_along_axis(S, idx_b, axis=2).mean(axis=2)   # (Q, M) on half B
        # Per-query router oracle.
        chosen = np.argmax(sel + jitter, axis=1)                  # (Q,)
        oracle_tot[s] = evl[np.arange(Q), chosen].mean()
        # Best FIXED model selected on A, evaluated on B (held-out).
        best_m = int(np.argmax(sel.mean(axis=0) + jitter))
        best_tot[s] = evl[:, best_m].mean()
    return float(oracle_tot.mean()), float(best_tot.mean())


def disagreement_rate(S: np.ndarray) -> float:
    """Fraction of queries where the models do NOT all agree (raw complementarity).

    Uses the per-(q,m) MAJORITY vote across samples as each model's verdict, then
    counts a query as a disagreement when those per-model verdicts are not all equal.
    """
    S = _validate_solves(S)
    Q, M, K = S.shape
    if Q == 0 or M <= 1:
        return 0.0
    votes = (S.mean(axis=2) >= 0.5).astype(int)  # (Q, M)
    disagree = (votes.min(axis=1) != votes.max(axis=1)).astype(float)
    return float(disagree.mean())


def _threshold_oracle(p: np.ndarray, thr: float = 0.5) -> tuple[float, float, float]:
    """Hard-threshold sensitivity report: treat a model as 'solves q' iff p>=thr.

    Returns (routing_oracle_thresh, best_single_thresh, headroom_thresh). The verdict
    is trusted only if it is stable across the probabilistic and this hard definition.
    """
    if p.size == 0:
        return 0.0, 0.0, 0.0
    solved = (p >= thr).astype(float)            # (Q, M)
    oracle = float(solved.max(axis=1).mean())    # router can solve q if ANY model can
    bs = float(solved.mean(axis=0).max())        # best fixed model under the threshold
    return oracle, bs, oracle - bs


def compute_stats(S: np.ndarray, *, crossfit_splits: int = 200, seed: int = 0) -> CeilingStats:
    """Compute all point-estimate ceiling quantities from the solves tensor."""
    S = _validate_solves(S)
    Q, M, K = S.shape
    p = p_hat(S)
    per_model = list(p.mean(axis=0)) if p.size else []
    # Full-K best_single for per-model REPORTING (what each fixed model scores on all K).
    bs, bs_m = best_single(p) if p.size else (0.0, 0)
    # Cross-fit oracle AND cross-fit best_single, both on the held-out half: their
    # difference is the unbiased headroom (see crossfit_oracle_and_best). Headroom uses
    # the cross-fit best_single so the two terms share the same estimation regime.
    oracle, bs_cf = crossfit_oracle_and_best(S, n_splits=crossfit_splits, seed=seed)
    oracle_naive = routing_oracle_naive(p)
    clair = clairvoyant_any(p)
    o_thr, bs_thr, h_thr = _threshold_oracle(p)
    return CeilingStats(
        n_queries=Q,
        n_models=M,
        k=K,
        per_model=[float(x) for x in per_model],
        best_single=bs,
        best_single_model=bs_m,
        routing_oracle=oracle,
        routing_oracle_naive=oracle_naive,
        clairvoyant_any=clair,
        routing_headroom=oracle - bs_cf,
        unroutable_noise=clair - oracle,
        disagreement_rate=disagreement_rate(S),
        routing_oracle_thresh=o_thr,
        best_single_thresh=bs_thr,
        routing_headroom_thresh=h_thr,
    )


# ---- bootstrap CIs (resample QUERIES with replacement) ----------------------
def bootstrap_ci(
    S: np.ndarray,
    stat_fn,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic of the solves tensor.

    Resamples QUERIES (axis 0) with replacement — the queries are the unit of
    statistical variation; the K samples per (q, m) are denoising, not independent
    data points. Returns (point_estimate, lo, hi) at the (1-alpha) level. The verdict
    is read off the CI, never the point estimate (plan §4).
    """
    S = _validate_solves(S)
    Q = S.shape[0]
    point = stat_fn(S)
    if Q == 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, Q, size=Q)
        vals[b] = stat_fn(S[idx])
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return float(point), lo, hi


def bootstrap_all(S: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
                  crossfit_splits: int = 100) -> dict:
    """Bootstrap CIs for the verdict-bearing quantities (plan §4).

    Each statistic is recomputed on every query-resample so the CI reflects
    query-level uncertainty (including how the cross-fit oracle moves with the
    query set).
    """
    # One cross-fit call per resample yields a CONSISTENT (oracle, best_single) pair, so
    # the headroom CI matches the headroom point estimate (both use cross-fit best_single).
    def stat_vec(s):
        oracle, bs_cf = crossfit_oracle_and_best(s, n_splits=crossfit_splits, seed=seed)
        clair = clairvoyant_any(p_hat(s))
        bs_full = best_single(p_hat(s))[0]
        return {
            "best_single": bs_full,
            "routing_oracle": oracle,
            "clairvoyant_any": clair,
            "routing_headroom": oracle - bs_cf,
            "unroutable_noise": clair - oracle,
        }

    S = _validate_solves(S)
    Q = S.shape[0]
    point = stat_vec(S)
    names = list(point.keys())
    if Q == 0:
        return {n: {"point": point[n], "ci_lo": point[n], "ci_hi": point[n]} for n in names}
    rng = np.random.default_rng(seed)
    draws = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        idx = rng.integers(0, Q, size=Q)
        v = stat_vec(S[idx])
        for n in names:
            draws[n][b] = v[n]
    out = {}
    for n in names:
        out[n] = {
            "point": float(point[n]),
            "ci_lo": float(np.quantile(draws[n], 0.025)),
            "ci_hi": float(np.quantile(draws[n], 0.975)),
        }
    return out


# ---- paired McNemar (manual; no scipy) --------------------------------------
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test on two per-query 0/1 correctness vectors.

    Both systems are scored on the SAME queries (plan §4), so the comparison is
    paired. b = #(A correct, B wrong), c = #(A wrong, B correct). We report the
    exact two-sided binomial p-value on the discordant pairs (n=b+c, k=min(b,c)),
    which is valid for small/large n and needs no scipy. Continuity-corrected chi^2
    is also returned for reference.
    """
    a = np.asarray(correct_a, dtype=int).ravel()
    b_vec = np.asarray(correct_b, dtype=int).ravel()
    if a.shape != b_vec.shape:
        raise ValueError("McNemar inputs must have the same length")
    b = int(np.sum((a == 1) & (b_vec == 0)))
    c = int(np.sum((a == 0) & (b_vec == 1)))
    n = b + c
    # Exact two-sided binomial p on discordant pairs (point null p=0.5).
    if n == 0:
        p_exact = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p_exact = min(1.0, 2.0 * tail)
    # Continuity-corrected chi-square (reference only).
    chi2 = ((abs(b - c) - 1) ** 2) / n if n > 0 else 0.0
    return {"b_only_a": b, "c_only_b": c, "n_discordant": n,
            "p_exact": float(p_exact), "chi2_cc": float(chi2)}


def router_gap_closed(trinity_acc: float, best_single_acc: float,
                      routing_oracle_acc: float) -> float:
    """(trinity - best_single) / (routing_oracle - best_single).

    Fraction of the REAL (achievable) headroom the trained router captures. Returns
    NaN when the denominator is ~0 (no achievable headroom -> the ratio is undefined,
    not zero), so callers must guard on the headroom CI before trusting it.
    """
    denom = routing_oracle_acc - best_single_acc
    if abs(denom) < 1e-9:
        return float("nan")
    return float((trinity_acc - best_single_acc) / denom)


# =============================================================================
# Matrix <-> tensor I/O (the on-disk format from plan §7.1)
# =============================================================================
def matrix_to_tensor(matrix: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert an oracle_matrix_<bench>.json dict into (S, query_ids, model_names).

    matrix["tasks"][i]["per_model"][model] is a list of K 0/1 sample outcomes. All
    (query, model) cells must share the same K; ragged input is an error (a collection
    bug we must not silently average over).
    """
    tasks = matrix["tasks"]
    if not tasks:
        return np.zeros((0, 0, 0)), [], []
    model_names = list(tasks[0]["per_model"].keys())
    k_set = {len(tasks[0]["per_model"][m]) for m in model_names}
    if len(k_set) != 1:
        raise ValueError(f"ragged K within a task: {k_set}")
    K = k_set.pop()
    Q, M = len(tasks), len(model_names)
    S = np.zeros((Q, M, K))
    qids: list[str] = []
    for qi, t in enumerate(tasks):
        qids.append(str(t.get("id", t.get("task_id", qi))))
        for mi, m in enumerate(model_names):
            cell = t["per_model"][m]
            if len(cell) != K:
                raise ValueError(f"task {qids[-1]} model {m}: expected K={K}, got {len(cell)}")
            S[qi, mi, :] = cell
    return S, qids, model_names


def analyze_matrix(
    matrix: dict,
    *,
    trinity_per_query: dict | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    crossfit_splits: int = 200,
) -> dict:
    """Full FP/FN-proof analysis of a matrix dict -> machine-readable report.

    trinity_per_query (optional): {query_id: 0/1} of the trained router's correctness
    on the SAME queries, used for router_gap_closed and a paired McNemar vs best-single.
    """
    S, qids, models = matrix_to_tensor(matrix)
    stats = compute_stats(S, crossfit_splits=crossfit_splits, seed=seed)
    cis = bootstrap_all(S, n_boot=n_boot, seed=seed,
                        crossfit_splits=max(40, crossfit_splits // 2))

    report: dict = {
        "benchmark": matrix.get("benchmark"),
        "level": matrix.get("level"),
        "k": stats.k,
        "n_queries": stats.n_queries,
        "models": models,
        "per_model_accuracy": dict(zip(models, stats.per_model)),
        "point_estimates": {
            "best_single": stats.best_single,
            "best_single_model": models[stats.best_single_model] if models else None,
            "routing_oracle": stats.routing_oracle,
            "routing_oracle_naive_biased": stats.routing_oracle_naive,
            "winners_curse_bias": stats.routing_oracle_naive - stats.routing_oracle,
            "clairvoyant_any_NOT_ACHIEVABLE": stats.clairvoyant_any,
            "routing_headroom": stats.routing_headroom,
            "unroutable_noise": stats.unroutable_noise,
            "disagreement_rate": stats.disagreement_rate,
        },
        "threshold_sensitivity_p>=0.5": {
            "routing_oracle": stats.routing_oracle_thresh,
            "best_single": stats.best_single_thresh,
            "routing_headroom": stats.routing_headroom_thresh,
        },
        "bootstrap_ci_95": cis,
        "notes": {
            "clairvoyant_any": "optimistic upper bound, NOT achievable by a single-pick router; measures noise not opportunity",
            "routing_oracle": "winner's-curse-debiased via split-half cross-fit; this is the honest routing ceiling",
            "verdict_source": "read the verdict off CIs, never the point estimates",
        },
    }

    # router_gap_closed + paired McNemar vs best single, if TRINITY data is supplied.
    if trinity_per_query is not None and stats.n_queries:
        p = p_hat(S)
        best_m = stats.best_single_model
        # Per-query best-single correctness via the per-(q,m) majority vote.
        best_correct = (p[:, best_m] >= 0.5).astype(int)
        tri = np.array([int(trinity_per_query.get(q, 0)) for q in qids])
        trinity_acc = float(tri.mean())
        gap = router_gap_closed(trinity_acc, stats.best_single, stats.routing_oracle)
        report["trinity"] = {
            "accuracy": trinity_acc,
            "router_gap_closed": gap,
            "mcnemar_vs_best_single": mcnemar(tri, best_correct),
        }

    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> dict:
    """CI-gated decision rule (plan §6).

    pool-bound:   headroom CI upper bound small (<=0.02) AND CI includes 0.
    router-bound: headroom CI lower bound > 0 AND (if known) gap_closed < 0.5.
    near-ceiling: headroom real AND gap_closed high.
    """
    h = report["bootstrap_ci_95"]["routing_headroom"]
    lo, hi = h["ci_lo"], h["ci_hi"]
    includes_zero = lo <= 0.0 <= hi
    tri = report.get("trinity", {})
    gap = tri.get("router_gap_closed")

    if hi <= 0.02 and includes_zero:
        label = "POOL_BOUND"
        msg = ("Routing cannot help on this pool: headroom CI upper bound <= 0.02 and "
               "includes 0. The lever is the model pool, not the router.")
    elif lo > 0.0:
        if gap is not None and not math.isnan(gap) and gap < 0.5:
            label = "ROUTER_BOUND"
            msg = ("Real headroom exists (CI lower bound > 0) and the router leaves "
                   ">50% of it on the table. Pursue router improvements.")
        elif gap is not None and not math.isnan(gap) and gap >= 0.5:
            label = "NEAR_CEILING"
            msg = ("Real headroom exists and the router already captures most of it. "
                   "Gains require a better pool, not more router tuning.")
        else:
            label = "ROUTER_BOUND"
            msg = ("Real headroom exists (CI lower bound > 0); router gap_closed unknown "
                   "(no TRINITY data supplied). Routing improvements are warranted.")
    else:
        label = "INCONCLUSIVE"
        msg = ("Headroom CI straddles 0 but upper bound > 0.02: cannot rule routing in "
               "or out. Widen the reachability level (L1/L2) or collect more samples.")
    return {"label": label, "headroom_ci_95": [lo, hi], "router_gap_closed": gap,
            "message": msg}


# =============================================================================
# Live collection (--collect). Reuses FireworksPool + reward.score_text.
# =============================================================================
@dataclass
class _CostMeter:
    """Running spend tracker fed off the per-call token counts (matches the ledger)."""

    price_in: dict[str, float] = field(default_factory=dict)   # $/1M prompt toks by short name
    price_out: dict[str, float] = field(default_factory=dict)  # $/1M completion toks
    spend: float = 0.0
    calls: int = 0

    def add(self, model: str, pt: int, ct: int) -> None:
        short = model.rsplit("/", 1)[-1]
        pin = self.price_in.get(short, 0.0)
        pout = self.price_out.get(short, 0.0)
        self.spend += pt / 1e6 * pin + ct / 1e6 * pout
        self.calls += 1


# Fireworks prices from the most recent commit (b51d775): $/1M (in, out).
_DEFAULT_PRICES = {
    "deepseek-v4-pro": (1.74, 3.48),
    "glm-5p2": (1.40, 4.40),
    "kimi-k2p6": (0.95, 4.00),
}


async def _collect(args) -> int:
    """Draw K samples per (query, model), grade, and persist matrix + raw JSONL.

    L0 (single-turn Worker) is implemented here (the cheap strict lower bound on the
    ceiling). The deployment decoding regime is matched: temperature 0, reasoning
    'minimal' — same as the Worker path in eval.py and how the router is evaluated.
    """
    import httpx

    from trinity.llm.fireworks_client import FireworksPool
    from trinity.orchestration import reward as R
    from trinity.orchestration.dataset import load_tasks
    from trinity.roles.prompts import build_messages
    from trinity.types import Role

    out_dir = _REPO / "experiments" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"oracle_raw_{args.benchmark}.jsonl"
    matrix_path = out_dir / f"oracle_matrix_{args.benchmark}.json"

    pool = FireworksPool(args.models)
    models = list(pool.models)
    tasks = load_tasks(args.benchmark, "test", max_items=args.max_items, seed=args.seed)
    print(f"[collect] benchmark={args.benchmark} level={args.level} K={args.k} "
          f"tasks={len(tasks)} models={models} -> {len(tasks)*len(models)*args.k} calls")

    meter = _CostMeter(
        price_in={m: _DEFAULT_PRICES.get(m, (0.0, 0.0))[0] for m in models},
        price_out={m: _DEFAULT_PRICES.get(m, (0.0, 0.0))[1] for m in models},
    )
    aborted = {"flag": False}

    # Per-(query,model) solve cells, filled as calls return.
    cells: dict[tuple[str, str], list[int]] = {
        (t.task_id, m): [] for t in tasks for m in models
    }
    raw_fh = open(raw_path, "w")

    async def one(task, model, rep, cli) -> None:
        if aborted["flag"]:
            return
        msgs = build_messages(Role.WORKER, task.prompt, [])
        t0 = time.time()
        err = None
        text = ""
        pt = ct = 0
        finish = None
        try:
            res = await pool.chat(model, msgs, max_tokens=args.max_tokens,
                                  temperature=0.0, reasoning=args.reasoning, client=cli)
            text, pt, ct, finish = res.text, res.prompt_tokens, res.completion_tokens, res.finish_reason
        except Exception as exc:  # network/HTTP after retries — record, do not crash the run
            err = f"{type(exc).__name__}: {exc}"
        latency = time.time() - t0

        correct = 0
        extracted = None
        if err is None:
            try:
                correct = int(R.score_text(args.benchmark, text, task.answer))
            except Exception as exc:
                err = f"grade: {type(exc).__name__}: {exc}"
            extracted = _extracted_for(args.benchmark, text)

        cells[(task.task_id, model)].append(correct)
        meter.add(model, pt, ct)
        rec = {
            "query_id": task.task_id, "benchmark": args.benchmark, "model": model,
            "role": "worker", "rep": rep, "prompt_messages": msgs,
            "raw_response_text": text, "extracted_answer": extracted,
            "gold_answer": task.answer, "correct": correct,
            "prompt_tokens": pt, "completion_tokens": ct,
            "finish_reason": finish, "latency_s": round(latency, 3), "error": err,
        }
        raw_fh.write(json.dumps(rec, default=str) + "\n")
        raw_fh.flush()

        if meter.calls % 25 == 0 or aborted["flag"]:
            print(f"[collect] calls={meter.calls} spend=${meter.spend:.3f} "
                  f"(cap=${args.max_cost_usd})", flush=True)
        if args.max_cost_usd > 0 and meter.spend > args.max_cost_usd and not aborted["flag"]:
            aborted["flag"] = True
            print(f"[collect] ABORT: spend ${meter.spend:.3f} exceeded cap "
                  f"${args.max_cost_usd}", flush=True)

    async with httpx.AsyncClient() as cli:
        jobs = [one(t, m, k, cli)
                for t in tasks for m in models for k in range(args.k)]
        await asyncio.gather(*jobs)
    raw_fh.close()

    # Assemble the compact matrix (plan §7.1 schema).
    matrix = {
        "benchmark": args.benchmark, "k": args.k, "level": args.level,
        "seed": args.seed, "aborted": aborted["flag"],
        "tasks": [
            {"id": t.task_id, "answer": t.answer,
             "per_model": {m: cells[(t.task_id, m)] for m in models}}
            for t in tasks
        ],
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, default=str))
    print(f"[collect] wrote {raw_path} and {matrix_path}")
    print(f"[collect] final: calls={meter.calls} spend=${meter.spend:.3f}")
    if aborted["flag"]:
        print("[collect] NOTE: run was cost-aborted; matrix is partial (ragged K).")
        return 2
    return 0


def _extracted_for(benchmark: str, text: str):
    """Best-effort 'what the grader extracted' for the raw log (debuggability)."""
    from trinity.orchestration import reward as R
    key = (benchmark or "").lower()
    if key in R.CHOICE_BENCHMARKS:
        return R.extract_choice_letter(text)
    if key in R.MATH_BENCHMARKS:
        return R.extract_boxed(text) or R.extract_last_number(text)
    if key in R.CODE_BENCHMARKS:
        return R.extract_code(text)[:200]
    return None


def _run_analyze(args) -> int:
    matrix = json.loads(Path(args.analyze).read_text())
    trinity_pq = None
    if args.trinity_per_query:
        trinity_pq = json.loads(Path(args.trinity_per_query).read_text())
    report = analyze_matrix(matrix, trinity_per_query=trinity_pq,
                            n_boot=args.n_boot, seed=args.seed,
                            crossfit_splits=args.crossfit_splits)
    bench = matrix.get("benchmark", "unknown")
    out = _REPO / "experiments" / "final" / f"oracle_report_{bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[analyze] wrote {out}")
    print(f"[analyze] VERDICT: {report['verdict']['label']} — {report['verdict']['message']}")
    return 0


# =============================================================================
# Self-test (synthetic, offline). Same checks as tests/test_oracle_ceiling.py.
# =============================================================================
def _selftest() -> int:
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # (a) 3 disjoint specialists -> routing_oracle = 1.0, headroom ~ 0.667.
    Q, K = 30, 5
    S = np.zeros((Q, 3, K))
    for q in range(Q):
        S[q, q % 3, :] = 1.0  # exactly one model solves each query, deterministically
    st = compute_stats(S, crossfit_splits=100, seed=0)
    check("(a) disjoint specialists: routing_oracle == 1.0",
          abs(st.routing_oracle - 1.0) < 1e-9)
    check("(a) disjoint specialists: best_single ~ 1/3",
          abs(st.best_single - 1.0 / 3.0) < 0.05)
    check("(a) disjoint specialists: headroom ~ 0.667",
          abs(st.routing_headroom - 2.0 / 3.0) < 0.05)

    # (b) 3 identical models -> headroom = 0.
    rng = np.random.default_rng(1)
    base = (rng.random((Q, 1, K)) < 0.6).astype(float)
    S_id = np.repeat(base, 3, axis=1)  # all three models give identical outcomes
    st_id = compute_stats(S_id, crossfit_splits=100, seed=0)
    check("(b) identical models: headroom == 0 (cross-fit)",
          abs(st_id.routing_headroom) < 1e-9)
    ci_id = bootstrap_all(S_id, n_boot=500, seed=0, crossfit_splits=40)
    check("(b) identical models: headroom CI includes 0",
          ci_id["routing_headroom"]["ci_lo"] <= 0.0 <= ci_id["routing_headroom"]["ci_hi"])

    # (c) pure noise (every cell p=0.5) -> headroom CI includes 0 (no FP on noise).
    rng = np.random.default_rng(2)
    S_noise = (rng.random((120, 3, 5)) < 0.5).astype(float)
    ci_noise = bootstrap_all(S_noise, n_boot=2000, seed=0, crossfit_splits=60)
    h = ci_noise["routing_headroom"]
    check("(c) pure noise: headroom CI includes 0 (no false positive)",
          h["ci_lo"] <= 0.0 <= h["ci_hi"])
    rep_noise = analyze_matrix(_tensor_to_matrix(S_noise, "noise"),
                               n_boot=2000, seed=0, crossfit_splits=60)
    check("(c) pure noise: verdict is not a false ROUTER_BOUND",
          rep_noise["verdict"]["label"] != "ROUTER_BOUND")

    # (d) cross-fit debiasing reduces the max-selection bias vs naive max_m p_hat.
    # 3 statistically identical p=0.5 models: TRUE oracle == 0.5; naive max is biased up.
    rng = np.random.default_rng(3)
    S_d = (rng.random((200, 3, 6)) < 0.5).astype(float)
    naive = routing_oracle_naive(p_hat(S_d))
    cf = routing_oracle_crossfit(S_d, n_splits=300, seed=0)
    check("(d) naive max is upward-biased above true 0.5", naive > 0.5 + 0.02)
    check("(d) cross-fit is closer to true 0.5 than naive",
          abs(cf - 0.5) < abs(naive - 0.5))
    check("(d) cross-fit reduces bias by a clear margin",
          (naive - 0.5) - abs(cf - 0.5) > 0.02)

    # McNemar sanity: identical vectors -> p=1.0; fully discordant -> small p.
    mc_same = mcnemar(np.array([1, 0, 1, 1]), np.array([1, 0, 1, 1]))
    check("McNemar identical -> p_exact == 1.0", abs(mc_same["p_exact"] - 1.0) < 1e-9)
    mc_diff = mcnemar(np.ones(20, int), np.zeros(20, int))
    check("McNemar fully discordant -> p_exact small", mc_diff["p_exact"] < 0.01)

    print(f"\n[selftest] {'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 0 if not failures else 1


def _tensor_to_matrix(S: np.ndarray, bench: str) -> dict:
    """Inverse of matrix_to_tensor for synthetic tests."""
    Q, M, K = S.shape
    models = [f"m{j}" for j in range(M)]
    return {
        "benchmark": bench, "k": K, "level": "L0", "seed": 0,
        "tasks": [
            {"id": f"q{q}", "answer": "x",
             "per_model": {models[j]: [int(v) for v in S[q, j, :]] for j in range(M)}}
            for q in range(Q)
        ],
    }


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-ceiling diagnostic")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true", help="live matrix collection")
    mode.add_argument("--analyze", metavar="MATRIX_JSON", help="analyze a matrix JSON")
    mode.add_argument("--selftest", action="store_true", help="offline synthetic tests")

    # collect args
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--k", type=int, default=5, help="samples per (query, model)")
    ap.add_argument("--level", default="L0", choices=["L0"],
                    help="reachability level (L0 single-turn Worker; L1/L2 are future)")
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--max-cost-usd", type=float, default=15.0, dest="max_cost_usd",
                    help="abort collection if running spend exceeds this (0 = no cap)")
    # analyze args
    ap.add_argument("--trinity-per-query", default="", dest="trinity_per_query",
                    help="JSON {query_id: 0/1} of TRINITY correctness for gap_closed")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--crossfit-splits", type=int, default=200, dest="crossfit_splits")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if args.analyze:
        sys.exit(_run_analyze(args))
    if args.collect:
        sys.exit(asyncio.run(_collect(args)))


if __name__ == "__main__":
    main()
