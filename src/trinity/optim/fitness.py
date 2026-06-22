"""Fitness evaluation for sep-CMA-ES candidates.

Fitness(θ) = mean binary reward R(τ) over a minibatch of `m_cma` task instances,
each run through the coordination loop with the policy configured by θ.

Concurrency model (see docs/SPEC.md §0.3.7, §5.2):
  - Candidates are evaluated SEQUENTIALLY because SVF mutates the single shared SLM's
    weights in place; two candidates cannot be live on the GPU at once.
  - Within one candidate, the `m_cma` trajectories share θ, so their (fast, serialized)
    SLM forwards interleave while the (slow) Fireworks calls run concurrently.
"""
from __future__ import annotations

import asyncio
from statistics import mean

from ..orchestration import reward as _reward
from ..orchestration.session import run_trajectory


async def evaluate_candidate(
    theta,
    spec,
    policy,
    pool,
    pool_models: list[str],
    minibatch: list,
    *,
    sample: bool = True,
    client=None,
    return_trajectories: bool = False,
    **run_kwargs,
) -> tuple[float, list]:
    """Configure the policy with θ and return (mean_reward, trajectories?)."""
    policy.configure(theta, spec)

    own_client = False
    if client is None:
        try:
            import httpx

            client = httpx.AsyncClient()
            own_client = True
        except Exception:
            client = None
    try:
        trajs = await asyncio.gather(
            *[
                run_trajectory(task, policy, pool, pool_models, sample=sample, client=client, **run_kwargs)
                for task in minibatch
            ]
        )
    finally:
        if own_client and client is not None:
            await client.aclose()

    rewards = []
    for t in trajs:
        r = float(_reward.score(t))
        t.reward = r
        rewards.append(r)
    fit = float(mean(rewards)) if rewards else 0.0
    return (fit, list(trajs)) if return_trajectories else (fit, [])


async def evaluate_population(
    thetas: list,
    spec,
    policy,
    pool,
    pool_models: list[str],
    minibatch_fn,
    *,
    sample: bool = True,
    on_candidate=None,
    **run_kwargs,
) -> list[float]:
    """Evaluate λ candidates sequentially (GPU constraint). `minibatch_fn(i)->tasks`
    yields the per-candidate minibatch (re-sampled each iteration for an unbiased J).

    `on_candidate(i, fit, elapsed_s)` is called after each candidate for progress logging.
    """
    import time

    fits: list[float] = []
    client = None
    try:
        import httpx

        client = httpx.AsyncClient()
    except Exception:
        client = None
    try:
        for i, theta in enumerate(thetas):
            t0 = time.time()
            mb = minibatch_fn(i)
            fit, _ = await evaluate_candidate(
                theta, spec, policy, pool, pool_models, mb, sample=sample, client=client, **run_kwargs
            )
            fits.append(fit)
            if on_candidate is not None:
                on_candidate(i, fit, time.time() - t0)
    finally:
        if client is not None:
            await client.aclose()
    return fits
