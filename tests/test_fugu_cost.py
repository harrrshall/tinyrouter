"""Offline tests for API-cost accounting (exact pricing + projections)."""
from __future__ import annotations

from trinity.fugu.cost import (
    CostMeter,
    estimate_eval_cost,
    estimate_grpo_cost,
    price_table,
    run_cost,
)
from trinity.fugu.workflow import CONDUCTOR_KEY, WorkflowRun

POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]


def test_run_cost_prices_exactly():
    # glm-5p2 is $1.40/1M in, $4.40/1M out. 1M each -> $5.80.
    total, bd = run_cost({"glm-5p2": (1_000_000, 1_000_000)})
    assert abs(total - 5.80) < 1e-9
    assert abs(bd["glm-5p2"] - 5.80) < 1e-9


def test_conductor_is_free_when_local():
    table = price_table(conductor_local=True)
    total, _ = run_cost({CONDUCTOR_KEY: (1_000_000, 1_000_000)}, table)
    assert total == 0.0
    # but priced when the conductor is a Fireworks model
    table2 = price_table("kimi-k2p6", conductor_local=False)
    total2, _ = run_cost({CONDUCTOR_KEY: (1_000_000, 0)}, table2)
    assert abs(total2 - 0.95) < 1e-9


def test_cost_meter_accumulates_and_caps():
    meter = CostMeter(cap_usd=0.002)
    run = WorkflowRun(
        workflow=None, parsed_ok=True, final_answer="x",
        n_llm_calls=2, prompt_tokens=1000, completion_tokens=1000,
        model_tokens={"glm-5p2": (1000, 1000)},
    )
    c = meter.add_run(run)
    assert c > 0 and meter.runs == 1 and meter.calls == 2
    # 1000 in + 1000 out on glm: ~0.0058 USD, over the 0.002 cap.
    assert meter.aborted is True
    rep = meter.report()
    assert rep["per_model"]["glm-5p2"]["usd"] > 0


def test_estimate_grpo_cost_projects_spend():
    est = estimate_grpo_cost(
        worker_names=POOL, group_size=64, iterations=200, questions_per_iter=4,
        avg_steps=2.5,
    )
    assert est["rollouts"] == 64 * 4 * 200
    assert est["worker_calls"] == int(round(est["rollouts"] * 2.5))
    assert est["total_usd"] > 0
    # default conductor is local -> no API charge for the conductor itself.
    assert est["conductor_api_usd"] == 0.0


def test_estimate_eval_cost_is_cheaper_than_training():
    ev = estimate_eval_cost(worker_names=POOL, n_tasks=120, reps=3, avg_steps=2.5)
    assert ev["total_usd"] > 0
    assert ev["rollouts"] == 120 * 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
