"""Verification tests for the simulation model (Section 4.5 of the paper).

Run from the repository root:  python -m pytest -q   (or: python tests/test_model.py)
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from governance_sim import (Env, Draws, POLICIES, run_policy, run_benchmark,  # noqa: E402
                            own_model_thresholds, aligned_gates)

SEED, N = 20261122, 12000
D = Draws(SEED, N)
BASE = Env()


def test_parity_with_original_implementation():
    """F and Q reproduce the original (v1) script exactly on identical draws."""
    import orig
    rng = np.random.default_rng(SEED)
    rows, _ = orig.sim_one(orig.SCENARIOS[0], rng)
    v1 = {r["policy"]: r["mean_net_value"] for r in rows}
    assert np.isclose(run_policy("F", BASE, D, 0.18)["net"].mean(), v1["fixed_6_9"], atol=1e-12)
    assert np.isclose(run_policy("Q", BASE, D, 0.18)["net"].mean(), v1["quarterly_3_6_9"], atol=1e-12)


def test_scale_invariance():
    """Doubling every monetary parameter doubles net value and leaves decisions unchanged."""
    e2 = replace(BASE, value=2 * BASE.value, cost=2 * BASE.cost, monitor_fee=2 * BASE.monitor_fee,
                 review_fee=2 * BASE.review_fee, ver_fee=2 * BASE.ver_fee)
    for pol in POLICIES:
        a = run_policy(pol, BASE, D, 0.18)
        b = run_policy(pol, e2, D, 0.18)
        assert np.allclose(b["net"], 2 * a["net"])
        assert np.array_equal(a["completed"], b["completed"])
    assert np.allclose(own_model_thresholds(BASE), own_model_thresholds(e2))


def test_benchmark_dominates_policies_with_same_information():
    """B (calibrated optimal stopping, costless review) bounds F, Q, GA and U."""
    for env in (BASE, replace(BASE, q_true=.62, q_assumed=.80), replace(BASE, block=4)):
        b = run_benchmark(env, D, False)["net"].mean()
        own = own_model_thresholds(env)
        for pol in ("F", "Q", "GA", "U"):
            assert b >= run_policy(pol, env, D, own)["net"].mean()


def test_own_model_thresholds_decline_over_time():
    thr = own_model_thresholds(BASE)[1:12]
    assert np.all(np.diff(thr) <= 1e-12)


def test_aligned_gates():
    assert aligned_gates(1) == (3, 6, 9)
    assert aligned_gates(3) == (3, 6, 9)
    assert aligned_gates(4) == (4, 8)
    assert aligned_gates(6) == (6,)


def test_verification_value_disappears_with_full_echo():
    """With rho close to 1 verification adds (almost) no information, so QV cannot beat Q."""
    env = replace(BASE, q_true=.62, q_assumed=.80, rho=0.999)
    own = own_model_thresholds(env)
    assert run_policy("QV", env, D, own)["net"].mean() <= run_policy("Q", env, D, own)["net"].mean() + 1e-9


def test_reproducibility():
    a = run_policy("E", BASE, Draws(SEED, 2000), 0.2)["net"]
    b = run_policy("E", BASE, Draws(SEED, 2000), 0.2)["net"]
    assert np.array_equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("passed:", name)
