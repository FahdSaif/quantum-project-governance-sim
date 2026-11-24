"""Verification tests for the simulation model (Section 4.5 of the paper).

Run from the repository root:  python -m pytest -q   (or: python tests/test_model.py)
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from governance_sim import (Env, Draws, POLICIES, run_policy, run_benchmark,  # noqa: E402
                            own_model_thresholds, policy_thresholds, policy_tables, aligned_gates,
                            renewal_aligned_gates, decision_points, routine_signals, H,
                            calendar_aligned_gates, realised_lags, block_starts)

SEED, N = 20261122, 12000
D = Draws(SEED, N)
BASE = Env()


# ------------------------------------------------------------------ v3 tests (unchanged)
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
        assert np.allclose(policy_thresholds(pol, BASE), policy_thresholds(pol, e2))
    assert np.allclose(own_model_thresholds(BASE), own_model_thresholds(e2))


def test_benchmark_dominates_policies_with_same_information():
    """B (calibrated optimal stopping, costless review) bounds F, Q, GA, GAL and U."""
    for env in (BASE, replace(BASE, q_true=.62, q_assumed=.80), replace(BASE, block=4),
                replace(BASE, block=4, lag=1)):
        b = run_benchmark(env, D, False)["net"].mean()
        for pol in ("F", "Q", "GA", "GAL", "U"):
            for thr in (own_model_thresholds(env), policy_thresholds(pol, env)):
                assert b >= run_policy(pol, env, D, thr)["net"].mean()


def test_own_model_thresholds_decline_over_time():
    thr = own_model_thresholds(BASE)[1:12]
    assert np.all(np.diff(thr) <= 1e-12)


def test_aligned_gates():
    assert aligned_gates(1) == (3, 6, 9)
    assert aligned_gates(2) == (4, 8)
    assert aligned_gates(3) == (3, 6, 9)
    assert aligned_gates(4) == (4, 8)
    assert aligned_gates(6) == (6,)


def test_verification_value_disappears_with_full_echo():
    """With rho close to 1 verification adds (almost) no information, so QV cannot beat Q."""
    env = replace(BASE, q_true=.62, q_assumed=.80, rho=0.999)
    for thr_q, thr_qv in ((own_model_thresholds(env),) * 2,
                          (policy_thresholds("Q", env), policy_thresholds("QV", env))):
        assert run_policy("QV", env, D, thr_qv)["net"].mean() <= run_policy("Q", env, D, thr_q)["net"].mean() + 1e-9


def test_reproducibility():
    a = run_policy("E", BASE, Draws(SEED, 2000), 0.2)["net"]
    b = run_policy("E", BASE, Draws(SEED, 2000), 0.2)["net"]
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ v4 tests (new features)
def test_policy_thresholds_equal_v3_for_continuous_review():
    """With decisions possible every period and no lag, the policy-specific rule IS the v3 rule."""
    for env in (BASE, replace(BASE, block=4), replace(BASE, q_true=.62, q_assumed=.80)):
        v3 = own_model_thresholds(env)
        for pol in ("U", "E", "E_m"):
            new = policy_thresholds(pol, env)
            same = v3 > 1e-3                     # identical wherever v3 is not a pure tie-break
            assert np.array_equal(new[same], v3[same])
            assert np.all(new[~same] <= v3[~same])


def test_policy_thresholds_anticipate_renewals_without_gate():
    """Q with four-period blocks: the gate at period 3 must anticipate the renewal at period 5
    (positive threshold), whereas the v3 rule assumed a free decision at period 4."""
    env = replace(BASE, block=4)
    thr = policy_thresholds("Q", env)
    assert thr[3] > 0.1 and thr[6] > 0.1
    assert thr[9] < 1e-3                        # final block already paid: stopping saves nothing
    assert np.all(thr[[t for t in range(1, H) if t not in (3, 6, 9)]] == 0)   # no decisions there


def test_policy_thresholds_are_optimal_within_schedule():
    """Under calibration and costless review, policy-specific thresholds should do at least as well
    as any constant threshold for scheduled gates (up to Monte Carlo noise)."""
    env = replace(BASE, block=4, review_fee=0.0)
    for pol in ("Q", "GA"):
        best_const = max(run_policy(pol, env, D, th)["net"].mean() for th in np.arange(0.02, 0.6, 0.02))
        assert run_policy(pol, env, D, policy_thresholds(pol, env))["net"].mean() >= best_const - 0.03


def test_nonrenewal_leaves_net_value_unchanged_and_removes_stranding():
    """Deciding now and not renewing saves exactly the renewals an immediate stop saves."""
    for env in (replace(BASE, block=4), replace(BASE, block=4, lag=1), replace(BASE, block=3, lag=2),
                replace(BASE, block=6, lag=2, q_true=.75, q_assumed=.95, beta=0.4),
                replace(BASE, block=2, lag=1, q_true=.6, q_assumed=.9, rho=.5, monitor_fee=.05)):
        enr = replace(env, nonrenewal=True)
        for pol in POLICIES:
            a = run_policy(pol, env, D, policy_thresholds(pol, env))
            b = run_policy(pol, enr, D, policy_thresholds(pol, enr))
            assert np.allclose(a["net"], b["net"])
            assert np.allclose(b["stranded"], 0.0)


def test_lag_aware_aligned_gates():
    assert renewal_aligned_gates(4, 0) == aligned_gates(4)
    assert renewal_aligned_gates(4, 1) == (3, 7)
    assert renewal_aligned_gates(1, 2) == (1, 4, 7)
    assert renewal_aligned_gates(6, 2) == (4,)
    env = replace(BASE, block=4, lag=1)
    assert decision_points("GAL", env) == (3, 7)
    # each GAL stop takes effect exactly at a renewal point
    assert all((t + env.lag) % env.block == 0 for t in decision_points("GAL", env))


def test_gal_equals_ga_without_lag():
    env = replace(BASE, block=4)
    a = run_policy("GA", env, D, policy_thresholds("GA", env))["net"]
    b = run_policy("GAL", env, D, policy_thresholds("GAL", env))["net"]
    assert np.array_equal(a, b)


def test_serial_correlation_preserves_marginal_accuracy():
    env = replace(BASE, persist=0.3)
    s = routine_signals(env, D)
    correct = np.where(D.viable[:, None], s, ~s)
    assert abs(correct.mean() - env.q_true) < 0.01
    s0 = routine_signals(BASE, D)
    rep = np.mean(s[:, 1:] == s[:, :-1]); rep0 = np.mean(s0[:, 1:] == s0[:, :-1])
    assert abs(rep - (env.persist + (1 - env.persist) * rep0)) < 0.01   # repeats as specified
    assert np.array_equal(routine_signals(replace(BASE, persist=0.0), D), s0)


# ------------------------------------------------------------------ v5 tests (new features)
def test_v5_regular_calendar_matches_block_rule():
    for b in (1, 2, 3, 4, 6):
        env = replace(BASE, block=b)
        assert calendar_aligned_gates(env) == aligned_gates(b)
        assert calendar_aligned_gates(replace(BASE, starts=block_starts(env))) == aligned_gates(b)
        a = run_policy("GA", env, D, policy_thresholds("GA", env))["net"]
        e2 = replace(BASE, starts=block_starts(env))
        assert np.array_equal(a, run_policy("GA", e2, D, policy_thresholds("GA", e2))["net"])


def test_v5_realised_lag_distribution():
    env = replace(BASE, lag=1, lag_bias=1, lag_spread=1)
    l = realised_lags(env, D)
    assert set(np.unique(l)) == {1, 2, 3} and abs(l.mean() - 2) < 0.03
    assert np.all(realised_lags(replace(BASE, lag=0, lag_bias=-1), D) == 0)      # truncated at 0
    assert np.all(realised_lags(replace(BASE, lag=2), D) == 2)                    # default: exact


def test_v5_exact_lag_reproduces_v4():
    """With no lag uncertainty the realised lag equals the planned lag (v4 behaviour)."""
    env = replace(BASE, block=4, lag=1)
    a = run_policy("GAL", env, D, policy_thresholds("GAL", env))["net"]
    b = run_policy("GAL", replace(env, lag_bias=0, lag_spread=0), D, policy_thresholds("GAL", env))["net"]
    assert np.array_equal(a, b)


def test_v5_policies_never_use_realised_lag():
    """Gates and stop rules depend only on the planned lag."""
    e1 = replace(BASE, block=4, lag=1); e2 = replace(e1, lag_bias=1, lag_spread=1)
    for pol in ("GAL", "GALp", "Q"):
        assert decision_points(pol, e1) == decision_points(pol, e2)
        assert np.array_equal(policy_thresholds(pol, e1), policy_thresholds(pol, e2))
    assert decision_points("GALp", e1) == decision_points("GAL", replace(e1, lag=2))


def test_v5_full_refund_removes_value_of_alignment():
    """With full refund of unused prepaid cost, commitments are reversible and B gains."""
    env = replace(BASE, block=4)
    b0 = run_benchmark(env, D, False)["net"].mean()
    b1 = run_benchmark(replace(env, recover=1.0), D, False)["net"].mean()
    assert b1 > b0
    r = run_policy("Q", replace(env, recover=0.5), D, policy_thresholds("Q", replace(env, recover=0.5)))
    assert np.allclose(r["refund"], 0.5 * r["stranded"])


def test_v5_report_fee_charged_equally():
    env = replace(BASE, report_fee=0.1)
    for pol in ("Q", "U", "GA"):
        a = run_policy(pol, BASE, D, 0.18); b = run_policy(pol, env, D, 0.18)
        active_periods = np.where(a["completed"], H, a["stop_t"])
        assert np.allclose(a["net"] - b["net"], 0.1 * active_periods)


def test_v5_value_based_verification_buys_only_when_worth_fee():
    """With a prohibitive fee no verification is bought, so QVo behaves like Q."""
    env = replace(BASE, ver_fee=50.0)
    a = run_policy("QVo", env, D, policy_tables("QVo", env))
    assert a["n_ver"].sum() == 0
    assert np.array_equal(a["net"], run_policy("Q", env, D, policy_thresholds("Q", env))["net"])


# ------------------------------------------------------------------ v7 tests
def test_v7_known_full_echo_has_no_information_value():
    """If the forum knows verification always echoes routine evidence, it never buys it."""
    env = replace(BASE, rho=1.0, rho_known=True)
    tabs = policy_tables("QVo", env)
    assert max(tabs["evsi"][0].max(), tabs["evsi"][1].max()) < 1e-9
    assert run_policy("QVo", env, D, tabs)["n_ver"].sum() == 0


def test_v7_window_measure():
    """A stop inside its decision window pays no block after the decision."""
    env = replace(BASE, block=4, lag=1)
    r = run_policy("GAL", env, D, policy_thresholds("GAL", env))
    assert r["decided"].any() and np.all(r["post_paid"][r["decided"]] == 0)
    r = run_policy("GA", env, D, policy_thresholds("GA", env))
    assert np.all(r["post_paid"][r["decided"] & (r["stop_t"] < H)] > 0)
    # waste per stop: GAL loses exactly the one-period lag; GA a whole further block
    eff = lambda r: r["decided"] & (r["stop_t"] < H)
    g = run_policy("GAL", env, D, policy_thresholds("GAL", env))
    assert np.allclose(g["waste"][eff(g)], 1.0) and np.all(r["waste"][eff(r)] >= 4.0)


def test_v7_refund_accounting():
    """Net value = completion value - payments - fees + refund, refund = kappa x unused prepaid."""
    env = replace(BASE, block=4, recover=0.4)
    r = run_policy("Q", env, D, policy_thresholds("Q", env))
    lhs = r["net"]; rhs = r["completed"] * D.viable * env.value - r["paid"] - r["gov"] + r["refund"]
    assert np.allclose(lhs, rhs) and np.allclose(r["refund"], 0.4 * r["stranded"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("passed:", name)
