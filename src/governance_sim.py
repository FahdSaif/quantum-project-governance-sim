"""Evidence-gated commitment in exploratory technology projects: simulation model (v3).

Synthetic computational experiment. No real projects, organisations or human data.
Python 3.10+, numpy.

Version history
  v1  original draft script (orig.py, included for verification).
  v2  corrected costs, common random numbers, tuned thresholds, QV/E_m controls,
      Bayes-optimal benchmark B, exploratory ensemble.
  v3  (this file) adds: own-model ("feasible") stopping thresholds; gates aligned with
      commitment renewals (GA, GAV); known-correlation verification; a verification-aware
      benchmark (BV); a continuation bias of the decision forum; decision lag.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace

H = 12
MONTHLY_COST = 1.0


@dataclass(frozen=True)
class Env:
    name: str = "base"
    q_true: float = 0.80        # true reliability of routine technical evidence
    q_assumed: float = 0.80     # reliability assumed by decision-makers
    block: int = 1              # non-refundable commitment block (periods)
    monitor_fee: float = 0.12   # per active period, evidence-triggered policies
    review_fee: float = 0.35    # per formal review
    ver_fee: float = 0.55       # per verification purchase
    q_ver: float = 0.82         # accuracy of independent verification evidence
    rho: float = 0.0            # prob. verification merely echoes routine evidence
    rho_known: bool = False     # decision-makers know rho and update correctly
    value: float = 28.0         # terminal benefit of a completed viable project
    beta: float = 0.0           # continuation bias: effective threshold = (1-beta)*threshold
    lag: int = 0                # periods between a stop decision and its taking effect
    cost: float = MONTHLY_COST  # operating cost per period


POLICIES = ("F", "Q", "QV", "GA", "GAV", "U", "E", "E_m")
VERIFYING = ("QV", "GAV", "E", "E_m")
TRIGGERED = ("U", "E", "E_m")


def aligned_gates(block: int) -> tuple[int, ...]:
    """Gates at the end of a commitment block, spaced at least three periods apart."""
    s = block * int(np.ceil(3 / block))
    return tuple(t for t in range(s, H, s))


def decision_points(policy: str, env: Env) -> tuple[int, ...]:
    if policy == "F":
        return (6, 9)
    if policy in ("Q", "QV"):
        return (3, 6, 9)
    if policy in ("GA", "GAV"):
        return aligned_gates(env.block)
    return tuple(range(1, H))


class Draws:
    """Common random numbers for one seed."""
    def __init__(self, seed: int, n: int):
        rng = np.random.default_rng(seed)
        self.n = n
        self.prior = rng.beta(3.0, 3.0, n)
        self.u_viable = rng.random(n)
        self.u_sig = rng.random((n, H))
        self.u_ver = rng.random((n, H))
        self.u_echo = rng.random((n, H))
        self.viable = self.u_viable < self.prior


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _ver_loglik(env: Env, q_ver: float, w, s):
    """Log-likelihood ratio of a verification signal w (given routine signal s)."""
    if not env.rho_known or env.rho == 0:
        return np.where(w, 1.0, -1.0) * np.log(q_ver / (1 - q_ver))
    r = env.rho
    p1 = r * s + (1 - r) * q_ver          # P(w=1 | V=1, s)
    p0 = r * s + (1 - r) * (1 - q_ver)    # P(w=1 | V=0, s)
    return np.where(w, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))


def as_path(thr) -> np.ndarray:
    """Threshold as a per-period array indexed by t (0..H)."""
    if np.ndim(thr) == 0:
        return np.full(H + 1, float(thr))
    return np.asarray(thr, dtype=float)


def run_policy(policy: str, env: Env, d: Draws, thr) -> dict:
    """Simulate a governance policy on common draws; thr is a scalar or per-period path."""
    n, viable = d.n, d.viable
    sig = d.u_sig < np.where(viable[:, None], env.q_true, 1 - env.q_true)
    q_ver = env.q_true if policy == "E_m" else env.q_ver
    indep = d.u_ver < np.where(viable[:, None], q_ver, 1 - q_ver)
    ver = np.where(d.u_echo < env.rho, sig, indep)
    ll = np.log(env.q_assumed / (1 - env.q_assumed))
    th = as_path(thr) * (1 - env.beta)
    points = set(decision_points(policy, env))
    trig, verifies = policy in TRIGGERED, policy in VERIFYING

    active = np.ones(n, bool); decided = np.zeros(n, bool); due = np.full(n, H + 1)
    paid = np.zeros(n); mon = np.zeros(n); rev = np.zeros(n)
    stranded = np.zeros(n); stop_t = np.full(n, H); n_ver = np.zeros(n)
    lo = np.log(d.prior / (1 - d.prior))
    for t in range(1, H + 1):
        if (t - 1) % env.block == 0:
            paid[active] += min(env.block, H - t + 1) * env.cost
        lo[active] += np.where(sig[active, t - 1], 1.0, -1.0) * ll
        if t == H:
            break
        if trig:
            mon[active & ~decided] += env.monitor_fee
        if t in points:
            open_ = active & ~decided
            p = _sig(lo)
            review = open_ & (p < th[t]) if trig else open_.copy()
            rev[review] += env.review_fee
            if verifies:
                buy = review & (p < th[t])
                rev[buy] += env.ver_fee; n_ver[buy] += 1
                lo[buy] += _ver_loglik(env, q_ver, ver[buy, t - 1], sig[buy, t - 1])
                p = _sig(lo)
            stop = review & (p < th[t])
            decided |= stop
            due[stop] = t + env.lag
        effect = active & decided & (due == t)
        stop_t[effect] = t
        paid_through = min(H, ((t - 1) // env.block + 1) * env.block)
        stranded[effect] = (paid_through - t) * env.cost
        active[effect] = False
    completed = active
    net = completed * viable * env.value - paid - mon - rev
    return dict(net=net, completed=completed, gov=mon + rev, stranded=stranded,
                stop_t=stop_t, paid=paid, n_ver=n_ver)


# ---------------------------------------------------------------- benchmarks
GRID = np.linspace(0.0, 1.0, 4001)


def _bayes(p, q):
    pp = p * q + (1 - p) * (1 - q)
    up = np.divide(p * q, pp, out=np.zeros_like(p), where=pp > 0)
    dn = np.divide(p * (1 - q), 1 - pp, out=np.zeros_like(p), where=(1 - pp) > 0)
    return pp, up, dn


def dp_tables(env: Env, q: float, with_ver: bool):
    """Backward induction with costless continuous review.
    Returns per-period continuation values C_t(p) and verification values Ver_t(p)."""
    p = GRID
    V = p * env.value
    C = np.zeros((H + 1, p.size)); VER = np.full((H + 1, p.size), -np.inf)
    for t in range(H - 1, 0, -1):
        c_next = min(env.block, H - t) * env.cost if t % env.block == 0 else 0.0
        pp, up, dn = _bayes(p, q)
        C[t] = -c_next + pp * np.interp(up, p, V) + (1 - pp) * np.interp(dn, p, V)
        A = np.maximum(C[t], 0.0)
        Vt = A
        if with_ver:
            pw, wup, wdn = _bayes(p, env.q_ver)
            VER[t] = -env.ver_fee + pw * np.interp(wup, p, A) + (1 - pw) * np.interp(wdn, p, A)
            Vt = np.maximum(A, VER[t])
        V = Vt
    return C, VER


def own_model_thresholds(env: Env) -> np.ndarray:
    """Stopping thresholds a decision-maker can compute from its OWN (possibly miscalibrated)
    model: stop when the expected value of continuing is negative under assumed reliability."""
    C, _ = dp_tables(replace(env, beta=0.0), env.q_assumed, False)
    thr = np.zeros(H + 1)
    for t in range(1, H):
        go = C[t] > 0
        thr[t] = GRID[np.argmax(go)] if go.any() else 1.0
    return thr


def run_benchmark(env: Env, d: Draws, with_ver: bool) -> dict:
    """Calibrated optimal stopping (B) or optimal stopping with costly verification (BV)."""
    C, VER = dp_tables(env, env.q_true, with_ver)
    n, viable = d.n, d.viable
    sig = d.u_sig < np.where(viable[:, None], env.q_true, 1 - env.q_true)
    ver = d.u_ver < np.where(viable[:, None], env.q_ver, 1 - env.q_ver)
    ll, vll = np.log(env.q_true / (1 - env.q_true)), np.log(env.q_ver / (1 - env.q_ver))
    active = np.ones(n, bool); paid = np.zeros(n); gov = np.zeros(n)
    stranded = np.zeros(n); stop_t = np.full(n, H); n_ver = np.zeros(n)
    lo = np.log(d.prior / (1 - d.prior))
    for t in range(1, H + 1):
        if (t - 1) % env.block == 0:
            paid[active] += min(env.block, H - t + 1) * env.cost
        lo[active] += np.where(sig[active, t - 1], 1.0, -1.0) * ll
        if t == H:
            break
        p = _sig(lo)
        cont = np.interp(p, GRID, C[t])
        if with_ver:
            vv = np.interp(p, GRID, VER[t])
            buy = active & (vv > np.maximum(cont, 0.0))
            gov[buy] += env.ver_fee; n_ver[buy] += 1
            lo[buy] += np.where(ver[buy, t - 1], 1.0, -1.0) * vll
            p = _sig(lo)
            cont = np.interp(p, GRID, C[t])
        stop = active & (cont <= 0)
        stop_t[stop] = t
        paid_through = min(H, ((t - 1) // env.block + 1) * env.block)
        stranded[stop] = (paid_through - t) * env.cost
        active[stop] = False
    net = active * viable * env.value - paid - gov
    return dict(net=net, completed=active, gov=gov, stranded=stranded, stop_t=stop_t,
                paid=paid, n_ver=n_ver)


def summarise(res: dict, d: Draws) -> dict:
    v = d.viable
    return dict(
        mean_net_value=float(res["net"].mean()),
        premature_stop_pct=float(100 * np.mean(~res["completed"][v])),
        nonviable_termination_pct=float(100 * np.mean(~res["completed"][~v])),
        mean_gov_overhead=float(res["gov"].mean()),
        mean_stranded_prepaid=float(res["stranded"].mean()),
        mean_verifications=float(res["n_ver"].mean()),
    )


THR_GRID = np.round(np.arange(0.02, 0.81, 0.02), 2)


def tune_threshold(policy: str, env: Env, train: list[Draws], grid=THR_GRID) -> float:
    vals = [np.mean([run_policy(policy, env, d, th)["net"].mean() for d in train]) for th in grid]
    return float(grid[int(np.argmax(vals))])
