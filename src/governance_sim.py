"""Evidence-gated commitment in exploratory technology projects: simulation model (v5).

Synthetic computational experiment. No real projects, organisations or human data.
Python 3.10+, numpy.

Version history
  v1  original draft script (orig.py, included for verification).
  v2  corrected costs, common random numbers, tuned thresholds, QV/E_m controls,
      Bayes-optimal benchmark B, exploratory ensemble.
  v3  adds: own-model ("feasible") stopping thresholds; gates aligned with
      commitment renewals (GA, GAV); known-correlation verification; a verification-aware
      benchmark (BV); a continuation bias of the decision forum; decision lag.
  v4  (this file) adds: policy-specific own-model thresholds computed by backward induction
      over each policy's OWN decision schedule and decision lag (policy_thresholds); an
      optional "decide now, do not renew" rule (Env.nonrenewal); a lag-aware aligned policy
      GAL (gates l periods before each renewal); serially correlated routine signals
      (Env.persist) as a structural robustness check.
  v5  (this file) adds, all switched off by default so that v4 results are unchanged:
      uncertain termination lag (planned lag Env.lag used for scheduling and stop rules;
      realised lag per project = lag + lag_bias + uniform offset in [-lag_spread, lag_spread]);
      irregular renewal calendars (Env.starts); partly recoverable commitments (Env.recover);
      a routine-evidence cost charged to every policy (Env.report_fee); and value-based
      verification policies (QVo, GAVo, Eo) that buy verification only when its expected
      value under the forum's own model exceeds its fee.
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
    nonrenewal: bool = False    # a stop decided inside a paid block takes effect at the block end
    persist: float = 0.0        # prob. a routine signal repeats the previous one (serial correlation)
    lag_bias: int = 0           # realised minus planned lag (systematic misestimation)
    lag_spread: int = 0         # realised lag offset uniform on [-spread, spread] (truncated at 0)
    starts: tuple | None = None # irregular calendar: periods at which blocks start (overrides block)
    recover: float = 0.0        # share of unused prepaid cost refunded when a project stops
    report_fee: float = 0.0     # cost of routine evidence per active period, charged to all policies


POLICIES = ("F", "Q", "QV", "GA", "GAV", "GAL", "U", "E", "E_m")
OPTIONAL_VER = ("QVo", "GAVo", "GALVo", "Eo")  # value-based (optional) verification
MARGIN = ("GALp",)                             # lag-aware gates planned with a one-period safety margin
VERIFYING = ("QV", "GAV", "E", "E_m") + OPTIONAL_VER
TRIGGERED = ("U", "E", "E_m", "Eo")


# ---------------------------------------------------------------- commitment calendar
def block_starts(env: Env) -> tuple[int, ...]:
    if env.starts is not None:
        st = tuple(sorted(set(int(k) for k in env.starts)))
        assert st[0] == 1 and st[-1] <= H
        return st
    return tuple(range(1, H + 1, env.block))


def block_payment(env: Env, k: int) -> float:
    """Payment due at the start of period k (zero if no block starts then)."""
    st = block_starts(env)
    if k not in st:
        return 0.0
    i = st.index(k)
    end = st[i + 1] - 1 if i + 1 < len(st) else H
    return (end - k + 1) * env.cost


def paid_through(env: Env, t: int) -> int:
    """Last period of the block containing t."""
    nxt = [k for k in block_starts(env) if k > t]
    return nxt[0] - 1 if nxt else H


def calendar_aligned_gates(env: Env) -> tuple[int, ...]:
    """GA for any calendar: renewal points (block ends before a further block), chosen greedily
    so that gates are at least three periods apart. Equals aligned_gates(b) for regular blocks."""
    gates, last = [], 0
    for k in block_starts(env)[1:]:
        r = k - 1
        if r >= 3 and r - last >= 3 and r < H:
            gates.append(r); last = r
    return tuple(gates)


def aligned_gates(block: int) -> tuple[int, ...]:
    """Gates at the end of a commitment block, spaced at least three periods apart."""
    s = block * int(np.ceil(3 / block))
    return tuple(t for t in range(s, H, s))


def renewal_aligned_gates(block: int, lag: int) -> tuple[int, ...]:
    """GAL: gates placed `lag` periods before the aligned renewal points, so that a stop
    decision takes effect exactly at the renewal. Equals aligned_gates when lag = 0."""
    return tuple(t - lag for t in aligned_gates(block) if t - lag >= 1)


def decision_points(policy: str, env: Env) -> tuple[int, ...]:
    if policy == "F":
        return (6, 9)
    if policy in ("Q", "QV", "QVo"):
        return (3, 6, 9)
    if policy in ("GA", "GAV", "GAVo"):
        return calendar_aligned_gates(env)
    if policy == "GALp":
        return decision_points("GAL", replace(env, lag=env.lag + 1))
    if policy in ("GAL", "GALVo"):
        return tuple(t - env.lag for t in calendar_aligned_gates(env) if t - env.lag >= 1)
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
        # drawn last so that all earlier streams are identical to v3
        self.u_rep = rng.random((n, H))
        self.u_lag = rng.random(n)        # v5: realised termination lag (drawn last)


def routine_signals(env: Env, d: Draws) -> np.ndarray:
    """Routine technical signals. With persist > 0 a signal repeats the previous one with that
    probability, otherwise it is a fresh draw; the marginal accuracy stays q."""
    fresh = d.u_sig < np.where(d.viable[:, None], env.q_true, 1 - env.q_true)
    if env.persist <= 0:
        return fresh
    sig = fresh.copy()
    rep = d.u_rep < env.persist
    for t in range(1, H):
        sig[:, t] = np.where(rep[:, t], sig[:, t - 1], fresh[:, t])
    return sig


def stop_effect(t: int, env: Env, lag: int | None = None) -> int:
    """Period at whose end a stop decided at t takes effect (planned lag unless given)."""
    due = t + (env.lag if lag is None else int(lag))
    if env.nonrenewal:
        due = paid_through(env, min(due, H))
    return due


def realised_lags(env: Env, d: Draws) -> np.ndarray:
    """Per-project realised termination lag; equals the planned lag by default."""
    k = env.lag_spread
    off = np.floor(d.u_lag * (2 * k + 1)).astype(int) - k if k > 0 else np.zeros(d.n, int)
    return np.maximum(0, env.lag + env.lag_bias + off)


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
    """Simulate a governance policy on common draws; thr is a scalar or per-period path.
    Optional-verification policies (QVo, GAVo, Eo) need policy-specific tables and take
    thr = policy_tables(policy, env)."""
    n, viable = d.n, d.viable
    sig = routine_signals(env, d)
    q_ver = env.q_true if policy == "E_m" else env.q_ver
    indep = d.u_ver < np.where(viable[:, None], q_ver, 1 - q_ver)
    ver = np.where(d.u_echo < env.rho, sig, indep)
    ll = np.log(env.q_assumed / (1 - env.q_assumed))
    optional = policy in OPTIONAL_VER
    if optional:
        tabs = thr if isinstance(thr, dict) else policy_tables(policy, env)
        thr = tabs["thr"]
    th = as_path(thr) * (1 - env.beta)
    points = set(decision_points(policy, env))
    trig, verifies = policy in TRIGGERED, policy in VERIFYING
    starts = set(block_starts(env))
    lag_i = realised_lags(env, d)
    uncertain_lag = env.lag_bias != 0 or env.lag_spread > 0

    active = np.ones(n, bool); decided = np.zeros(n, bool); due = np.full(n, H + 1)
    paid = np.zeros(n); mon = np.zeros(n); rev = np.zeros(n); refund = np.zeros(n)
    stranded = np.zeros(n); stop_t = np.full(n, H); n_ver = np.zeros(n); post_paid = np.zeros(n)
    dec_t = np.full(n, H); waste = np.zeros(n)
    lo = np.log(d.prior / (1 - d.prior))
    for t in range(1, H + 1):
        if t in starts:
            paid[active] += block_payment(env, t)
            post_paid[active & decided] += block_payment(env, t)
        if env.report_fee:
            mon[active] += env.report_fee
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
                if optional:
                    s_t = sig[:, t - 1]
                    ev = np.where(s_t, np.interp(p, GRID, tabs["evsi"][1][t]), np.interp(p, GRID, tabs["evsi"][0][t]))
                    buy = review & (ev > env.ver_fee)
                else:
                    buy = review & (p < th[t])
                rev[buy] += env.ver_fee; n_ver[buy] += 1
                lo[buy] += _ver_loglik(env, q_ver, ver[buy, t - 1], sig[buy, t - 1])
                p = _sig(lo)
            stop = review & (p < th[t])
            decided |= stop
            dec_t[stop] = t
            if uncertain_lag:
                due[stop] = [stop_effect(t, env, l) for l in lag_i[stop]]
            else:
                due[stop] = stop_effect(t, env)
        effect = active & decided & (due == t)
        stop_t[effect] = t
        stranded[effect] = (paid_through(env, t) - t) * env.cost
        refund[effect] = env.recover * stranded[effect]
        # committed spending lost to a stop: paid periods from the decision to the end of the paid
        # block in which the stop takes effect, net of refunds (minimum: the execution time itself)
        waste[effect] = (paid_through(env, t) - dec_t[effect]) * env.cost - refund[effect]
        active[effect] = False
    completed = active
    net = completed * viable * env.value - paid - mon - rev + refund
    return dict(net=net, completed=completed, gov=mon + rev, stranded=stranded,
                stop_t=stop_t, paid=paid, n_ver=n_ver, refund=refund, decided=decided,
                post_paid=post_paid, waste=waste)


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
        c_next = block_payment(env, t + 1)
        pp, up, dn = _bayes(p, q)
        C[t] = -c_next + pp * np.interp(up, p, V) + (1 - pp) * np.interp(dn, p, V)
        A = np.maximum(C[t], env.recover * (paid_through(env, t) - t) * env.cost)
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


def policy_tables(policy: str, env: Env) -> dict:
    """Own-model stop rule by backward induction under the assumed reliability over the policy's
    OWN decision schedule and PLANNED lag (v4). Between decision points the forum must continue
    and pay any renewal that falls due. Fees are excluded so that policies differ only in their
    schedule. v5: unused prepaid cost is partly refunded on stopping (Env.recover), and for
    optional-verification policies the table also holds the expected value of one verification
    signal under the forum's model (independent accuracy q_ver), net of nothing (compare with fee)."""
    if policy in MARGIN:   # plan the decision window for one period more than the point estimate
        return policy_tables("GAL", replace(env, lag=env.lag + 1))
    q, R, c = env.q_assumed, env.value, env.cost
    p = GRID
    pts = set(decision_points(policy, env))
    def cost_between(t, u):  # renewals paid in periods t+1..u
        return sum(block_payment(env, k) for k in range(t + 1, min(u, H) + 1))
    W = p * R
    thr = np.zeros(H + 1)
    evsi_s = (np.zeros((H + 1, p.size)), np.zeros((H + 1, p.size)))
    for t in range(H - 1, 0, -1):
        pp, up, dn = _bayes(p, q)
        C = -block_payment(env, t + 1) + pp * np.interp(up, p, W) + (1 - pp) * np.interp(dn, p, W)
        if t in pts:
            due = stop_effect(t, env)
            salvage = env.recover * (paid_through(env, min(due, H)) - due) * c if due < H else 0.0
            if due >= H or (cost_between(due, H) == 0 and salvage == 0):
                # a stop that cannot take effect before completion, or that would avoid no
                # further payment, saves nothing: never stop (v3 broke this tie towards stopping)
                thr[t] = 0.0
                W = C
                continue
            S = -cost_between(t, due) + salvage
            go = C > S            # ties resolved towards stopping, as in v3
            thr[t] = GRID[np.argmax(go)] if go.any() else 1.0
            A = np.maximum(C, S)
            if policy in OPTIONAL_VER:
                for si in (0, 1):
                    evsi_s[si][t] = _evsi(env, p, A, si)
            W = A
        else:
            W = C
    return dict(thr=thr, evsi=evsi_s)


def _evsi(env: Env, p, A, s: int):
    """Expected value of one verification signal at belief p, given today's routine signal s,
    under the FORUM's verification model: independent with accuracy q_ver, or, if the forum
    knows rho, echoing s with probability rho. A = max(continue, stop) at the same review.
    Myopic (one signal, no future verification); not dynamically optimal."""
    qv = env.q_ver
    if env.rho_known and env.rho > 0:
        p1 = env.rho * s + (1 - env.rho) * qv          # P(w=1 | V=1, s)
        p0 = env.rho * s + (1 - env.rho) * (1 - qv)    # P(w=1 | V=0, s)
    else:
        p1, p0 = qv, 1 - qv
    pw = p * p1 + (1 - p) * p0
    up = np.divide(p * p1, pw, out=np.zeros_like(p), where=pw > 0)
    dn = np.divide(p * (1 - p1), 1 - pw, out=np.zeros_like(p), where=(1 - pw) > 0)
    return pw * np.interp(up, p, A) + (1 - pw) * np.interp(dn, p, A) - A


def policy_thresholds(policy: str, env: Env) -> np.ndarray:
    """v4 own-model thresholds (see policy_tables). For U/E/E_m with lag 0 this equals
    own_model_thresholds except for one tie: v3 stopped projects with belief below one grid
    step even when the stop avoided no further renewal."""
    return policy_tables(policy, env)["thr"]


def run_benchmark(env: Env, d: Draws, with_ver: bool) -> dict:
    """Calibrated optimal stopping (B) or optimal stopping with costly verification (BV)."""
    C, VER = dp_tables(env, env.q_true, with_ver)
    n, viable = d.n, d.viable
    sig = routine_signals(env, d)
    ver = d.u_ver < np.where(viable[:, None], env.q_ver, 1 - env.q_ver)
    ll, vll = np.log(env.q_true / (1 - env.q_true)), np.log(env.q_ver / (1 - env.q_ver))
    active = np.ones(n, bool); paid = np.zeros(n); gov = np.zeros(n)
    stranded = np.zeros(n); stop_t = np.full(n, H); n_ver = np.zeros(n)
    lo = np.log(d.prior / (1 - d.prior))
    starts = set(block_starts(env)); refund = np.zeros(n)
    for t in range(1, H + 1):
        if t in starts:
            paid[active] += block_payment(env, t)
        if env.report_fee:
            gov[active] += env.report_fee
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
        salvage = env.recover * (paid_through(env, t) - t) * env.cost
        stop = active & (cont <= salvage)
        stop_t[stop] = t
        stranded[stop] = (paid_through(env, t) - t) * env.cost
        refund[stop] = env.recover * stranded[stop]
        active[stop] = False
    net = active * viable * env.value - paid - gov + refund
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
