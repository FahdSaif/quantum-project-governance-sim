"""Runs all experiments reported in the manuscript (model v5); writes CSVs to ./results."""
import csv, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from governance_sim import (H, Env, Draws, POLICIES, run_policy, run_benchmark, own_model_thresholds,
                            policy_thresholds, policy_tables, OPTIONAL_VER, summarise, tune_threshold,
                            THR_GRID)

OUT = Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(exist_ok=True)
BASE_SEED = 20261122
N = 10_000
TRAIN = [Draws(BASE_SEED + i, N) for i in range(5)]
TEST = [Draws(BASE_SEED + 100 + i, N) for i in range(20)]

S1 = Env("S1 Calibrated, strong evidence", .80, .80, 1, .12)
SCENARIOS = [
    S1,
    Env("S2 Overconfident, weak evidence", .62, .80, 1, .12),
    Env("S3 Calibrated, weak evidence", .62, .62, 1, .12),
    Env("S4 Large commitment blocks", .80, .80, 4, .12),
    Env("S5 Costly monitoring", .80, .80, 1, .45),
    Env("S6 Shared-bias verification (misperceived)", .62, .80, 1, .12, rho=0.6),
    Env("S6b Shared-bias verification (known)", .62, .80, 1, .12, rho=0.6, rho_known=True),
    replace(S1, name="S7 Escalation-prone forum", beta=0.5),
    replace(S1, name="S8 Two-period decision lag", lag=2),
    Env("S9 Large blocks, one-period lag", .80, .80, 4, .12, lag=1),
]
CONTRASTS = ["U-Q", "QV-Q", "E-QV", "E-Q", "E-E_m", "GA-Q", "GAL-Q", "GAL-GA", "GAV-QV", "GAV-Q", "E-GAV",
             "E-U"]
# own-model rules: (a) v3 continuous-review rule, (b) policy-specific (primary), (c) (b) + non-renewal
OWN_RULES = {"own_v3": "v3", "own_model": "policy", "own_nr": "policy_nr"}


def own_thr(rule, pol, env):
    if rule == "v3":
        return own_model_thresholds(env), env
    if rule == "policy":
        return policy_thresholds(pol, env), env
    e = replace(env, nonrenewal=True)
    return policy_thresholds(pol, e), e


def write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def agg(res, label, env, regime, thr):
    summ = [summarise(r, d) for r, d in zip(res, TEST)]
    row = dict(scenario=env.name, regime=regime, policy=label, threshold=thr,
               seed_sd=float(np.std([r["net"].mean() for r in res], ddof=1)))
    for k in summ[0]:
        row[k] = float(np.mean([s[k] for s in summ]))
    return row


def paired_row(env, regime, contrast, diff):
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(diff.size)
    return dict(scenario=env.name, regime=regime, contrast=contrast, diff=float(m),
                lo=float(m - 1.96 * se), hi=float(m + 1.96 * se))


def core():
    rows, paired = [], []
    for env in SCENARIOS:
        nets, govs = {}, {}
        for regime in ("tuned", *OWN_RULES, "fixed_0.18"):
            for pol in POLICIES:
                e = env
                if regime == "tuned":
                    th = tune_threshold(pol, env, TRAIN); thr_label = th
                elif regime in OWN_RULES:
                    th, e = own_thr(OWN_RULES[regime], pol, env); thr_label = "own-model path"
                else:
                    th = 0.18; thr_label = 0.18
                res = [run_policy(pol, e, d, th) for d in TEST]
                rows.append(agg(res, pol, env, regime, thr_label))
                nets[(regime, pol)] = np.concatenate([r["net"] for r in res])
                govs[(regime, pol)] = np.concatenate([r["gov"] for r in res])
        for lab, wv in (("B", False), ("BV", True)):
            if wv and env.rho > 0:
                continue
            res = [run_benchmark(env, d, wv) for d in TEST]
            rows.append(agg(res, lab, env, "benchmark", "optimal"))
        for regime in ("tuned", *OWN_RULES):
            for c in CONTRASTS:
                a, b = c.split("-", 1)
                paired.append(paired_row(env, regime, c, nets[(regime, a)] - nets[(regime, b)]))
            # GA - Q split into a review-fee part (fewer gates) and a decision-timing part
            for c in ("GA-Q", "GAL-Q"):
                a, b = c.split("-", 1)
                fee = govs[(regime, b)] - govs[(regime, a)]
                paired.append(paired_row(env, regime, c + "|fee", fee))
                paired.append(paired_row(env, regime, c + "|timing", nets[(regime, a)] - nets[(regime, b)] - fee))
        print("done", env.name, flush=True)
    write("core_results.csv", rows); write("core_paired_contrasts.csv", paired)


def sweep():
    rows = []
    for env in SCENARIOS[:3]:
        for pol in ("Q", "U", "E", "E_m"):
            for th in THR_GRID:
                v = np.mean([run_policy(pol, env, d, th)["net"].mean() for d in TEST])
                rows.append(dict(scenario=env.name, policy=pol, threshold=float(th), mean_net_value=float(v)))
    write("threshold_sweep.csv", rows)


def beta_sweep():
    """Continuation bias: loss relative to beta = 0 for verified vs unverified pairs (S1 base)."""
    rows = []
    pols = ("Q", "QV", "U", "E", "GA", "GAV")
    base = {}
    for beta in np.round(np.arange(0, 0.61, 0.1), 1):
        env = replace(S1, name=f"beta={beta}", beta=float(beta))
        for regime in ("own_model", "tuned"):
            for pol in pols:
                th = policy_thresholds(pol, env) if regime == "own_model" else tune_threshold(pol, env, TRAIN)
                res = [run_policy(pol, env, d, th) for d in TEST]
                net = np.concatenate([r["net"] for r in res])
                summ = [summarise(r, d) for r, d in zip(res, TEST)]
                if beta == 0:
                    base[(regime, pol)] = net
                loss = base[(regime, pol)] - net
                rows.append(dict(beta=float(beta), regime=regime, policy=pol, mean_net_value=float(net.mean()),
                                 loss=float(loss.mean()), loss_se=float(loss.std(ddof=1) / np.sqrt(loss.size)),
                                 nonviable_termination_pct=float(np.mean([s["nonviable_termination_pct"] for s in summ])),
                                 mean_stop_period_nonviable=float(np.mean([r["stop_t"][~d.viable].mean() for r, d in zip(res, TEST)]))))
    write("beta_sweep.csv", rows)


def structural():
    """Serially correlated routine signals (persist = 0.3), treated as independent by the forum."""
    rows, paired = [], []
    for env0 in (SCENARIOS[0], SCENARIOS[1], SCENARIOS[3]):
        for kappa in (0.0, 0.3):
            env = replace(env0, persist=kappa)
            nets = {}
            for pol in POLICIES:
                res = [run_policy(pol, env, d, policy_thresholds(pol, env)) for d in TEST]
                nets[pol] = np.concatenate([r["net"] for r in res])
                rows.append(dict(scenario=env0.name, persist=kappa, policy=pol, mean_net_value=float(nets[pol].mean())))
            for c in ("U-Q", "QV-Q", "E-QV", "E-E_m", "GA-Q", "GAL-Q"):
                a, b = c.split("-", 1)
                r = paired_row(env0, f"persist={kappa}", c, nets[a] - nets[b]); r["persist"] = kappa
                paired.append(r)
    write("structural_results.csv", rows); write("structural_contrasts.csv", paired)


def explore(n_futures=1500):
    names = ["q_true", "miscal", "block_idx", "monitor_fee", "review_fee", "ver_fee",
             "ver_adv", "rho", "rho_known", "value", "beta", "lag_idx"]
    lo = np.array([0.55, -0.10, 0, 0.00, 0.10, 0.10, -0.05, 0.0, 0, 18, 0.0, 0])
    hi = np.array([0.85, 0.20, 5, 0.50, 0.60, 1.00, 0.25, 0.8, 2, 40, 0.6, 3])
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=BASE_SEED).random(n_futures), lo, hi)
    blocks = [1, 2, 3, 4, 6]
    grid = np.round(np.arange(0.04, 0.81, 0.04), 2)
    pols = ("F", "Q", "QV", "GA", "GAV", "GAL", "U", "E")
    rows = []
    for i, x in enumerate(X):
        p = dict(zip(names, x))
        qa = float(np.clip(p["q_true"] + p["miscal"], 0.51, 0.95))
        qv = float(np.clip(p["q_true"] + p["ver_adv"], 0.51, 0.97))
        env = Env(f"F{i}", p["q_true"], qa, blocks[min(int(p["block_idx"]), 4)], p["monitor_fee"],
                  p["review_fee"], p["ver_fee"], qv, p["rho"], bool(int(p["rho_known"])), p["value"],
                  p["beta"], min(int(p["lag_idx"]), 2))
        tr, te = Draws(BASE_SEED + 10_000 + i, 3000), Draws(BASE_SEED + 20_000 + i, 6000)
        out = dict(future=i, q_true=env.q_true, q_assumed=qa, miscal=qa - env.q_true, block=env.block,
                   monitor_fee=env.monitor_fee, review_fee=env.review_fee, ver_fee=env.ver_fee,
                   q_ver=qv, ver_adv=qv - env.q_true, rho=env.rho, rho_known=int(env.rho_known),
                   value=env.value, beta=env.beta, lag=env.lag)
        own_v3 = own_model_thresholds(env)
        for pol in pols:
            vals = [run_policy(pol, env, tr, th)["net"].mean() for th in grid]
            th = float(grid[int(np.argmax(vals))])
            out[f"{pol}_thr"] = th
            out[f"{pol}_net"] = float(run_policy(pol, env, te, th)["net"].mean())
            r = run_policy(pol, env, te, policy_thresholds(pol, env))
            out[f"{pol}_own"] = float(r["net"].mean())
            out[f"{pol}_gov"] = float(r["gov"].mean())
            out[f"{pol}_v3"] = float(run_policy(pol, env, te, own_v3)["net"].mean())
            enr = replace(env, nonrenewal=True)
            out[f"{pol}_nr"] = float(run_policy(pol, enr, te, policy_thresholds(pol, enr))["net"].mean())
        out["B_net"] = float(run_benchmark(env, te, False)["net"].mean())
        for suffix in ("net", "own", "v3", "nr"):
            best = max(out[f"{q}_{suffix}"] for q in pols)
            for pol in pols:
                out[f"{pol}_regret_{suffix}"] = best - out[f"{pol}_{suffix}"]
            out[f"winner_{suffix}"] = max(pols, key=lambda q: out[f"{q}_{suffix}"])
        rows.append(out)
        if i % 250 == 0:
            print("future", i, flush=True)
    write("exploratory_futures.csv", rows)


# ================================================================== v5 experiments
def rule(pol, env):
    return policy_tables(pol, env) if pol in OPTIONAL_VER else policy_thresholds(pol, env)


def nets(pol, env, draws=None):
    th = rule(pol, env)
    res = [run_policy(pol, env, d, th) for d in (draws or TEST)]
    return np.concatenate([r["net"] for r in res]), res


def lag_uncertainty():
    """Experiment A: gates and stop rules use the PLANNED lag; the realised lag per project is
    planned + bias + uniform offset in [-spread, spread] (truncated at 0). GALp plans one extra
    period. Policies never see the realised lag."""
    rows, paired = [], []
    for b in (2, 3, 4, 6):
        for lp in (0, 1, 2):
            for bias in (-1, 0, 1):
                if lp + bias < 0:
                    continue
                for spread in (0, 1):
                    env = Env(f"b{b} plan{lp} bias{bias:+d} spread{spread}", .80, .80, b, .12,
                              lag=lp, lag_bias=bias, lag_spread=spread)
                    v = {p: nets(p, env)[0] for p in ("Q", "GA", "GAL", "GALp", "U")}
                    orc = nets("GAL", replace(env, lag=lp + bias, lag_bias=0))[0]
                    r = dict(block=b, planned=lp, bias=bias, spread=spread, true_mean=lp + bias)
                    for p, x in v.items():
                        r[p] = float(x.mean())
                    r["GAL_oracle"] = float(orc.mean())
                    rows.append(r)
                    for c, diff in (("GAL-Q", v["GAL"] - v["Q"]), ("GALp-Q", v["GALp"] - v["Q"]),
                                    ("GA-Q", v["GA"] - v["Q"]), ("GALp-GAL", v["GALp"] - v["GAL"]),
                                    ("GAL-oracle", v["GAL"] - orc)):
                        pr = paired_row(env, "own_model", c, diff); pr.update(block=b, planned=lp, bias=bias, spread=spread)
                        paired.append(pr)
    write("lagunc_results.csv", rows); write("lagunc_contrasts.csv", paired)


VR_POLS = ("Q", "QV", "QVo", "GA", "GAV", "GAVo", "GAL", "GALVo", "U", "E", "Eo")


def verification_rule():
    """Experiment C: mandatory verification before stopping vs value-based verification, in the
    focal scenarios and over continuation bias."""
    rows, paired = [], []
    envs = list(SCENARIOS) + [replace(S1, name=f"S1 beta={b}", beta=b) for b in (0.2, 0.4, 0.6)] \
        + [replace(SCENARIOS[1], name=f"S2 beta={b}", beta=b) for b in (0.3, 0.6)]
    for env in envs:
        v = {}
        for p in VR_POLS:
            x, res = nets(p, env)
            v[p] = x
            rows.append(dict(scenario=env.name, policy=p, beta=env.beta, mean_net_value=float(x.mean()),
                             verifications=float(np.mean([r["n_ver"].mean() for r in res])),
                             nonviable_termination_pct=float(np.mean([summarise(r, d)["nonviable_termination_pct"] for r, d in zip(res, TEST)])),
                             premature_stop_pct=float(np.mean([summarise(r, d)["premature_stop_pct"] for r, d in zip(res, TEST)]))))
        for c in ("QV-Q", "QVo-Q", "QVo-QV", "E-U", "Eo-U", "Eo-E", "GAVo-GA", "GALVo-GAL"):
            a, b = c.split("-", 1)
            paired.append(paired_row(env, "own_model", c, v[a] - v[b]))
    write("verrule_results.csv", rows); write("verrule_contrasts.csv", paired)


def recoverable():
    """Partly recoverable commitments: share r of unused prepaid cost refunded on stopping.
    Each policy's value is split into completion value, net operating spend (payments minus
    refunds) and governance fees, so that differences can be attributed."""
    rows = []
    for b, lag in ((4, 0), (4, 1), (2, 0), (6, 0)):
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            env = Env(f"b{b} lag{lag} r{r}", .80, .80, b, .12, lag=lag, recover=r)
            out = dict(block=b, lag=lag, recover=r)
            v = {}
            for p in ("Q", "GA", "GAL", "U"):
                x, res = nets(p, env)
                v[p] = x; out[p] = float(x.mean())
                out[p + "_value"] = float(np.mean([(rr["completed"] * d.viable * env.value).mean() for rr, d in zip(res, TEST)]))
                out[p + "_opnet"] = float(np.mean([(rr["paid"] - rr["refund"]).mean() for rr in res]))
                out[p + "_gov"] = float(np.mean([rr["gov"].mean() for rr in res]))
                out[p + "_stranded_net"] = float(np.mean([((1 - r) * rr["stranded"]).mean() for rr in res]))
            out["B"] = float(np.mean([run_benchmark(env, d, False)["net"].mean() for d in TEST]))
            for c in ("GA-Q", "GAL-Q", "U-Q"):
                a, bb = c.split("-")
                d = v[a] - v[bb]; out[c] = float(d.mean()); out[c + "_se"] = float(d.std(ddof=1) / np.sqrt(d.size))
            rows.append(out)
    write("recoverable.csv", rows)


def irregular(n_cal=200):
    """Irregular renewal calendars: random block starts (2-5 blocks over 12 periods)."""
    rng = np.random.default_rng(BASE_SEED + 7)
    rows = []
    for k in range(n_cal):
        cuts = sorted(rng.choice(np.arange(2, 13), size=int(rng.integers(1, 5)), replace=False))
        st = tuple([1] + [int(c) for c in cuts])
        for lag in (0, 1, 2):
            env = Env(f"cal{k} lag{lag}", .80, .80, 1, .12, lag=lag, starts=st)
            v = {p: float(np.mean([run_policy(p, env, d, policy_thresholds(p, env))["net"].mean() for d in TEST[:5]]))
                 for p in ("Q", "GA", "GAL", "GALp", "U")}
            rows.append(dict(calendar=k, starts=" ".join(map(str, st)), n_blocks=len(st), lag=lag,
                             longest_block=int(max(np.diff(list(st) + [13]))), **v))
    write("irregular.csv", rows)


def evidence_cost():
    """Fairness check: a routine-evidence cost per active period charged to every policy."""
    rows = []
    for env0 in (SCENARIOS[0], SCENARIOS[3], SCENARIOS[4]):
        for rf in (0.0, 0.12, 0.30):
            env = replace(env0, report_fee=rf)
            v = {p: nets(p, env)[0] for p in ("Q", "QV", "GA", "GAL", "U", "E")}
            out = dict(scenario=env0.name, report_fee=rf, **{p: float(x.mean()) for p, x in v.items()})
            for c in ("U-Q", "E-QV", "GA-Q"):
                a, b = c.split("-"); out[c] = float((v[a] - v[b]).mean())
            rows.append(out)
    write("evidence_cost.csv", rows)


def never_stop(env, d):
    """Net value if nothing is ever stopped (no governance costs)."""
    return d.viable * env.value - H * env.cost


def flex_index():
    """Exercisable flexibility index EF_g = (V_g - V_never) / (V_B - V_never): the share of the value
    of ideal stopping (benchmark B: calibrated, costless, continuous, immediate) that policy g captures.
    Also a purely temporal measure: committed spending lost per stop (paid periods from the decision
    to the end of the block in which the stop takes effect; its minimum is the execution time)."""
    rows = []
    pols = ("F", "Q", "QV", "QVo", "GA", "GAL", "GALp", "GALVo", "U", "E", "Eo")
    for env in SCENARIOS:
        vn = np.mean([never_stop(env, d).mean() for d in TEST])
        vb = np.mean([run_benchmark(env, d, False)["net"].mean() for d in TEST])
        for p in pols:
            x, res = nets(p, env)
            dec = np.concatenate([r["decided"] for r in res]); wst = np.concatenate([r["waste"] for r in res])
            eff = np.concatenate([r["decided"] & (r["stop_t"] < H) for r in res])
            rows.append(dict(scenario=env.name, policy=p, value=float(x.mean()), never=float(vn), B=float(vb),
                             EF=(float(x.mean()) - vn) / (vb - vn),
                             stops_per_project=float(dec.mean()),
                             waste_per_stop=float(wst[eff].mean()) if eff.any() else float("nan")))
    write("flex_index.csv", rows)


def explore_v5():
    """Study 3: the 1,500 Study-2 futures with uncertain termination lag (bias in {-1,0,1},
    spread in {0,1}) and the v5 policies; own-model rules only (no outcome information)."""
    base = pd.read_csv(OUT / "exploratory_futures.csv")
    rng = np.random.default_rng(BASE_SEED + 30)
    bias = rng.integers(-1, 2, len(base)); spread = rng.integers(0, 2, len(base))
    pols = ("F", "Q", "QV", "QVo", "GA", "GAV", "GAL", "GALp", "GALVo", "U", "E", "Eo")
    rows = []
    for i, f in base.iterrows():
        env = Env(f"F{i}", f.q_true, f.q_assumed, int(f.block), f.monitor_fee, f.review_fee, f.ver_fee,
                  f.q_ver, f.rho, bool(f.rho_known), f.value, f.beta, int(f.lag),
                  lag_bias=int(bias[i]) if f.lag + bias[i] >= 0 else 0, lag_spread=int(spread[i]))
        te = Draws(BASE_SEED + 20_000 + i, 6000)
        out = {k: f[k] for k in ("future", "q_true", "q_assumed", "miscal", "block", "monitor_fee", "review_fee",
                                 "ver_fee", "q_ver", "ver_adv", "rho", "rho_known", "value", "beta", "lag")}
        out.update(lag_bias=env.lag_bias, lag_spread=env.lag_spread)
        for p in pols:
            r = run_policy(p, env, te, rule(p, env))
            out[f"{p}_own"] = float(r["net"].mean()); out[f"{p}_gov"] = float(r["gov"].mean())
        best = max(out[f"{q}_own"] for q in pols)
        for p in pols:
            out[f"{p}_regret"] = best - out[f"{p}_own"]
        out["winner"] = max(pols, key=lambda q: out[f"{q}_own"])
        rows.append(out)
        if i % 250 == 0:
            print("v5 future", i, flush=True)
    write("exploratory_futures_v5.csv", rows)


if __name__ == "__main__":
    t = time.time()
    steps = sys.argv[1:] or ["core", "sweep", "beta_sweep", "structural", "explore",
                             "lag_uncertainty", "verification_rule", "recoverable", "irregular",
                             "evidence_cost", "flex_index", "explore_v5"]
    for s in steps:
        globals()[s](); print(s, round(time.time() - t, 1), "s", flush=True)
    json.dump(dict(model="v5", base_seed=BASE_SEED, n_per_seed=N, train_seeds=5, test_seeds=20,
                   futures=1500, threshold_grid=[float(g) for g in THR_GRID]),
              open(OUT / "run_config.json", "w"), indent=2)
