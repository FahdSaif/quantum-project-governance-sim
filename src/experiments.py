"""Runs all experiments reported in the manuscript (model v4); writes CSVs to ./results."""
import csv, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from governance_sim import (Env, Draws, POLICIES, run_policy, run_benchmark, own_model_thresholds,
                            policy_thresholds, summarise, tune_threshold, THR_GRID)

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


if __name__ == "__main__":
    t = time.time()
    steps = sys.argv[1:] or ["core", "sweep", "beta_sweep", "structural", "explore"]
    for s in steps:
        globals()[s](); print(s, round(time.time() - t, 1), "s", flush=True)
    json.dump(dict(model="v4", base_seed=BASE_SEED, n_per_seed=N, train_seeds=5, test_seeds=20,
                   futures=1500, threshold_grid=[float(g) for g in THR_GRID]),
              open(OUT / "run_config.json", "w"), indent=2)
