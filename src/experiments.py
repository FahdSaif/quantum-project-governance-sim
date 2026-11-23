"""Runs all experiments reported in the manuscript (model v3); writes CSVs to ./results."""
import csv, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from governance_sim import (Env, Draws, POLICIES, run_policy, run_benchmark, own_model_thresholds,
                            summarise, tune_threshold, THR_GRID)

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
]
CONTRASTS = ["U-Q", "QV-Q", "E-QV", "E-Q", "E-E_m", "GA-Q", "GAV-QV", "GAV-Q", "E-GAV"]


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


def core():
    rows, paired = [], []
    for env in SCENARIOS:
        nets = {}
        own = own_model_thresholds(env)
        for regime in ("tuned", "own_model", "fixed_0.18"):
            for pol in POLICIES:
                if regime == "tuned":
                    th = tune_threshold(pol, env, TRAIN); thr_label = th
                elif regime == "own_model":
                    th = own; thr_label = "own-model path"
                else:
                    th = 0.18; thr_label = 0.18
                res = [run_policy(pol, env, d, th) for d in TEST]
                rows.append(agg(res, pol, env, regime, thr_label))
                nets[(regime, pol)] = np.concatenate([r["net"] for r in res])
        for lab, wv in (("B", False), ("BV", True)):
            if wv and env.rho > 0:
                continue
            res = [run_benchmark(env, d, wv) for d in TEST]
            rows.append(agg(res, lab, env, "benchmark", "optimal"))
        for regime in ("tuned", "own_model"):
            for c in CONTRASTS:
                a, b = c.split("-", 1)
                diff = nets[(regime, a)] - nets[(regime, b)]
                m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(diff.size)
                paired.append(dict(scenario=env.name, regime=regime, contrast=c, diff=float(m),
                                   lo=float(m - 1.96 * se), hi=float(m + 1.96 * se)))
        print("done", env.name, flush=True)
    write("core_results.csv", rows); write("core_paired_contrasts.csv", paired)


def sweep():
    rows = []
    for env in SCENARIOS[:3]:
        for pol in ("Q", "U", "E", "E_m"):
            for th in THR_GRID:
                v = np.mean([run_policy(pol, env, d, th)["net"].mean() for d in TEST[:5]])
                rows.append(dict(scenario=env.name, policy=pol, threshold=float(th), mean_net_value=float(v)))
    write("threshold_sweep.csv", rows)


def explore(n_futures=1500):
    names = ["q_true", "miscal", "block_idx", "monitor_fee", "review_fee", "ver_fee",
             "ver_adv", "rho", "rho_known", "value", "beta", "lag_idx"]
    lo = np.array([0.55, -0.10, 0, 0.00, 0.10, 0.10, -0.05, 0.0, 0, 18, 0.0, 0])
    hi = np.array([0.85, 0.20, 5, 0.50, 0.60, 1.00, 0.25, 0.8, 2, 40, 0.6, 3])
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=BASE_SEED).random(n_futures), lo, hi)
    blocks = [1, 2, 3, 4, 6]
    grid = np.round(np.arange(0.04, 0.81, 0.04), 2)
    pols = ("F", "Q", "QV", "GA", "GAV", "U", "E")
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
        own = own_model_thresholds(env)
        for pol in pols:
            vals = [run_policy(pol, env, tr, th)["net"].mean() for th in grid]
            th = float(grid[int(np.argmax(vals))])
            out[f"{pol}_thr"] = th
            out[f"{pol}_net"] = float(run_policy(pol, env, te, th)["net"].mean())
            out[f"{pol}_own"] = float(run_policy(pol, env, te, own)["net"].mean())
        out["B_net"] = float(run_benchmark(env, te, False)["net"].mean())
        for suffix in ("net", "own"):
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
    steps = sys.argv[1:] or ["core", "sweep", "explore"]
    for s in steps:
        globals()[s](); print(s, round(time.time() - t, 1), "s", flush=True)
    json.dump(dict(model="v3", base_seed=BASE_SEED, n_per_seed=N, train_seeds=5, test_seeds=20,
                   futures=1500, threshold_grid=[float(g) for g in THR_GRID]),
              open(OUT / "run_config.json", "w"), indent=2)
