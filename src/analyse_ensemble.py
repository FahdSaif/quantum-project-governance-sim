"""Scenario discovery and global sensitivity for the exploratory ensemble (writes results/ensemble_*.csv)."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

R = Path(__file__).resolve().parents[1] / "results"
x = pd.read_csv(R / "exploratory_futures.csv")
P = ["F", "Q", "QV", "GA", "GAV", "U", "E"]
FEAT = ["q_true", "miscal", "block", "monitor_fee", "review_fee", "ver_fee", "ver_adv", "rho",
        "rho_known", "value", "beta", "lag"]
LABEL = {"q_true": "Evidence reliability q", "miscal": "Miscalibration", "block": "Commitment block",
         "monitor_fee": "Monitoring fee", "review_fee": "Review fee", "ver_fee": "Verification fee",
         "ver_adv": "Verification accuracy advantage", "rho": "Shared-bias share", "rho_known": "Correlation known",
         "value": "Terminal value", "beta": "Continuation bias", "lag": "Decision lag"}
out = {}

# regret tables (both regimes)
reg = []
for suf, regime in (("own", "own-model thresholds"), ("net", "tuned thresholds")):
    for p in P:
        r = x[f"{p}_regret_{suf}"]
        reg.append(dict(regime=regime, policy=p, best_pct=100 * (x[f"winner_{suf}"] == p).mean(),
                        mean_regret=r.mean(), p90_regret=r.quantile(.9), max_regret=r.max()))
pd.DataFrame(reg).to_csv(R / "ensemble_regret.csv", index=False)

# outcome contrasts (own-model = primary)
s = "own"
x["timing"] = x[f"E_{s}"] - x[f"QV_{s}"]
x["verif"] = x[f"QV_{s}"] - x[f"Q_{s}"]
x["alignv"] = x[f"GA_{s}"] - x[f"Q_{s}"]
x["trig_wins"] = (x[[f"U_{s}", f"E_{s}"]].max(axis=1) > x[[f"F_{s}", f"Q_{s}", f"QV_{s}", f"GA_{s}", f"GAV_{s}"]].max(axis=1)).astype(int)

imp_rows = []
for target in ["timing", "verif", "alignv"]:
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1, oob_score=True)
    rf.fit(x[FEAT], x[target])
    pi = permutation_importance(rf, x[FEAT], x[target], n_repeats=10, random_state=0, n_jobs=-1)
    tot = pi.importances_mean.clip(min=0).sum()
    for f, v in zip(FEAT, pi.importances_mean):
        imp_rows.append(dict(target=target, factor=LABEL[f], share=max(v, 0) / tot, oob_r2=rf.oob_score_))
pd.DataFrame(imp_rows).to_csv(R / "ensemble_importance.csv", index=False)

# scenario-discovery trees
trees = {}
for name, y in [("trig_wins", x.trig_wins), ("verif_pays", (x.verif > 0).astype(int)), ("align_pays", (x.alignv > 0.05).astype(int))]:
    t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=75, random_state=0).fit(x[FEAT], y)
    trees[name] = dict(text=export_text(t, feature_names=FEAT, show_weights=True), accuracy=t.score(x[FEAT], y),
                       base_rate=float(y.mean()))
json.dump(trees, open(R / "ensemble_trees.json", "w"), indent=2)

# P4 threshold: share of futures in which the best event-triggered policy wins, by monitoring fee
fine = x[(x.block <= 2) & (x.lag == 0)]
bins = np.arange(0, 0.55, 0.05)
share = fine.groupby(pd.cut(fine.monitor_fee, bins), observed=False).trig_wins.agg(["mean", "size"])
share.to_csv(R / "ensemble_p4_monitoring.csv")
out["p4_share_by_fee_fine_blocks"] = {str(k): [round(v["mean"], 3), int(v["size"])] for k, v in share.iterrows()}
out["trig_wins_share_all"] = float(x.trig_wins.mean())
out["gap_B_minus_best_feasible_own"] = float((x.B_net - x[[f"{p}_own" for p in P]].max(axis=1)).mean())
out["gap_B_minus_best_tuned"] = float((x.B_net - x[[f"{p}_net" for p in P]].max(axis=1)).mean())
out["B_dominates_F_Q_U_GA_own_min_margin"] = float((x.B_net - x[[f"{p}_own" for p in ["F", "Q", "U", "GA"]]].max(axis=1)).min())
out["B_dominates_F_Q_U_GA_tuned_min_margin"] = float((x.B_net - x[[f"{p}_net" for p in ["F", "Q", "U", "GA"]]].max(axis=1)).min())
out["align_pays_share_block_ge3"] = float((x[x.block >= 3].alignv > 0.05).mean())
out["align_mean_gain_block_ge3"] = float(x[x.block >= 3].alignv.mean())
out["verif_pays_adv_gt_175_rho_le_2"] = float((x[(x.ver_adv > .175) & (x.rho <= .2)].verif > 0).mean())
out["verif_pays_adv_lt_025_rho_gt_6"] = float((x[(x.ver_adv < .025) & (x.rho > .6)].verif > 0).mean())
out["trig_wins_fee_le_05_block_le_2_lag0"] = [float(x[(x.monitor_fee <= .05) & (x.block <= 2) & (x.lag == 0)].trig_wins.mean()),
                                              int(((x.monitor_fee <= .05) & (x.block <= 2) & (x.lag == 0)).sum())]
out["trig_wins_fee_gt_30"] = float(x[x.monitor_fee > .30].trig_wins.mean())
out["trig_wins_block_ge4"] = float(x[x.block >= 4].trig_wins.mean())
# effect of known correlation where rho > 0.4
hi = x[x.rho > 0.4]
out["verif_mean_rho_gt_4_known_vs_unknown"] = [float(hi[hi.rho_known == 1].verif.mean()), float(hi[hi.rho_known == 0].verif.mean())]
json.dump(out, open(R / "ensemble_summary.json", "w"), indent=2)
print(json.dumps(out, indent=1))
for k, v in trees.items():
    print(k, v["accuracy"], v["base_rate"]); print(v["text"])
print(pd.DataFrame(reg).round(2).to_string())
imp = pd.DataFrame(imp_rows)
print(imp.pivot(index="factor", columns="target", values="share").round(3))
print(imp.groupby("target").oob_r2.first())
