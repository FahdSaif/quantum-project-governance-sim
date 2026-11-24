"""Builds manuscript figures from ./results CSVs (model v5)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
F = Path(__file__).resolve().parents[1] / "figures"; F.mkdir(exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "savefig.dpi": 300})

# ---- Figure 2: decomposition (own-model thresholds)
p = pd.read_csv(R / "core_paired_contrasts.csv"); p = p[p.regime == "own_model"]
sc = list(dict.fromkeys(p.scenario))
lab = ["S1\nCalibrated,\nstrong", "S2\nOverconf.,\nweak", "S3\nCalibrated,\nweak", "S4\nLarge\nblocks",
       "S5\nCostly\nmonitoring", "S6\nShared bias\n(misperc.)", "S6b\nShared bias\n(known)",
       "S7\nEscalation-\nprone forum", "S8\nDecision\nlag", "S9\nLarge blocks\n+ lag"]
series = [("QV-Q", C1, "Verification at gates (QV − Q)"),
          ("E-QV", C2, "Triggered timing, given verification (E − QV)"),
          ("GA-Q", C3, "Aligning gates with renewals (GA − Q)"),
          ("GAL-Q", C4, "Aligning decision effects with renewals (GA$_\\ell$ − Q)")]
fig, ax = plt.subplots(figsize=(6.9, 3.4))
x = np.arange(len(sc)); w = 0.2
for k, (c, col, name) in enumerate(series):
    d = p[p.contrast == c].set_index("scenario").loc[sc]
    off = (k - 1.5) * (w + 0.005)
    ax.bar(x + off, d["diff"], w, color=col, label=name, zorder=3)
    ax.errorbar(x + off, d["diff"], yerr=[d["diff"] - d["lo"], d["hi"] - d["diff"]], fmt="none",
                ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=4)
ax.axhline(0, color=INK2, lw=0.8, zorder=2)
ax.set_xticks(x, lab, fontsize=6.2)
ax.set_ylabel("Paired difference in mean\nnet value (value units)")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(F / "fig2_decomposition.pdf"); plt.close(fig)

# ---- Figure 3: threshold sensitivity
s = pd.read_csv(R / "threshold_sweep.csv")
names = list(dict.fromkeys(s.scenario))
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
for ax, nm in zip(axes, names):
    d = s[s.scenario == nm]
    for pol, lbl, col in [("Q", "Q", C1), ("U", "U", C2), ("E", "E", C3), ("E_m", "E$_m$", C4)]:
        dd = d[d.policy == pol]
        ax.plot(dd.threshold, dd.mean_net_value, color=col, lw=1.6, label=lbl)
    ax.set_title(nm, fontsize=8, color=INK); ax.set_xlabel("Stopping threshold (belief)")
    ax.set_xlim(0, 0.82); ax.yaxis.grid(True, color=GRID, lw=0.6)
axes[0].set_ylabel("Mean net value")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig3_threshold.pdf"); plt.close(fig)

# ---- Figure 4: scenario discovery (own-model thresholds)
x = pd.read_csv(R / "exploratory_futures.csv")
gate = x[["F_own", "Q_own", "QV_own", "GA_own", "GAV_own", "GAL_own"]].max(axis=1)
x["trig"] = x[["U_own", "E_own"]].max(axis=1) > gate
x["verpay"] = x.QV_own > x.Q_own
x["alignpay"] = (x.GA_own - x.Q_own) > 0.05
x["alignlpay"] = (x.GAL_own - x.Q_own) > 0.05
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
fig.subplots_adjust(wspace=0.45, hspace=0.6)
axes = axes.ravel()
mb = pd.cut(x.monitor_fee, [0, .05, .10, .15, .20, .30, .50], include_lowest=True)
g1 = x.groupby([x.block, mb], observed=False).trig.mean().unstack()
ab = pd.cut(x.ver_adv, [-.05, .05, .15, .25], include_lowest=True)
fb = pd.cut(x.ver_fee, [.1, .4, .7, 1.0], include_lowest=True)
g2 = x.groupby([fb, ab], observed=False).verpay.mean().unstack()
g3 = x.groupby([x.block, x.lag], observed=False).alignpay.mean().unstack()
g4 = x.groupby([x.block, x.lag], observed=False).alignlpay.mean().unstack()
panels = [
    (axes[0], g1, "Monitoring fee (× operating cost)", "Commitment block",
     ["0–0.05", "0.05–\n0.10", "0.10–\n0.15", "0.15–\n0.20", "0.20–\n0.30", "0.30–\n0.50"], [str(b) for b in g1.index], "(a) Event-triggered review is best"),
    (axes[1], g2, "Verification accuracy advantage", "Verification fee",
     ["−0.05–0.05", "0.05–0.15", "0.15–0.25"], ["0.1–0.4", "0.4–0.7", "0.7–1.0"], "(b) Mandatory verification improves gates"),
    (axes[2], g3, "Decision lag (periods)", "Commitment block",
     ["0", "1", "2"], [str(b) for b in g3.index], "(c) GA beats quarterly gates"),
    (axes[3], g4, "Decision lag (periods)", "Commitment block",
     ["0", "1", "2"], [str(b) for b in g4.index], "(d) GA$_\\ell$ beats quarterly gates"),
]
for ax, g, xl, yl, xt, yt, ttl in panels:
    im = ax.imshow(g.values * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto", origin="lower")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=(7 if g.shape[1] > 4 else 8), color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(xt)), xt, fontsize=7)
    ax.set_yticks(range(len(yt)), yt, fontsize=7.5)
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.set_title(ttl, fontsize=8.5, color=INK)
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.set_label("Share of sampled futures (%)", fontsize=8); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
fig.savefig(F / "fig4_exploratory.pdf", bbox_inches="tight"); plt.close(fig)



# ---- Figure: misjudged termination lag (Experiment A)
lu = pd.read_csv(R / "lagunc_results.csv")
for c in ("GA", "GAL", "GALp"):
    lu[c + "_d"] = lu[c] - lu.Q
groups = [(1, 0, "Under-\nestimated\nby 1"), (1, 1, "Under-\nestimated\nby 1, ±1 noise"),
          (0, 0, "Correct"), (0, 1, "Correct,\n±1 noise"),
          (-1, 0, "Over-\nestimated\nby 1"), (-1, 1, "Over-\nestimated\nby 1, ±1 noise")]
ser = [("GA_d", C3, "Renewal-dated gates (GA − Q)"), ("GAL_d", C4, "Lag-aware gates (GA$_\\ell$ − Q)"),
       ("GALp_d", C1, "Lag-aware gates with one-period margin (GA$_\\ell^{+}$ − Q)")]
fig, ax = plt.subplots(figsize=(6.9, 3.3))
x = np.arange(len(groups)); w = 0.26
for k, (c, col, name) in enumerate(ser):
    m, lo, hi = [], [], []
    for bias, spread, _ in groups:
        g = lu[(lu.bias == bias) & (lu.spread == spread)][c]
        m.append(g.mean()); lo.append(g.min()); hi.append(g.max())
    m, lo, hi = map(np.array, (m, lo, hi))
    off = (k - 1) * (w + 0.01)
    ax.bar(x + off, m, w, color=col, label=name, zorder=3)
    ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=4)
ax.axhline(0, color=INK2, lw=0.8, zorder=2)
ax.set_xticks(x, [g[2] for g in groups], fontsize=6.6)
ax.set_xlabel("Realised execution lag relative to the lag planned for", fontsize=7.5)
ax.set_ylabel("Difference from quarterly gates\n(value units)")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(F / "fig_lagunc.pdf"); plt.close(fig)
print("figures written")
