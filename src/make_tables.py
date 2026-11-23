"""Generates LaTeX tables directly from result CSVs (model v4; no hand-typed numbers)."""
from pathlib import Path
import json
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
T = Path(__file__).resolve().parents[1] / "tables"; T.mkdir(exist_ok=True)
core = pd.read_csv(R / "core_results.csv")
con = pd.read_csv(R / "core_paired_contrasts.csv")
scen = list(dict.fromkeys(core.scenario))
sid = {s: s.split(" ", 1)[0] for s in scen}
sname = {s: s.split(" ", 1)[1] for s in scen}
POL = ["F", "Q", "QV", "GA", "GAV", "GAL", "U", "E", "E_m"]
PL = {"E_m": r"E$_m$", "GAL": r"GA$_\ell$"}
NC = len(POL) + 3


def f2(v):
    return f"{v:.2f}"


# --- Table 3: net values, two panels
def core_table(panels, fname):
  lines = [r"\begin{tabular}{@{}l" + "r" * (NC - 1) + "@{}}", r"\toprule",
           "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" & B & BV \\", r"\midrule"]
  for regime, title in panels:
    lines.append(r"\multicolumn{" + str(NC) + r"}{@{}l}{\textit{" + title + r"}} \\")
    for s in scen:
        d = core[core.scenario == s]
        t = d[d.regime == regime].set_index("policy")
        vals = [t.loc[p, "mean_net_value"] for p in POL]
        best = max(vals)
        cells = [(r"\textbf{" + f2(v) + "}") if abs(v - best) < 5e-3 else f2(v) for v in vals]
        b = d[(d.regime == "benchmark") & (d.policy == "B")].mean_net_value
        bv = d[(d.regime == "benchmark") & (d.policy == "BV")].mean_net_value
        cells += [f2(b.iloc[0]), f2(bv.iloc[0]) if len(bv) else "--"]
        lines.append(f"{sid[s]} {sname[s]} & " + " & ".join(cells) + r" \\")
    if regime != panels[-1][0]:
        lines.append(r"\addlinespace[3pt]")
  lines += [r"\bottomrule", r"\end{tabular}"]
  (T / fname).write_text("\n".join(lines))


core_table((("own_model", "Panel A. Own-model thresholds computed over each policy's own decision schedule"),
            ("tuned", "Panel B. Thresholds tuned on training runs")), "tab_core.tex")
core_table((("own_v3", "Own-model thresholds under the v3 rule (continuous-review value function for every policy)"),), "stab_v3.tex")

# --- Table 4: contrasts, two panels
CS = ["U-Q", "QV-Q", "E-QV", "E-E_m", "GA-Q", "GAL-Q"]
CL = {"U-Q": "U$-$Q", "QV-Q": "QV$-$Q", "E-QV": "E$-$QV", "E-E_m": "E$-$E$_m$", "GA-Q": "GA$-$Q", "GAL-Q": r"GA$_\ell-$Q"}
def contrast_table(regime, fname, ci=True):
    lines = [r"\begin{tabular}{@{}l" + "r" * len(CS) + "@{}}", r"\toprule",
             "Scenario & " + " & ".join(CL[c] for c in CS) + r" \\", r"\midrule"]
    for s in scen:
        d = con[(con.scenario == s) & (con.regime == regime)].set_index("contrast")
        if ci:
            lines.append(f"{sid[s]} & " + " & ".join(f"{d.loc[c,'diff']:+.2f} [{d.loc[c,'lo']:+.2f}, {d.loc[c,'hi']:+.2f}]" for c in CS) + r" \\")
        else:
            lines.append(f"{sid[s]} {sname[s]} & " + " & ".join(f"{d.loc[c,'diff']:+.2f}" for c in CS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / fname).write_text("\n".join(lines))
    return float(max(((con[(con.regime == regime) & con.contrast.isin(CS)].hi - con[(con.regime == regime) & con.contrast.isin(CS)].lo) / 2).max(), 0))
hw = contrast_table("own_model", "tab_contrasts.tex", ci=False)
(T / "tab_contrasts_hw.tex").write_text(f"{hw:.2f}")
contrast_table("tuned", "stab_contrasts_tuned.tex")

# --- Table 5: regret, all futures and futures with b in {2,4,6}
rg = pd.read_csv(R / "ensemble_regret.csv")
RP = ["F", "Q", "QV", "GA", "GAV", "GAL", "U", "E"]
def cell(v, m): return (r"\textbf{" + f2(v) + "}") if abs(v - m) < 1e-9 else f2(v)
def regret_table(regimes, fname):
    lines = [r"\begin{tabular}{@{}l" + "rrr" * len(regimes) + "@{}}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{" + t + "}" for _, t in regimes) + r" \\",
             "".join(r"\cmidrule(lr){" + f"{2+3*k}-{4+3*k}" + "}" for k in range(len(regimes))),
             "Policy" + r" & Best (\%) & Mean & Max." * len(regimes) + r" \\", r"\midrule"]
    for sub, title in (("all", "All 1,500 futures"), ("b246", "Futures in which GA differs from Q by construction ($b\\in\\{2,4,6\\}$)")):
        n = int(rg[rg.subset == sub].n.iloc[0])
        lines.append(r"\multicolumn{" + str(1 + 3 * len(regimes)) + r"}{@{}l}{\textit{" + title + f" ($n={n:,}$)" + r"}} \\")
        tabs = [rg[(rg.subset == sub) & (rg.regime == reg)].set_index("policy") for reg, _ in regimes]
        for p in RP:
            cells = []
            for t in tabs:
                cells += [f"{t.loc[p, 'best_pct']:.1f}", cell(t.loc[p, "mean_regret"], t.mean_regret.min()),
                          cell(t.loc[p, "max_regret"], t.max_regret.min())]
            lines.append(PL.get(p, p) + " & " + " & ".join(cells) + r" \\")
        if sub == "all":
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / fname).write_text("\n".join(lines))
regret_table((("own-model (policy-specific)", "Own-model thresholds"), ("tuned thresholds", "Tuned thresholds")), "tab_regret.tex")
regret_table((("own-model (v3 rule)", "Continuous-review rule"),), "stab_regret_v3.tex")

# --- Table 6: normalised permutation importance
im = pd.read_csv(R / "ensemble_importance.csv")
TG = ["timing", "verif", "alignv", "alignl"]
piv = im.pivot(index="factor", columns="target", values="share")[TG]
piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
r2 = im.groupby("target").oob_r2.first()
lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Uncertain factor & Timing (E$-$QV) & Verification (QV$-$Q) & Alignment (GA$-$Q) & Lag-aware (GA$_\ell-$Q) \\", r"\midrule"]
for f, row in piv.iterrows():
    lines.append(f"{f} & " + " & ".join(f"{100*row[t]:.0f}" for t in TG) + r" \\")
lines += [r"\midrule", "Out-of-bag $R^2$ & " + " & ".join(f"{r2[t]:.2f}" for t in TG) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
(T / "tab_importance.tex").write_text("\n".join(lines))

# --- Table 5 (new): alignment under the three own-model rules, with fee/timing split
summ = json.load(open(R / "ensemble_summary.json"))
def cv(sc, reg, c):
    d = con[(con.scenario.str.startswith(sc + " ")) & (con.regime == reg)].set_index("contrast").loc[c]
    return d
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Own-model thresholds} & & \multicolumn{2}{c}{Split of (b)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){7-8}",
         r"Setting & Contrast & (a) v3 rule & (b) Policy-specific & (c) (b) + non-renewal & Tuned & Review fees & Timing \\", r"\midrule"]
for sc, lab in (("S4", "S4: $b=4$, $\\ell=0$"), ("S9", "S9: $b=4$, $\\ell=1$"), ("S8", "S8: $b=1$, $\\ell=2$")):
    for c, cl in (("GA-Q", "GA$-$Q"), ("GAL-Q", r"GA$_\ell-$Q")):
        if sc == "S4" and c == "GAL-Q":
            continue
        vals = [cv(sc, r, c)["diff"] for r in ("own_v3", "own_model", "own_nr", "tuned")]
        fee, tim = cv(sc, "own_model", c + "|fee")["diff"], cv(sc, "own_model", c + "|timing")["diff"]
        lines.append(f"{lab if c == 'GA-Q' else ''} & {cl} & " + " & ".join(f"{v:+.2f}" for v in vals) + f" & {fee:+.2f} & {tim:+.2f}" + r" \\")
a = summ["align_b246"]
lines.append(r"\addlinespace[2pt]")
lines.append(f"Ensemble, $b\\in\\{{2,4,6\\}}$ ($n={a['n']}$) & GA$-$Q & {summ['GA_minus_Q_b246_v3']:+.2f} & {summ['GA_minus_Q_b246_policy']:+.2f} & {summ['GA_minus_Q_b246_policy']:+.2f} & {summ['GA_minus_Q_b246_tuned']:+.2f} & {a['mean_fee_part']:+.2f} & {a['mean_timing_part']:+.2f}" + r" \\")
lines.append(f" & GA$_\\ell-$Q & {summ['GAL_minus_Q_b246_v3']:+.2f} & {summ['GAL_minus_Q_b246_policy']:+.2f} & {summ['GAL_minus_Q_b246_policy']:+.2f} & {summ['GAL_minus_Q_b246_tuned']:+.2f} & {a['l_mean_fee_part']:+.2f} & {a['l_mean_timing_part']:+.2f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "tab_align.tex").write_text("\n".join(lines))

# --- Supplementary: continuation-bias sweep
bs = pd.read_csv(R / "beta_sweep.csv")
lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Loss vs.\ $\beta=0$} & \multicolumn{2}{c}{Ratio} & \multicolumn{2}{c}{Tuned: extra loss} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"$\beta$ & Q & QV & U & E & QV/Q & E/U & QV$-$Q & E$-$U \\", r"\midrule"]
o = bs[bs.regime == "own_model"].pivot(index="beta", columns="policy", values="loss")
tu = bs[bs.regime == "tuned"].pivot(index="beta", columns="policy", values="loss")
for b in o.index[1:]:
    r = o.loc[b]; t = tu.loc[b]
    lines.append(f"{b:.1f} & {r.Q:.2f} & {r.QV:.2f} & {r.U:.2f} & {r.E:.2f} & {r.QV/r.Q:.1f} & {r.E/r.U:.1f} & {t.QV-t.Q:+.2f} & {t.E-t.U:+.2f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_beta.tex").write_text("\n".join(lines))

# --- Supplementary: serially correlated signals
st = pd.read_csv(R / "structural_contrasts.csv")
SC = ["U-Q", "QV-Q", "E-QV", "E-E_m", "GA-Q"]
lines = [r"\begin{tabular}{@{}ll" + "r" * len(SC) + "@{}}", r"\toprule",
         r"Scenario & Signals & " + " & ".join(CL[c] for c in SC) + r" \\", r"\midrule"]
for sc in dict.fromkeys(st.scenario):
    for k, kap in enumerate((0.0, 0.3)):
        d = st[(st.scenario == sc) & (st.persist == kap)].set_index("contrast")
        lines.append(f"{sid[sc] if k == 0 else ''} & {'independent' if kap == 0 else 'persistence 0.3'} & " + " & ".join(f"{d.loc[c, 'diff']:+.2f}" for c in SC) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_struct.tex").write_text("\n".join(lines))

# --- Supplementary tables
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Scenario & Policy & Viable stopped (\%) & Non-viable stopped (\%) & Gov.\ overhead & Stranded prepaid & Verifications \\", r"\midrule"]
for s in scen:
    d = core[(core.scenario == s) & (core.regime == "own_model")].set_index("policy")
    for i, p in enumerate(["Q", "QV", "GA", "GAL", "U", "E"]):
        r = d.loc[p]
        lines.append(f"{sid[s] if i == 0 else ''} & {PL.get(p, p)} & {r.premature_stop_pct:.1f} & {r.nonviable_termination_pct:.1f} & "
                     f"{r.mean_gov_overhead:.2f} & {r.mean_stranded_prepaid:.2f} & {r.mean_verifications:.2f}" + r" \\")
    if s != scen[-1]:
        lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_secondary.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}l" + "r" * len(POL) + "@{}}", r"\toprule", "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" \\", r"\midrule"]
for s in scen:
    t = core[(core.scenario == s) & (core.regime == "fixed_0.18")].set_index("policy")
    lines.append(f"{sid[s]} & " + " & ".join(f2(t.loc[p, 'mean_net_value']) for p in POL) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_fixed.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}l" + "r" * len(POL) + "@{}}", r"\toprule", "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" \\", r"\midrule"]
for s in scen:
    t = core[(core.scenario == s) & (core.regime == "tuned")].set_index("policy")
    lines.append(f"{sid[s]} & " + " & ".join(f"{float(t.loc[p, 'threshold']):.2f}" for p in POL) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_thresholds.tex").write_text("\n".join(lines))

CS2 = ["GAV-QV", "GAV-Q", "E-GAV", "E-Q"]
lines = [r"\begin{tabular}{@{}l" + "c" * len(CS2) + "@{}}", r"\toprule",
         "Scenario & " + " & ".join(c.replace("-", "$-$") for c in CS2) + r" \\", r"\midrule"]
for regime, title in (("own_model", "Own-model thresholds"), ("tuned", "Tuned thresholds")):
    lines.append(r"\multicolumn{5}{@{}l}{\textit{" + title + r"}} \\")
    for s in scen:
        d = con[(con.scenario == s) & (con.regime == regime)].set_index("contrast")
        lines.append(f"{sid[s]} & " + " & ".join(f"{d.loc[c,'diff']:+.2f} [{d.loc[c,'lo']:+.2f}, {d.loc[c,'hi']:+.2f}]" for c in CS2) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_contrasts2.tex").write_text("\n".join(lines))

trees = json.load(open(R / "ensemble_trees.json"))
txt = []
for k, v in trees.items():
    txt.append(r"\paragraph{" + k.replace("_", " ") + f" (base rate {100*v['base_rate']:.0f}\\%, accuracy {100*v['accuracy']:.0f}\\%)" + "}")
    txt.append(r"\begin{verbatim}" + "\n" + v["text"] + r"\end{verbatim}")
(T / "stab_trees.tex").write_text("\n".join(txt))
print("tables written")
