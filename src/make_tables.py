"""Generates LaTeX tables directly from result CSVs (model v3; no hand-typed numbers)."""
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
POL = ["F", "Q", "QV", "GA", "GAV", "U", "E", "E_m"]
PL = {"E_m": r"E$_m$"}


def f2(v):
    return f"{v:.2f}"


# --- Table 3: net values, two panels
lines = [r"\begin{tabular}{@{}l" + "r" * 10 + "@{}}", r"\toprule",
         "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" & B & BV \\", r"\midrule"]
for regime, title in (("own_model", "Panel A. Own-model (feasible) thresholds"), ("tuned", "Panel B. Thresholds tuned on training runs")):
    lines.append(r"\multicolumn{11}{@{}l}{\textit{" + title + r"}} \\")
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
    if regime == "own_model":
        lines.append(r"\addlinespace[3pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "tab_core.tex").write_text("\n".join(lines))

# --- Table 4: contrasts, two panels
CS = ["U-Q", "QV-Q", "E-QV", "E-E_m", "GA-Q"]
CL = {"U-Q": "U$-$Q", "QV-Q": "QV$-$Q", "E-QV": "E$-$QV", "E-E_m": "E$-$E$_m$", "GA-Q": "GA$-$Q"}
lines = [r"\begin{tabular}{@{}l" + "c" * len(CS) + "@{}}", r"\toprule",
         "Scenario & " + " & ".join(CL[c] for c in CS) + r" \\", r"\midrule"]
for regime, title in (("own_model", "Panel A. Own-model (feasible) thresholds"), ("tuned", "Panel B. Tuned thresholds")):
    lines.append(r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
    for s in scen:
        d = con[(con.scenario == s) & (con.regime == regime)].set_index("contrast")
        lines.append(f"{sid[s]} & " + " & ".join(f"{d.loc[c,'diff']:+.2f} [{d.loc[c,'lo']:+.2f}, {d.loc[c,'hi']:+.2f}]" for c in CS) + r" \\")
    if regime == "own_model":
        lines.append(r"\addlinespace[3pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "tab_contrasts.tex").write_text("\n".join(lines))

# --- Table 5: regret (both regimes)
rg = pd.read_csv(R / "ensemble_regret.csv")
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Own-model thresholds} & \multicolumn{3}{c}{Tuned thresholds} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Policy & Best (\%) & Mean regret & Max.\ regret & Best (\%) & Mean regret & Max.\ regret \\", r"\midrule"]
own = rg[rg.regime.str.startswith("own")].set_index("policy"); tun = rg[rg.regime.str.startswith("tuned")].set_index("policy")
mins = {k: (own[k].min(), tun[k].min()) for k in ("mean_regret", "max_regret")}
def cell(v, m): return (r"\textbf{" + f2(v) + "}") if abs(v - m) < 1e-9 else f2(v)
for p in ["F", "Q", "QV", "GA", "GAV", "U", "E"]:
    o, t = own.loc[p], tun.loc[p]
    lines.append(f"{p} & {o.best_pct:.1f} & {cell(o.mean_regret, mins['mean_regret'][0])} & {cell(o.max_regret, mins['max_regret'][0])} & "
                 f"{t.best_pct:.1f} & {cell(t.mean_regret, mins['mean_regret'][1])} & {cell(t.max_regret, mins['max_regret'][1])}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "tab_regret.tex").write_text("\n".join(lines))

# --- Table 6: importance
im = pd.read_csv(R / "ensemble_importance.csv")
piv = im.pivot(index="factor", columns="target", values="share")[["timing", "verif", "alignv"]]
piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
r2 = im.groupby("target").oob_r2.first()
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Uncertain factor & Timing (E$-$QV) & Verification (QV$-$Q) & Alignment (GA$-$Q) \\", r"\midrule"]
for f, row in piv.iterrows():
    lines.append(f"{f} & {100*row.timing:.0f} & {100*row.verif:.0f} & {100*row.alignv:.0f}" + r" \\")
lines += [r"\midrule", f"Out-of-bag $R^2$ & {r2['timing']:.2f} & {r2['verif']:.2f} & {r2['alignv']:.2f}" + r" \\",
          r"\bottomrule", r"\end{tabular}"]
(T / "tab_importance.tex").write_text("\n".join(lines))

# --- Supplementary tables
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Scenario & Policy & Viable stopped (\%) & Non-viable stopped (\%) & Gov.\ overhead & Stranded prepaid & Verifications \\", r"\midrule"]
for s in scen:
    d = core[(core.scenario == s) & (core.regime == "own_model")].set_index("policy")
    for i, p in enumerate(["Q", "QV", "GA", "U", "E"]):
        r = d.loc[p]
        lines.append(f"{sid[s] if i == 0 else ''} & {p} & {r.premature_stop_pct:.1f} & {r.nonviable_termination_pct:.1f} & "
                     f"{r.mean_gov_overhead:.2f} & {r.mean_stranded_prepaid:.2f} & {r.mean_verifications:.2f}" + r" \\")
    if s != scen[-1]:
        lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_secondary.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}", r"\toprule", "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" \\", r"\midrule"]
for s in scen:
    t = core[(core.scenario == s) & (core.regime == "fixed_0.18")].set_index("policy")
    lines.append(f"{sid[s]} & " + " & ".join(f2(t.loc[p, 'mean_net_value']) for p in POL) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "stab_fixed.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}", r"\toprule", "Scenario & " + " & ".join(PL.get(p, p) for p in POL) + r" \\", r"\midrule"]
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
