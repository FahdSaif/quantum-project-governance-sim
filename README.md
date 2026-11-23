# Evidence-gated commitment in exploratory technology projects

Simulation code and outputs for the paper

> **Timing is everything: Aligning governance decisions with commitments in exploratory technology projects** (manuscript submitted).

The model compares governance policies for exploratory technology projects in which technical evidence arrives gradually, spending is committed in non-refundable blocks, and a governance forum decides whether to continue. Policies differ in **when** stop decisions can be taken (fixed, quarterly, renewal-dated, lag-aware aligned or evidence-triggered) and **whether** adverse evidence is verified first.

All data are **synthetic**. No real projects, organisations or people are represented.

## Repository structure

| Path | Contents |
|---|---|
| `src/governance_sim.py` | The model: projects, governance policies, policy-specific and continuous-review own-model thresholds, do-not-renew rule, serially correlated signals, Bayes-optimal benchmarks (B, BV) |
| `src/experiments.py` | Study 1 (ten focal scenarios), continuation-bias sweep, serial-correlation check and Study 2 (1,500 sampled futures) |
| `src/analyse_ensemble.py` | Regret, random-forest permutation importance, scenario-discovery trees |
| `src/make_tables.py` | Builds every LaTeX table in the paper from `results/` |
| `src/figures.py` | Builds Figures 2–4 from `results/` |
| `src/orig.py` | Original (v1) implementation, kept unchanged for the verification check |
| `tests/test_model.py` | 14 verification tests (parity with v1, scale invariance, benchmark dominance, policy-specific thresholds, do-not-renew equivalence, lag-aware gates, serial correlation) |
| `CHANGELOG.md` | Model and analysis changes by version |
| `results/` | Simulation outputs used in the paper |
| `figures/`, `tables/` | Figures and tables as they appear in the paper |

## Policies

| Code | Policy |
|---|---|
| F | Fixed gates after periods 6 and 9 |
| Q | Quarterly gates after periods 3, 6 and 9 |
| QV | Quarterly gates, verification of adverse evidence before stopping |
| GA | Renewal-dated gates: end of block, every b·⌈3/b⌉ periods (b=1,3: 3,6,9; b=2,4: 4,8; b=6: 6) |
| GAV | Renewal-dated gates with verification |
| GAL | Lag-aware aligned gates: GA gates moved ℓ periods earlier so that stops take effect at renewals |
| U | Evidence-triggered review, no verification |
| E | Evidence-triggered review with verification |
| E_m | As E, but verification no more accurate than routine evidence (control) |
| B, BV | Bayes-optimal benchmarks without / with costly verification |

## Reproducing the results

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
bash run_all.sh          # about 10 minutes; regenerates results/, tables/ and figures/
python -m pytest -q      # verification tests
```

Every random draw is seeded (base seed `20261122`; see `results/run_config.json`), so a rerun reproduces the published numbers exactly on the same package versions. (Random-forest importance values in `results/ensemble_importance.csv` can differ in the 16th decimal place because of parallel computation; this does not affect any reported number.)

## Licence

Code: MIT (see `LICENSE`). Results files: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the paper once published.
