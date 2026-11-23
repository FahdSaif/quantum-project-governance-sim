# Changelog

## v4.0.0 (2026-11-23)

Changes made after an internal pre-submission senior review. All results, tables and figures were regenerated from code.

### Model (`governance_sim.py`)
- **Policy-specific own-model thresholds** (`policy_thresholds`): backward induction under the assumed reliability over each policy's *own* decision schedule and decision lag. Between decision points the forum must continue and pay any renewal that falls due. This is now the primary specification. The v3 rule (`own_model_thresholds`, one continuous-review threshold path for all policies) is kept for comparison.
- A stop that could not take effect before completion, or that would avoid no further renewal, is never taken (v3 broke this tie towards stopping at beliefs below 0.00025).
- **Do-not-renew rule** (`Env.nonrenewal`): a stop decided inside a paid block takes effect at the end of that block. Leaves decisions and net value unchanged (tested); removes stranded prepaid cost from the accounts.
- **Lag-aware aligned gates** `GAL` (`renewal_aligned_gates`): GA gates moved ℓ periods earlier.
- **Serially correlated routine signals** (`Env.persist`): each signal repeats the previous one with probability `persist`; marginal accuracy unchanged. The extra random stream is drawn last, so all v3 draws are unchanged.

### Experiments and analysis
- Study 1: scenario S9 (four-period blocks, one-period lag); regimes `own_model` (policy-specific), `own_nr` (+ do-not-renew), `own_v3` (continuous-review rule), `tuned`, `fixed_0.18`; contrasts GAL−Q, GAL−GA, E−U; review-fee / timing split of GA−Q and GAL−Q.
- `beta_sweep`: continuation bias 0–0.6 (S1 base), own-model and tuned thresholds.
- `structural`: serially correlated signals (persist = 0.3) for S1, S2, S4.
- Threshold sweep (Fig. 3) now uses all 20 test seeds.
- Study 2: GAL added; own-model (policy-specific), v3 rule, do-not-renew and tuned regimes; regret reported for all futures and for b ∈ {2, 4, 6}; importance for GAL−Q; importance relabelled as normalised permutation importance.
- Tests: 7 → 14.

### Reproduction check
- Under the v3 rule, Study 1 reproduces the v3 published values exactly (maximum difference 0.0).

## v3.0.0
Own-model thresholds, renewal-aligned gates (GA, GAV), known-correlation verification, BV benchmark, continuation bias, decision lag.
