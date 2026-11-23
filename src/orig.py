"""ORIGINAL (v1) implementation, kept unchanged for the verification check in the paper (Section 4.5).
Superseded by governance_sim.py; see README.

Reproducible synthetic experiment for evidence/commitment governance.
No real projects, no human data. Python 3.10+; numpy required.
Source accompanies the manuscript. Run: python quantum_project_governance_simulation.py
"""
import numpy as np
import csv
from pathlib import Path

SEED = 20261122
N = 12000
H = 12
MONTHLY_COST = 1.0
TERMINAL_VALUE = 28.0
THRESHOLD = 0.18
REVIEW_FEE = 0.35
VERIFICATION_FEE = 0.55
Q_VERIFY = 0.82
POLICIES = ("fixed_6_9", "quarterly_3_6_9", "trigger_unverified", "trigger_verified")
SCENARIOS = (
    ("calibrated_strong", .80, .80, 1, .12),
    ("overconfident_weak", .62, .80, 1, .12),
    ("calibrated_weak", .62, .62, 1, .12),
    ("large_lockin", .80, .80, 4, .12),
    ("costly_monitoring", .80, .80, 1, .45),
)

def sim_one(scenario, rng):
    name, q_true, q_assumed, commitment_block, monitoring_fee = scenario
    prior = rng.beta(3.0,3.0, size=N)
    viable = rng.random(N) < prior
    signals = rng.random((N,H)) < np.where(viable[:,None],q_true,1-q_true)
    verification = rng.random((N,H)) < np.where(viable[:,None],Q_VERIFY,1-Q_VERIFY)
    loglik = np.log(q_assumed/(1-q_assumed))
    ver_loglik = np.log(Q_VERIFY/(1-Q_VERIFY))
    rows = []
    outcomes = {}
    for policy in POLICIES:
        active = np.ones(N, dtype=bool)
        paid = np.zeros(N, dtype=float)
        review_costs = np.zeros(N, dtype=float)
        monitor_costs = np.zeros(N, dtype=float)
        prior_odds = np.log(prior/(1-prior))
        posterior_logodds = prior_odds.copy()
        stopped_at = np.full(N,H, dtype=int)
        prepaid_undelivered = np.zeros(N,dtype=float)
        for t in range(1,H+1):
            # Nonrefundable future operating blocks are contracted at period start.
            if (t-1) % commitment_block == 0:
                paid[active] += min(commitment_block,H-t+1)*MONTHLY_COST
            if policy.startswith('trigger'):
                monitor_costs[active] += monitoring_fee
            posterior_logodds[active] += np.where(signals[active,t-1],1,-1)*loglik
            posterior = 1.0/(1.0+np.exp(-np.clip(posterior_logodds,-100,100)))
            if policy == "fixed_6_9":
                eligible = t in (6,9)
            elif policy == "quarterly_3_6_9":
                eligible = t in (3,6,9)
            else:
                eligible = (t < H)
            if not eligible:
                continue
            if policy.startswith('trigger'):
                review = active & (posterior < THRESHOLD)
            else:
                review = active.copy()
            if policy == 'trigger_verified':
                review_costs[review] += VERIFICATION_FEE
                posterior_logodds[review] += np.where(verification[review,t-1],1,-1)*ver_loglik
                posterior = 1.0/(1.0+np.exp(-np.clip(posterior_logodds,-100,100)))
            elif policy.startswith('trigger'):
                review_costs[review] += REVIEW_FEE
            else:
                review_costs[review] += REVIEW_FEE
            terminate = review & (posterior < THRESHOLD)
            stopped_at[terminate] = t
            active[terminate] = False
            paid_months = min(H, ((t-1)//commitment_block+1)*commitment_block)
            prepaid_undelivered[terminate] = (paid_months-t)*MONTHLY_COST
        completed = active
        payoff = completed*viable*TERMINAL_VALUE - paid - monitor_costs - review_costs
        outcomes[policy] = payoff
        m = {
            'scenario':name, 'policy':policy,
            'N':N,'q_true':q_true,'q_assumed':q_assumed,
            'block':commitment_block, 'monitor_fee':monitoring_fee,
            'mean_net_value':float(np.mean(payoff)),
            'mean_total_spend':float(np.mean(paid+monitor_costs+review_costs)),
            'mean_gov_overhead':float(np.mean(monitor_costs+review_costs)),
            'nonviable_operating_spend':float(np.mean(paid[~viable])),
            'premature_stop_pct':float(100*np.mean(~completed[viable])),
            'nonviable_termination_pct':float(100*np.mean(~completed[~viable])),
            'mean_unrecoverable_prepaid':float(np.mean(prepaid_undelivered)),
            'completion_pct':float(100*np.mean(completed)),
            'mean_stop_month_nonviable':float(np.mean(stopped_at[(~viable) & (~completed)])) if np.any((~viable)&(~completed)) else np.nan,
        }
        rows.append(m)
    # paired differences against quarterly (controls Monte Carlo noise)
    paired = []
    reference = outcomes['quarterly_3_6_9']
    for policy in POLICIES:
        if policy == 'quarterly_3_6_9': continue
        difference = outcomes[policy]-reference
        mean=float(np.mean(difference))
        se=float(np.std(difference,ddof=1)/np.sqrt(N))
        paired.append({'scenario':name,'policy':policy,'reference':'quarterly_3_6_9',
                       'paired_diff':mean,'ci_lower':mean-1.96*se,'ci_upper':mean+1.96*se})
    return rows, paired

def main():
    rng = np.random.default_rng(SEED)
    allrows=[]; allpaired=[]
    for s in SCENARIOS:
        r,p=sim_one(s,rng)
        allrows += r; allpaired += p
    root=Path(__file__).resolve().parent
    for filename,rows in [("quantum_simulation_results.csv",allrows),("quantum_simulation_paired_differences.csv",allpaired)]:
        with (root/filename).open('w', newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    print('Seed',SEED,'N',N)
    for r in allrows:
        print(f"{r['scenario']:20s} {r['policy']:22s} net={r['mean_net_value']:7.3f} "
              f"prem={r['premature_stop_pct']:6.2f}% nonviable_stop={r['nonviable_termination_pct']:6.2f}% "
              f"overhead={r['mean_gov_overhead']:6.3f} prepaid={r['mean_unrecoverable_prepaid']:6.3f}")
    print('Paired comparisons:')
    for r in allpaired:
        print(r)

if __name__=='__main__':main()
