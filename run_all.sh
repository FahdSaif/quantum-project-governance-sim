#!/usr/bin/env bash
# Reproduces every result, table and figure reported in the paper.
set -euo pipefail
cd "$(dirname "$0")"
python src/experiments.py core sweep explore   # Study 1 and Study 2 (~5 min)
python src/analyse_ensemble.py                 # regret, importance, trees
python src/make_tables.py                      # tables/*.tex
python src/figures.py                          # figures/*.pdf
echo "Done. Outputs in results/, tables/ and figures/."
