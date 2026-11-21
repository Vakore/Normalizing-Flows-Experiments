#!/usr/bin/env bash
# Run every experiment and regenerate all figures and tables.
#   ./run_all.sh          full run (roughly 20-45 min on a laptop CPU)
#   ./run_all.sh --quick  short run to check that everything works (~2 min)
set -euo pipefail
cd "$(dirname "$0")"

STEPS_LIN=2000; STEPS=4000; STEPS_CMP=8000; SEEDS=3
if [[ "${1:-}" == "--quick" ]]; then STEPS_LIN=200; STEPS=300; STEPS_CMP=300; SEEDS=1; fi

echo "== 0. Checking every flow layer";            python 00_check_flows.py
echo "== 1. Change of variables (1-D demo)";       python 01_change_of_variables.py
echo "== 2. Cost of log-determinants";             python 02_logdet_cost.py
echo "== 3. Linear flows";                         python 03_linear_flows.py --steps "$STEPS_LIN"
echo "== 4. Coupling & autoregressive flows";      python 04_coupling_flows.py --steps "$STEPS"
echo "== 5. Flow vs VAE vs GAN";                   python 05_compare_vae_gan.py --steps "$STEPS_CMP" --seeds "$SEEDS"
echo "Done. Figures are in ../figures, tables in ../results."
