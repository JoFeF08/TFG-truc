#!/bin/bash
set -e

PESOS_COS=${1:?"Us: run_f4_ablacio_48M.sh <pesos_cos.pth> [steps]"}
STEPS=${2:-48000000}

PESOS_COS="$(realpath "$PESOS_COS")"
SAVE_DIR="$(pwd)/TFG_Doc/notebooks/4_memoria/resultats/ppo_ablacio_pool_48M"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "=== F4-ablació 48M (baseline per F5) ==="
echo "Pesos COS : $PESOS_COS"
echo "Steps     : $STEPS"
echo "Save dir  : $SAVE_DIR"
echo ""

python3 RL/entrenament/entrenamentsComparatius/fase4/entrenament_f4_ablacio_48M.py \
    --pesos_cos   "$PESOS_COS" \
    --steps       "$STEPS"     \
    --save_dir    "$SAVE_DIR"  \
    --num_envs    32           \
    --n_partides  1

echo ""
echo "=== F4-ablació 48M completat ==="
