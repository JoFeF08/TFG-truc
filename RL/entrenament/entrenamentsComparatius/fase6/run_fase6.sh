#!/bin/bash
set -e
PESOS_COS=${1:?"Ús: run_fase6.sh <pesos_cos.pth> [steps] [num_envs]"}
STEPS=${2:-48000000}
NUM_ENVS=${3:-32}
F5_MODEL="TFG_Doc/notebooks/5_selfplay/resultats/ppo_selfplay_pool_9snaps/best_robust.zip"
SAVE_DIR="TFG_Doc/notebooks/6_nfsp/resultats/ppo_nfsp"

echo "=== Fase 6: NFSP complet (PPO + SL average policy) ==="
echo "Pesos COS  : $PESOS_COS"
echo "Model inic : $F5_MODEL"
echo "Steps      : $STEPS"
echo "Num envs   : $NUM_ENVS"
echo "Save dir   : $SAVE_DIR"

python3 RL/entrenament/entrenamentsComparatius/fase6/entrenament_fase6.py \
    --pesos_cos     "$PESOS_COS"  \
    --model_inicial "$F5_MODEL"   \
    --steps         "$STEPS"      \
    --save_dir      "$SAVE_DIR"   \
    --num_envs      "$NUM_ENVS"   \
    --n_partides    1             \
    --eta           0.5           \
    --reservoir_cap 200000        \
    --sl_lr         5e-4          \
    --sl_every      50000

echo "=== Fase 6 completada ==="
