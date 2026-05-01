#!/bin/bash
set -e

PESOS_COS=${1:?"Us: run_fase5.sh <pesos_cos.pth> [steps]"}
STEPS=${2:-12000000}

F4_MODEL="TFG_Doc/notebooks/4_memoria/resultats/ppo_ablacio_pool/best.zip"
SAVE_DIR="TFG_Doc/notebooks/5_selfplay/resultats/ppo_selfplay_pool"

echo "=== Fase 5: Self-Play Mixt ==="
echo "Pesos COS   : $PESOS_COS"
echo "Model inici : $F4_MODEL"
echo "Steps       : $STEPS"
echo "Save dir    : $SAVE_DIR"
echo ""

python RL/entrenament/entrenamentsComparatius/fase5/entrenament_fase5.py \
    --pesos_cos     "$PESOS_COS"  \
    --model_inicial "$F4_MODEL"   \
    --steps         "$STEPS"      \
    --save_dir      "$SAVE_DIR"   \
    --num_envs      32            \
    --n_partides    5

echo ""
echo "=== Fase 5 completada ==="
