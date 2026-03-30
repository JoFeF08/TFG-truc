#!/bin/bash
# run_ppo_1env.sh — Executa PPO amb 1 env i guarda el temps

TIMESTEPS=${1:-8000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/entrenament_comparatiu.py"
OUT_DIR="RL/notebooks/finals/1_comparacio_inicial/resultats_comparativa_1env_30_03_0005h/ppo"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo ">> PPO 1env | ${TIMESTEPS} steps | Guardat a: ${OUT_DIR}"

START=$(date +%s)

python3 "$SCRIPT" \
    --agent ppo \
    --total_timesteps "$TIMESTEPS" \
    --save_dir "$OUT_DIR" \
    --num_envs 1

END=$(date +%s)
DURATION=$((END - START))
HORES=$((DURATION / 3600))
MINUTS=$(( (DURATION % 3600) / 60 ))
SEGONS=$((DURATION % 60))

echo ">> PPO 1env completat en ${HORES}h ${MINUTS}m ${SEGONS}s"

RESUM="RL/notebooks/finals/1_comparacio_inicial/resultats_comparativa_1env_30_03_0005h/resum_temps.txt"
sed -i "s/^ppo:.*/ppo: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)/" "$RESUM"
echo ">> Resum actualitzat"
