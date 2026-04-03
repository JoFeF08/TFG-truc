#!/bin/bash
# run_ppo_1env.sh — Executa PPO amb 1 env

TIMESTEPS=${1:-8000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase1/entrenament_comparatiu.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="RL/notebooks/finals/1_comparacio_inicial/resultats_ppo_1env_${TIMESTAMP}"
OUT_DIR="${OUT_BASE}/ppo"
RESUM="${OUT_BASE}/resum_temps.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"
echo "PPO 1env - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

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

echo "ppo: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ">> PPO 1env completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
echo ">> Resultats a: ${OUT_BASE}"
