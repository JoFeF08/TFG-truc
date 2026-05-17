#!/bin/bash
# =============================================================================
# Fase 1 (temps): Comparació DQN RLCard / NFSP RLCard / DQN SB3 / PPO SB3
#
# Ús:  bash RL/entrenament/entrenamentsComparatius/fase1/run_fase1_temps.sh [MAX_SECONDS]
# Per defecte: 14400s (4 hores) per agent
# =============================================================================

set -e

MAX_SECONDS=${1:-14400}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase1/entrenament_comparatiu.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/1_comparacio_inicial/resultats_fase1_temps_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_temps.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 1 (temps): DQN RLCard / NFSP RLCard / DQN SB3 / PPO SB3 - $(date)" > "$RESUM"
echo "Temps màxim per agent: ${MAX_SECONDS}s ($(echo "scale=2; $MAX_SECONDS/3600" | bc)h)" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_and_time() {
    local name=$1
    local agent=$2
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name]"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)
    python3 "$SCRIPT" \
        --agent "$agent" \
        --max_time_seconds "$MAX_SECONDS" \
        --total_timesteps 999000000 \
        --save_dir "$out_dir"
    local END=$(date +%s)
    local DURATION=$((END - START))
    local HORES=$((DURATION / 3600))
    local MINUTS=$(( (DURATION % 3600) / 60 ))
    local SEGONS=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"
    echo ">> ${name} completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
}

run_and_time "dqn_rlcard" "dqn"
run_and_time "nfsp_rlcard" "nfsp"
run_and_time "dqn_sb3"    "dqn_sb3"
run_and_time "ppo_sb3"    "ppo"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
