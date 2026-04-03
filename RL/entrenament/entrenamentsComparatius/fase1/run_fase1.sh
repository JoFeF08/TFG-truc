#!/bin/bash
# =============================================================================
# Fase 1: Comparació DQN / NFSP / PPO en entorn paral·lel
#
# Ús:  bash RL/entrenament/entrenamentsComparatius/fase1/run_fase1.sh [TIMESTEPS]
# Per defecte: 24M timesteps per agent
# =============================================================================

set -e

TIMESTEPS=${1:-24000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase1/entrenament_comparatiu.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/1_comparacio_inicial/resultats_comparativa_parallel_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_temps.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 1: Comparació parallel - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
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
        --total_timesteps "$TIMESTEPS" \
        --save_dir "$out_dir"
    local END=$(date +%s)
    local DURATION=$((END - START))
    local HORES=$((DURATION / 3600))
    local MINUTS=$(( (DURATION % 3600) / 60 ))
    local SEGONS=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"
    echo ">> ${name} completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
}

run_and_time "dqn"  "dqn"
run_and_time "nfsp" "nfsp"
run_and_time "ppo"  "ppo"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
