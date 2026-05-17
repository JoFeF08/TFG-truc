#!/bin/bash
# =============================================================================
# Fase 2 (control): 4 agents × 24M steps en partides, sense self-play
#
# Ús:  bash RL/entrenament/entrenamentsComparatius/fase2/run_fase2_control.sh [STEPS]
# Per defecte: 24M steps per agent
# =============================================================================

set -e

STEPS=${1:-24000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase2/entrenament_fase2_curriculum.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/2_curriculum_learning/resultats_fase2_control_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_control.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 2 (control): DQN RLCard / NFSP RLCard / DQN SB3 / PPO SB3 - $(date)" > "$RESUM"
echo "Steps per agent: ${STEPS}" >> "$RESUM"
echo "Mode: control (partides directes, sense self-play)" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_agent() {
    local name=$1
    local agent=$2
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name] — mode control"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)
    python3 "$SCRIPT" \
        --agent "$agent" \
        --mode control \
        --mans_steps 0 \
        --partides_steps "$STEPS" \
        --save_dir "$out_dir"
    local END=$(date +%s)
    local DURATION=$((END - START))
    local H=$((DURATION / 3600))
    local M=$(( (DURATION % 3600) / 60 ))
    local S=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${H}h ${M}m ${S}s)" >> "$RESUM"
    echo ">> ${name} completat en ${H}h ${M}m ${S}s"
}

run_agent "dqn_rlcard"  "dqn"
run_agent "nfsp_rlcard" "nfsp"
run_agent "dqn_sb3"     "dqn_sb3"
run_agent "ppo_sb3"     "ppo"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments de control completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
