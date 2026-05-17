#!/bin/bash
# =============================================================================
# Fase 2 (curriculum): 4 agents × (12M mans + 12M partides finetune)
#
# Ús:  bash RL/entrenament/entrenamentsComparatius/fase2/run_fase2_curriculum.sh [MANS] [PARTIDES]
# Per defecte: 12M mans + 12M partides per agent
# =============================================================================

set -e

MANS=${1:-12000000}
PARTIDES=${2:-12000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase2/entrenament_fase2_curriculum.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/2_curriculum_learning/resultats_fase2_curriculum_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_curriculum.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 2 (curriculum): DQN RLCard / NFSP RLCard / DQN SB3 / PPO SB3 - $(date)" > "$RESUM"
echo "Mans: ${MANS} steps | Partides finetune: ${PARTIDES} steps" >> "$RESUM"
echo "Mode: curriculum (mans → partides)" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_agent() {
    local name=$1
    local agent=$2
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name] — mode curriculum"
    echo ">> Fase A: ${MANS} mans | Fase B: ${PARTIDES} partides"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)
    python3 "$SCRIPT" \
        --agent "$agent" \
        --mode curriculum \
        --mans_steps "$MANS" \
        --partides_steps "$PARTIDES" \
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
echo ">> Tots els experiments de curriculum completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
