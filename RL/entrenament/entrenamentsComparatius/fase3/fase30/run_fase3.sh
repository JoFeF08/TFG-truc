#!/bin/bash
# =============================================================================
# Fase 3: Valor del Feature Extractor COS
# 2 algorismes × 3 protocols nous (control_cos, curriculum_cos, mans_mlp) = 6 runs
#
# Execucions reutilitzades (no es tornen a córrer):
#   - control sense COS  → Fase 2 (DQN 60.5%, PPO 35%)
#   - curriculum sense COS → Fase 2 (DQN 56%, PPO 75%)
#   - mans amb COS (scratch) → Fase 3.5 ppo_sb3_scratch / dqn_sb3_scratch
#
# Ús:
#   bash RL/entrenament/entrenamentsComparatius/fase3/run_fase3.sh [STEPS]
#
# Arguments:
#   STEPS — Steps totals per run (defecte 24000000).
#            Per al protocol curriculum es divideix en 12M+12M.
# =============================================================================

set -e

STEPS=${1:-24000000}

SCRIPT="RL/entrenament/entrenamentsComparatius/fase3/fase30/entrenament_fase3.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/3_feature_extractor/resultats/resultats_fase3_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_fase3.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 3: Valor del Feature Extractor COS - $(date)" > "$RESUM"
echo "Steps per run: ${STEPS}" >> "$RESUM"
echo "6 runs nous: control_cos × 2, curriculum_cos × 2, mans_mlp × 2" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

cleanup_zombies() {
    pkill -9 -f "entrenament_fase3.py" 2>/dev/null || true
    pkill -9 -f "multiprocessing.forkserver" 2>/dev/null || true
    sleep 2
}

run_experiment() {
    local agent=$1
    local protocol=$2
    local use_cos=$3   # "cos" o "mlp"
    local name="${agent}_${protocol}_${use_cos}"
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name]"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)

    local args=(--agent "$agent" --protocol "$protocol" --steps "$STEPS" --save_dir "$out_dir")
    if [[ "$use_cos" == "cos" ]]; then
        args+=(--cos)
    fi

    python3 "$SCRIPT" "${args[@]}"

    local END=$(date +%s)
    local DURATION=$((END - START))
    local H=$((DURATION / 3600))
    local M=$(( (DURATION % 3600) / 60 ))
    local S=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${H}h ${M}m ${S}s)" >> "$RESUM"
    echo ">> ${name} completat en ${H}h ${M}m ${S}s"

    cleanup_zombies
    echo ">> Pausa de 60s (cooldown RAM/GPU)..."
    sleep 60
}

trap cleanup_zombies EXIT INT TERM

# 6 runs nous:
run_experiment "ppo_sb3" "control"    "cos"
run_experiment "dqn_sb3" "control"    "cos"
run_experiment "ppo_sb3" "curriculum" "cos"
run_experiment "dqn_sb3" "curriculum" "cos"
run_experiment "ppo_sb3" "mans"       "mlp"
run_experiment "dqn_sb3" "mans"       "mlp"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments de Fase 3 completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
