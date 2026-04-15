#!/bin/bash
# =============================================================================
# Fase 3: Feature Extractor Preentrenat
# 2 algorismes (DQN-SB3, PPO-SB3) × 3 variants (scratch, frozen, finetune) = 6 runs
#
# Ús:
#   bash RL/entrenament/entrenamentsComparatius/fase3/run_fase3.sh <PESOS_COS> [STEPS]
#
# Arguments:
#   PESOS_COS  — Ruta al best_pesos_cos_truc.pth (produït per preentrenar_cos.py).
#                Obligatori per a les variants frozen i finetune.
#   STEPS      — Steps per run (defecte 24000000).
# =============================================================================

set -e

if [[ $# -lt 1 ]]; then
    echo "Ús: $0 <PESOS_COS> [STEPS]"
    echo ""
    echo "Exemple:"
    echo "  $0 RL/entrenament/entrenamentEstatTruc/registres/15_04_26_a_les_1030/models/best_pesos_cos_truc.pth"
    exit 1
fi

PESOS_COS="$1"
STEPS=${2:-24000000}

if [[ ! -f "$PESOS_COS" ]]; then
    echo "ERROR: No existeix el fitxer de pesos: $PESOS_COS"
    exit 1
fi

SCRIPT="RL/entrenament/entrenamentsComparatius/fase3/entrenament_fase3.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/3_feature_extractor_preentrenat/resultats_fase3_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_fase3.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 3: Feature Extractor Preentrenat - $(date)" > "$RESUM"
echo "Pesos cos: ${PESOS_COS}" >> "$RESUM"
echo "Steps per run: ${STEPS}" >> "$RESUM"
echo "2 algorismes × 3 variants (scratch, frozen, finetune) = 6 runs" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_variant() {
    local agent=$1
    local variant=$2
    local name="${agent}_${variant}"
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name]"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)

    local args=(--agent "$agent" --variant "$variant" --steps "$STEPS" --save_dir "$out_dir")
    if [[ "$variant" != "scratch" ]]; then
        args+=(--pesos_cos "$PESOS_COS")
    fi

    python3 "$SCRIPT" "${args[@]}"

    local END=$(date +%s)
    local DURATION=$((END - START))
    local H=$((DURATION / 3600))
    local M=$(( (DURATION % 3600) / 60 ))
    local S=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${H}h ${M}m ${S}s)" >> "$RESUM"
    echo ">> ${name} completat en ${H}h ${M}m ${S}s"
}

# Ordre: primer DQN (3 variants), després PPO (3 variants).
run_variant "dqn_sb3" "scratch"
run_variant "dqn_sb3" "frozen"
run_variant "dqn_sb3" "finetune"
run_variant "ppo_sb3" "scratch"
run_variant "ppo_sb3" "frozen"
run_variant "ppo_sb3" "finetune"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments de Fase 3 completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
