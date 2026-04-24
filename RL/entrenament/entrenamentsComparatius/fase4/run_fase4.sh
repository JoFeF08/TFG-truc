#!/bin/bash
# =============================================================================
# Fase 4: Memòria d'Oponent (RecurrentPPO + Sessions Multi-Partida)
# 2 runs: ablacio (PPO + pool divers) i complet (PPO-LSTM + pool divers).
#
# Ús:
#   bash RL/entrenament/entrenamentsComparatius/fase4/run_fase4.sh <PESOS_COS> [STEPS]
#
# Arguments:
#   PESOS_COS  — Ruta al best_pesos_cos_truc.pth (produït per preentrenar_cos.py).
#   STEPS      — Steps per run (defecte 12000000).
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
STEPS=${2:-12000000}

if [[ ! -f "$PESOS_COS" ]]; then
    echo "ERROR: No existeix el fitxer de pesos: $PESOS_COS"
    exit 1
fi

SCRIPT="RL/entrenament/entrenamentsComparatius/fase4/entrenament_fase4.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/4_memoria/resultats"
RESUM="${OUT_BASE}/resum_fase4_${TIMESTAMP}.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 4: Memòria d'Oponent - $(date)" > "$RESUM"
echo "Pesos cos: ${PESOS_COS}" >> "$RESUM"
echo "Steps per run: ${STEPS}" >> "$RESUM"
echo "Variants: ablacio (PPO) + complet (PPO-LSTM) = 2 runs" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

cleanup_zombies() {
    pkill -9 -f "entrenament_fase4.py" 2>/dev/null || true
    pkill -9 -f "multiprocessing.forkserver" 2>/dev/null || true
    sleep 2
}

run_variant() {
    local variant=$1
    local name="ppo_${variant}"
    local out_dir="${OUT_BASE}/ppo_${variant}_pool"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name]"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)

    python3 "$SCRIPT" \
        --variant "$variant" \
        --pesos_cos "$PESOS_COS" \
        --steps "$STEPS" \
        --save_dir "$out_dir"

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

run_variant "ablacio"
run_variant "complet"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments de Fase 4 completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
