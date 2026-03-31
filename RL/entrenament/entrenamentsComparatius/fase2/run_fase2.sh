#!/bin/bash
# =============================================================================
# Fase 2: Comparació de 3 modes d'entrenament PPO amb Cos (CosMultiInput)
#   - scratch:  cos aleatori, tot entrena des de zero
#   - frozen:   cos SL pre-entrenat, congelat
#   - finetune: cos SL pre-entrenat, descongelat al 15%
#
# Ús:  bash RL/entrenament/entrenamentsComparatius/fase2/run_fase2.sh [TIMESTEPS]
# Per defecte: 24M timesteps per mode (~3.5h total amb GPU)
# =============================================================================

set -e

TIMESTEPS=${1:-24000000}
SCRIPT="RL/entrenament/entrenamentsComparatius/fase2/entrenament_fase2_cos.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="RL/notebooks/finals/2_comparacio_cos/resultats_fase2_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_temps.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

echo "Fase 2: Comparació Cos - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_and_time() {
    local name=$1
    local mode=$2
    local extra_args=$3
    local out_dir="${OUT_BASE}/${name}"

    echo ""
    echo "=========================================="
    echo ">> Iniciant [$name] mode=$mode"
    echo "=========================================="

    mkdir -p "$out_dir"

    local START=$(date +%s)
    python "$SCRIPT" \
        --mode "$mode" \
        --total_timesteps "$TIMESTEPS" \
        --save_dir "$out_dir" \
        $extra_args
    local END=$(date +%s)
    local DURATION=$((END - START))
    local HORES=$((DURATION / 3600))
    local MINUTS=$(( (DURATION % 3600) / 60 ))
    local SEGONS=$((DURATION % 60))

    echo "${name}: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"
    echo ">> ${name} completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
}

# Executar els 3 modes
run_and_time "scratch"  "scratch"  "--cos_weights none"
run_and_time "frozen"   "frozen"   ""
run_and_time "finetune" "finetune" ""

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
