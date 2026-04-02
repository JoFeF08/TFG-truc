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
PRETRAIN_SCRIPT="RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py"
REGISTRES_DIR="RL/entrenament/entrenamentEstatTruc/registres"
COS_WEIGHTS_DEFAULT="${REGISTRES_DIR}/22_03_26_a_les_0118/models/best_pesos_cos_truc.pth"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
OUT_BASE="TFG_Doc/notebooks/2_comparacio_cos/resultats_fase2_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_temps.txt"

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$OUT_BASE"

# Si no hi ha pesos pre-entrenats del cos, llençar el pre-entrenament SL
COS_WEIGHTS="$COS_WEIGHTS_DEFAULT"
if [ ! -f "$COS_WEIGHTS" ]; then
    echo ""
    echo "=========================================="
    echo ">> No s'han trobat pesos pre-entrenats del cos:"
    echo "   $COS_WEIGHTS"
    echo ">> Iniciant pre-entrenament supervisat del cos..."
    echo "=========================================="
    python3 "$PRETRAIN_SCRIPT"
    # Agafar el model més recent generat pel pre-entrenament
    LATEST_RUN=$(ls -dt "${REGISTRES_DIR}"/*/  2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No s'ha trobat cap registre de pre-entrenament a ${REGISTRES_DIR}" >&2
        exit 1
    fi
    COS_WEIGHTS="${LATEST_RUN}models/best_pesos_cos_truc.pth"
    if [ ! -f "$COS_WEIGHTS" ]; then
        echo "ERROR: Pre-entrenament completat però no s'ha trobat el fitxer de pesos: $COS_WEIGHTS" >&2
        exit 1
    fi
    echo ">> Pre-entrenament completat. Pesos: $COS_WEIGHTS"
fi

echo "Fase 2: Comparació Cos - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "Cos pre-entrenat: $COS_WEIGHTS" >> "$RESUM"
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
    python3 "$SCRIPT" \
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
run_and_time "frozen"   "frozen"   "--cos_weights \"$COS_WEIGHTS\""
run_and_time "finetune" "finetune" "--cos_weights \"$COS_WEIGHTS\""

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "=========================================="
echo ">> Tots els experiments completats!"
echo ">> Resultats a: ${OUT_BASE}"
echo "=========================================="
cat "$RESUM"
