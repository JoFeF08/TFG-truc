#!/bin/bash
# run_comparatiu.sh
# Executa els tres algorismes (DQN, NFSP, PPO) en condicions idèntiques.
# Tots els resultats es guarden a un directori timestampat.
#
# Ús:
#   bash run_comparatiu.sh
#   bash run_comparatiu.sh --timesteps 6000000    # per proves ràpides

TIMESTEPS=24000000
SCRIPT="RL/entrenament/entrenamentsComparatius/entrenament_comparatiu.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")
BASE_DIR="resultats_comparativa_${TIMESTAMP}"

# Opcions de línia de comandes
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift ;;
        *) echo "Opció desconeguda: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$BASE_DIR"
RESUM="${BASE_DIR}/resum_temps.txt"
echo "Comparativa algorismes RL - $(date)" > "$RESUM"
echo "Timesteps per algorisme: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

export PYTHONPATH="$(pwd):$PYTHONPATH"

run_and_time() {
    local AGENT=$1
    local OUT_DIR="${BASE_DIR}/${AGENT}"

    echo "==================================================="
    echo ">> Iniciant [$AGENT] | Guardat a: $OUT_DIR"
    echo "==================================================="

    START=$(date +%s)

    python3 "$SCRIPT" \
        --agent "$AGENT" \
        --total_timesteps "$TIMESTEPS" \
        --save_dir "$OUT_DIR"

    END=$(date +%s)
    DURATION=$((END - START))
    HORES=$((DURATION / 3600))
    MINUTS=$(( (DURATION % 3600) / 60 ))
    SEGONS=$((DURATION % 60))

    echo ">> [$AGENT] completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
    echo "${AGENT}: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"

    echo "Esperant 15 segons abans del següent..."
    sleep 15
}

run_and_time "dqn"
run_and_time "nfsp"
run_and_time "ppo"

echo "-------------------------------------------" >> "$RESUM"
echo "Tots els experiments completats: $(date)" >> "$RESUM"

echo ""
echo "==================================================="
echo " TOTS ELS EXPERIMENTS COMPLETATS!"
echo " Resultats a: $BASE_DIR"
echo " Resum de temps: $RESUM"
echo "==================================================="
