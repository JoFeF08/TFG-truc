#!/bin/bash
# run_parallel_16M.sh
# Executa DQN, NFSP i PPO en mode paral·lel (16/48 envs) durant 16M timesteps.
# Les dades seqüencials (1 env, 8M) ja existeixen de l'experiment anterior.
#
# Ús:
#   bash run_parallel_16M.sh                     # 16M steps (defecte)
#   bash run_parallel_16M.sh --timesteps 20000000

TIMESTEPS=16000000
SCRIPT="RL/entrenament/entrenamentsComparatius/entrenament_comparatiu.py"
TIMESTAMP=$(date +"%d_%m_%H%Mh")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift ;;
        *) echo "Opció desconeguda: $1"; exit 1 ;;
    esac
    shift
done

export PYTHONPATH="$(pwd):$PYTHONPATH"

OUT_BASE="RL/notebooks/finals/1_comparacio_inicial/resultats_comparativa_parallel_16M_${TIMESTAMP}"
RESUM="${OUT_BASE}/resum_temps.txt"

mkdir -p "$OUT_BASE"
echo "Comparativa PARAL·LEL 16M - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

run_and_time() {
    local AGENT=$1
    local OUT_DIR=$2

    echo "==================================================="
    echo ">> Iniciant [$AGENT] | ${TIMESTEPS} steps | Guardat a: $OUT_DIR"
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

    pkill -f "entrenament_comparatiu" 2>/dev/null || true
    sleep 5
}

echo ""
echo "###################################################"
echo "# PARAL·LEL 16M: DQN / NFSP / PPO"
echo "###################################################"
echo ""

run_and_time "dqn"  "${OUT_BASE}/dqn"
run_and_time "nfsp" "${OUT_BASE}/nfsp"
run_and_time "ppo"  "${OUT_BASE}/ppo"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

echo ""
echo "==================================================="
echo " EXPERIMENT COMPLETAT!"
echo " Resultats a: ${OUT_BASE}"
echo "==================================================="
