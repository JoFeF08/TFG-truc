#!/bin/bash
# run_comparatiu.sh
# Executa els tres algorismes (DQN, NFSP, PPO) en dues condicions:
#   1. Paral·lel (defaults: 16/48 envs)
#   2. Seqüencial (1 env)
#
# Ús:
#   bash run_comparatiu.sh                          # 8M steps (defecte)
#   bash run_comparatiu.sh --timesteps 3000000      # custom

TIMESTEPS=8000000
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

run_and_time() {
    local AGENT=$1
    local OUT_DIR=$2
    local EXTRA_ARGS=$3

    echo "==================================================="
    echo ">> Iniciant [$AGENT] | Guardat a: $OUT_DIR"
    echo "==================================================="

    START=$(date +%s)

    python3 "$SCRIPT" \
        --agent "$AGENT" \
        --total_timesteps "$TIMESTEPS" \
        --save_dir "$OUT_DIR" \
        $EXTRA_ARGS

    END=$(date +%s)
    DURATION=$((END - START))
    HORES=$((DURATION / 3600))
    MINUTS=$(( (DURATION % 3600) / 60 ))
    SEGONS=$((DURATION % 60))

    echo ">> [$AGENT] completat en ${HORES}h ${MINUTS}m ${SEGONS}s"
    echo "${AGENT}: ${DURATION}s (${HORES}h ${MINUTS}m ${SEGONS}s)" >> "$RESUM"

    sleep 15
}

# ===================== FASE 1: PARAL·LEL =====================
BASE_PARALLEL="resultats_comparativa_parallel_${TIMESTAMP}"
mkdir -p "$BASE_PARALLEL"
RESUM="${BASE_PARALLEL}/resum_temps.txt"
echo "Comparativa PARAL·LEL - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

echo ""
echo "###################################################"
echo "# FASE 1: PARAL·LEL (16/48 envs) - ${TIMESTEPS} steps"
echo "###################################################"
echo ""

run_and_time "dqn"  "${BASE_PARALLEL}/dqn"  ""
run_and_time "nfsp" "${BASE_PARALLEL}/nfsp" ""
run_and_time "ppo"  "${BASE_PARALLEL}/ppo"  ""

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

# ===================== FASE 2: SEQÜENCIAL =====================
BASE_SEQ="resultats_comparativa_1env_${TIMESTAMP}"
mkdir -p "$BASE_SEQ"
RESUM="${BASE_SEQ}/resum_temps.txt"
echo "Comparativa SEQÜENCIAL (1 env) - $(date)" > "$RESUM"
echo "Timesteps: $TIMESTEPS" >> "$RESUM"
echo "-------------------------------------------" >> "$RESUM"

echo ""
echo "###################################################"
echo "# FASE 2: SEQÜENCIAL (1 env) - ${TIMESTEPS} steps"
echo "###################################################"
echo ""

run_and_time "dqn"  "${BASE_SEQ}/dqn"  "--num_envs 1"
run_and_time "nfsp" "${BASE_SEQ}/nfsp" "--num_envs 1"
run_and_time "ppo"  "${BASE_SEQ}/ppo"  "--num_envs 1"

echo "-------------------------------------------" >> "$RESUM"
echo "Completat: $(date)" >> "$RESUM"

# ===================== RESUM FINAL =====================
echo ""
echo "==================================================="
echo " TOTS ELS EXPERIMENTS COMPLETATS!"
echo " Paral·lel: $BASE_PARALLEL"
echo " Seqüencial: $BASE_SEQ"
echo "==================================================="
