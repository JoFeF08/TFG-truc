#!/bin/bash
# Experiments fase2 amb FINETUNE complet (cos + GRU descongelats)
# Carrega els millors models fase1 dels dos runs anteriors (1246h i 2317h)
#
# UNFREEZE_FRACTION=0.15/0.20 → col·lapse (política inestable quan es descongela)
# UNFREEZE_FRACTION=0.60      → descongelar quan la política ja és robusta

TIMESTEPS_FULL=24000000
UNFREEZE_FRACTION=0.60

TIMESTAMP=$(date +"%d_%m_%H%Mh")
BASE_DIR="experiments_finetune_${TIMESTAMP}"

mkdir -p "$BASE_DIR"

# Paths als best.pt de fase1 existents
MLP_FASE1_RUN1="experiments_comparativa_26_03_1246h/mlp_fase1_mans/best.pt"
MLP_FASE1_RUN2="experiments_comparativa_26_03_2317h/mlp_fase1_mans/best.pt"
GRU_FASE1_RUN1="experiments_comparativa_26_03_1246h/gru_fase1_mans/best.pt"
GRU_FASE1_RUN2="experiments_comparativa_26_03_2317h/gru_fase1_mans/best.pt"

echo "==================================================="
echo " EXPERIMENTS FASE2 FINETUNE (cos + GRU descongelats)"
echo " Directori Resultats: $BASE_DIR"
echo "==================================================="

run_and_time() {
    local name=$1
    local script_path=$2
    local timesteps=$3
    local extra_args=$4
    local out_dir="${BASE_DIR}/${name}"

    echo "---------------------------------------------------"
    echo ">> Iniciant [$name]"
    echo ">> Lloc de guardat: $out_dir"

    mkdir -p "$out_dir"

    start_time=$(date +%s)
    export PYTHONPATH="$(pwd):$PYTHONPATH"

    python3 "$script_path" --total_timesteps "$timesteps" --save_dir "$out_dir" $extra_args

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ">> Finalitzat [$name] en ${duration} segons."
    echo "${name}: ${duration} segons" >> "${BASE_DIR}/resum_temps.txt"
}

# --- MLP Fase2 Finetune ---
if [ -f "$MLP_FASE1_RUN1" ]; then
    run_and_time "mlp_fase2_finetune_run1" \
        "RL/entrenament/entrenamentsPropis/ppo/entrenament_ppo_mlp.py" \
        "$TIMESTEPS_FULL" \
        "--mode finetune --unfreeze_fraction $UNFREEZE_FRACTION --load_model $MLP_FASE1_RUN1"
else
    echo "ERROR: No s'ha trobat $MLP_FASE1_RUN1. Saltant."
fi

if [ -f "$MLP_FASE1_RUN2" ]; then
    run_and_time "mlp_fase2_finetune_run2" \
        "RL/entrenament/entrenamentsPropis/ppo/entrenament_ppo_mlp.py" \
        "$TIMESTEPS_FULL" \
        "--mode finetune --unfreeze_fraction $UNFREEZE_FRACTION --load_model $MLP_FASE1_RUN2"
else
    echo "ERROR: No s'ha trobat $MLP_FASE1_RUN2. Saltant."
fi

# --- GRU Fase2 Finetune ---
if [ -f "$GRU_FASE1_RUN1" ]; then
    run_and_time "gru_fase2_finetune_run1" \
        "RL/entrenament/entrenamentsPropis/ppo_gru/entrenament_ppo_gru.py" \
        "$TIMESTEPS_FULL" \
        "--mode finetune --unfreeze_fraction $UNFREEZE_FRACTION --load_model $GRU_FASE1_RUN1"
else
    echo "ERROR: No s'ha trobat $GRU_FASE1_RUN1. Saltant."
fi

if [ -f "$GRU_FASE1_RUN2" ]; then
    run_and_time "gru_fase2_finetune_run2" \
        "RL/entrenament/entrenamentsPropis/ppo_gru/entrenament_ppo_gru.py" \
        "$TIMESTEPS_FULL" \
        "--mode finetune --unfreeze_fraction $UNFREEZE_FRACTION --load_model $GRU_FASE1_RUN2"
else
    echo "ERROR: No s'ha trobat $GRU_FASE1_RUN2. Saltant."
fi

echo "==================================================="
echo " TOTS ELS EXPERIMENTS COMPLETATS!"
echo " Revisa els models a: $BASE_DIR"
echo " Temps totals a: $BASE_DIR/resum_temps.txt"
echo "==================================================="
