TIMESTEPS_MANS=24000000
TIMESTEPS_FULL=24000000

# Obtenim la data actual per la carpeta de resultats
TIMESTAMP=$(date +"%d_%m_%H%Mh")
BASE_DIR="experiments_comparativa_${TIMESTAMP}"

mkdir -p "$BASE_DIR"

echo "==================================================="
echo " INICIANT CASCADA D'EXPERIMENTS (MLP vs GRU)"
echo " Directori Resultats: $BASE_DIR"
echo "==================================================="

# Funció auxiliar per executar i cronometrar, agafa 4 params
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
    
    # Marquem l'inici del rellotge
    start_time=$(date +%s)
    
    # Assegurem que l'arrel del projecte es pot importar
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    python3 "$script_path" --total_timesteps "$timesteps" --save_dir "$out_dir" $extra_args
    
    # Calculem la diferència de temps
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ">> Finalitzat [$name] en ${duration} segons."
    echo "${name}: ${duration} segons" >> "${BASE_DIR}/resum_temps.txt"
}

# Curriculum 
run_and_time "mlp_fase1_mans" "RL/entrenament/entrenamentsPropis/ppo/entrenament_ppo_mlp_ma.py" "$TIMESTEPS_MANS" "--mode finetune"

# MLP Finetune
BEST_MLP_MA="${BASE_DIR}/mlp_fase1_mans/best.pt"
if [ -f "$BEST_MLP_MA" ]; then
    run_and_time "mlp_fase2_finetune" "RL/entrenament/entrenamentsPropis/ppo/entrenament_ppo_mlp.py" "$TIMESTEPS_FULL" "--mode finetune --load_model $BEST_MLP_MA"
else
    echo "ERROR: No s'ha trobat el model $BEST_MLP_MA. Saltant."
fi

# MLP Scratch
run_and_time "mlp_baseline_scratch" "RL/entrenament/entrenamentsPropis/ppo/entrenament_ppo_mlp.py" "$TIMESTEPS_FULL" "--mode finetune"

# GRU Curriculum
run_and_time "gru_fase1_mans" "RL/entrenament/entrenamentsPropis/ppo_gru/entrenament_ppo_gru_ma.py" "$TIMESTEPS_MANS" "--mode finetune"

# GRU Finetune
BEST_GRU_MA="${BASE_DIR}/gru_fase1_mans/best.pt"
if [ -f "$BEST_GRU_MA" ]; then
    run_and_time "gru_fase2_finetune" "RL/entrenament/entrenamentsPropis/ppo_gru/entrenament_ppo_gru.py" "$TIMESTEPS_FULL" "--mode finetune --load_model $BEST_GRU_MA"
else
    echo "ERROR: No s'ha trobat el model $BEST_GRU_MA. Saltant."
fi

# GRU Scratch
run_and_time "gru_baseline_scratch" "RL/entrenament/entrenamentsPropis/ppo_gru/entrenament_ppo_gru.py" "$TIMESTEPS_FULL" "--mode finetune"


echo "==================================================="
echo " TOTS ELS EXPERIMENTS COMPLETATS AMB ÈXIT!"
echo " Revisa els models generats i la durada a: $BASE_DIR/resum_temps.txt"
echo "==================================================="
