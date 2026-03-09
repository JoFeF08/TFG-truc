#!/bin/bash

# --- CONFIGURACIÓ ---
EPISODIS=100000
PYTHON_EXEC="python3"
SCRIPT_NAME="entrenaments_unificats.py"
# Obtenir la ruta absoluta del directori on es troba aquest script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "================================================================"
echo " INICIANT BATERIA D'ENTRENAMENTS (6 variants) "
echo "================================================================"
echo "Planificació: Seqüencial per optimitzar RAM i VRAM."
echo "Configuració: $EPISODIS episodis per variant."
echo ""

# Llista d'agents i modes per iterar
AGENTS=("nfsp" "dqn")
MODES=("frozen" "finetune" "scratch")

# Bucle principal
for AGENT in "${AGENTS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "------------------------------------------------------------"
        echo ">> EXECUTANT: Agent: $AGENT | Mode: $MODE"
        echo "------------------------------------------------------------"
        
        # Execució de l'entrenament
        $PYTHON_EXEC "$BASE_DIR/$SCRIPT_NAME" \
            --agent "$AGENT" \
            --mode "$MODE" \
            --episodes $EPISODIS
        

        echo "Esperant 30 segons abans del següent..."
        sleep 30
        echo ""
    done
done

echo "================================================================"
echo " PROCESSAMENT FINALITZAT "
echo "================================================================"
