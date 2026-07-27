from joc.controlador import Controlador, ModelInteractiu
from joc.vista.vista_desktop.vista_desktop import VistaDesktop

import sys
import os

def resource_path(relative_path):
    """
    Get absolute path to resource.
    - PyInstaller onefile: usa sys._MEIPASS (directori d'extracció temporal)
    - Nuitka onefile:      usa __file__ (apunta al directori d'extracció de Nuitka)
    - Desenvolupament:     usa el directori del propi script
    """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller onefile
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# Pesos: entrenament --stage cartes --opponent mixt (mans sense apostes).
# Checkpoint de 1.6M, no el "final" (2M) -- mesurat empíricament millor
# (59.0% vs Random, enfront del 56.3% del final; corba d'aprenentatge amb
# una petita davallada als últims ~400k passos).
MODEL_PATH = resource_path("RL\\entrenament\\registres_sb3\\25_07_26_a_les_1725\\pool\\ppo_truc_1600000_steps.zip")
TIPUS_AGENT = "sb3"          # "sb3" | "nfsp_sl" | "regles"
ALGORISME    = "maskable_ppo"  # només per "sb3": "maskable_ppo" | "ppo" | "dqn" | "ppo_lstm"
VARIANT_REGLES = "conservador"  # només per "regles": conservador|equilibrat|agressiu|farol

PARTIDES_SESSIO = 1

# Spec del jugador IA segons el tipus
if TIPUS_AGENT == "sb3":
    _spec_ia = {"tipus": "sb3", "algorisme": ALGORISME, "ruta": MODEL_PATH}
elif TIPUS_AGENT == "regles":
    _spec_ia = {"tipus": "regles", "variant": VARIANT_REGLES}
else:
    _spec_ia = {"tipus": TIPUS_AGENT, "ruta": MODEL_PATH}

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    # El model es va entrenar en l'etapa "cartes" del currículum (sense
    # truc ni envit) -- cal jugar-hi en les mateixes condicions.
    "permetre_apostes": False,
    "permetre_truc": False,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: _spec_ia,
    },
}


def _reset_memoria_agents(controlador):
    """Crida reset_memoria() a tots els agents IA del controlador (si el suporten)."""
    try:
        models = getattr(controlador, '_models', None) or getattr(controlador.model, '_models', None)
        if models:
            for m in models.values():
                if m is not None and hasattr(m, 'reset_memoria'):
                    m.reset_memoria()
    except Exception:
        pass


if __name__ == "__main__":
    print("Iniciant demo.py...")
    vista = VistaDesktop()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        for partida_idx in range(PARTIDES_SESSIO):
            if partida_idx == 0:
                _reset_memoria_agents(controlador)  # inici de sessió
            print(f"Executant partida {partida_idx + 1}/{PARTIDES_SESSIO}...")
            controlador.executar_partida(override_config=config)
    except KeyboardInterrupt:
        print("Sortint...")
        vista.mostrar_sortint()

