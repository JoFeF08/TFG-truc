import importlib
try:
    import rlcard.envs.registration
    _original_init = rlcard.envs.registration.EnvSpec.__init__
    def _patched_init(self, env_id, entry_point=None):
        self.env_id = env_id
        if entry_point:
            mod_name, class_name = entry_point.split(':')
            try:
                mod = importlib.import_module(mod_name)
                self._entry_point = getattr(mod, class_name)
            except (ImportError, ModuleNotFoundError):
                self._entry_point = None
        else:
            self._entry_point = None
    rlcard.envs.registration.EnvSpec.__init__ = _patched_init
except Exception:
    pass

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


MODEL_PATH = resource_path("TFG_Doc/notebooks/4_memoria/resultats/ppo_ablacio_pool/best.zip")
TIPUS_AGENT = "sb3"          # per Fase 4 LSTM: "sb3" amb algorisme="ppo_lstm"
ALGORISME    = "ppo"         # "ppo" | "dqn" | "ppo_lstm"

PARTIDES_SESSIO = 1

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: {
            "tipus": TIPUS_AGENT,
            "algorisme": ALGORISME,
            "ruta": MODEL_PATH
        },
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

