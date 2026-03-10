import importlib
try:
    import rlcard.envs.registration
    # Patch EnvSpec.__init__ to ignore missing modules (like blackjack)
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

from controlador import Controlador, ModelInteractiu
from vista.vista_desktop.vista_desktop import VistaDesktop

import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


DEFAULT_MODEL_PATH = r"C:\Users\ferri\Documents\ProjectesCodi\TFG-truc\entrenament\entrenamentsUnificats\registres\resultats_comparativa\dqn_finetune_0803_0024\models\best.pt"

is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, 'nuitka_version')

if is_frozen:
    MODEL_PATH = resource_path(os.path.join("models", "best.onnx"))
    TIPUS_AGENT = "onnx"
else:
    MODEL_PATH = DEFAULT_MODEL_PATH
    TIPUS_AGENT = "dqn"

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: {"tipus": TIPUS_AGENT, "ruta": MODEL_PATH, "amb_cos": True},
    },
}


if __name__ == "__main__":
    print("Iniciant demo.py...")
    print("Creant VistaDesktop...")
    vista = VistaDesktop()
    print("Creant ModelInteractiu...")
    model = ModelInteractiu()
    print("Creant Controlador...")
    controlador = Controlador(vista, model)
    try:
        print("Executant partida...")
        controlador.executar_partida(override_config=config)
    except KeyboardInterrupt:
        print("Sortint...")
        vista.mostrar_sortint()

