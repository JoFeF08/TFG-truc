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


MODEL_PATH = resource_path("best.pt")
TIPUS_AGENT = "ppo_mlp"

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: {
            "tipus": TIPUS_AGENT, 
            "ruta": MODEL_PATH, 
        },
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

