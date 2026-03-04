from controlador import Controlador, ModelInteractiu
from vista.vista_desktop.vista_desktop import VistaDesktop

MODEL_PATH = r"c:\Users\ferri\Documents\ProjectesCodi\TFG-truc\entrenament\registres\27_2_26_a_les_2002\models\nfsp_truc_p0.pt"

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: {"tipus": "nfsp", "ruta": MODEL_PATH},
    },
}

if __name__ == "__main__":
    vista = VistaDesktop()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        controlador.executar_partida(override_config=config)
    except KeyboardInterrupt:
        vista.mostrar_sortint()
