from controlador import Controlador, ModelInteractiu
from vista.vista_desktop.vista_desktop import VistaDesktop

MODEL_PATH = r"C:\Users\ferri\Documents\ProjectesCodi\TFG-truc\entrenament\entrenamentsUnificats\registres\resultats_comparativa\dqn_finetune_0803_0024\models\best.pt"

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: {"tipus": "dqn", "ruta": MODEL_PATH, "amb_cos": True},
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
