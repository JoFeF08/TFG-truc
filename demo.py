from controlador import Controlador, ModelInteractiu
from vista import VistaDesktop


if __name__ == "__main__":
    vista = VistaDesktop()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        controlador.bucle_principal()
    except KeyboardInterrupt:
        vista.mostrar_sortint()
