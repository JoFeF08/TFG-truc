from controlador import Controlador, ModelInteractiu
from vista import VistaConsola


if __name__ == "__main__":
    vista = VistaConsola()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        controlador.bucle_principal()
    except KeyboardInterrupt:
        vista.mostrar_sortint()
