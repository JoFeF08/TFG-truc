import os
from pathlib import Path

import argparse

from controlador import Controlador, ModelInteractiu
from vista import VistaConsola


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo del joc Truc")
    parser.add_argument(
        "--vista",
        choices=["consola", "web", "desktop"],
        default="consola",
        help="Vista: consola (terminal), web (servidor) o desktop (Tkinter)",
    )
    args = parser.parse_args()

    if args.vista == "web":
        import uvicorn
        from vista.vista_web.backend.main import app

        port = int(os.environ.get("PORT", 8001))
        print(f"Servidor web a http://localhost:{port} — obre el navegador per jugar.")
        uvicorn.run(app, host="0.0.0.0", port=port)
    elif args.vista == "desktop":
        from vista.vista_desktop import VistaDesktop

        vista = VistaDesktop()
        model = ModelInteractiu()
        controlador = Controlador(vista, model)
        try:
            controlador.bucle_principal()
        except KeyboardInterrupt:
            vista.mostrar_sortint()
    else:
        vista = VistaConsola()
        model = ModelInteractiu()
        controlador = Controlador(vista, model)
        try:
            controlador.bucle_principal()
        except KeyboardInterrupt:
            vista.mostrar_sortint()
