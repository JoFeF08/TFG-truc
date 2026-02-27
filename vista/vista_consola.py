import sys
import os

# Permet executar aquest fitxer directament: afegir l'arrel del projecte a sys.path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import time

from entorn.cartes_accions import ACTION_LIST
from vista.interficie_vista import Vista


class VistaConsola(Vista):
    """Vista per consola"""

    BOT_DELAY_S: float = 1.0

    CONFIG_PER_DEFECTE: dict = {
        "num_jugadors": 2,
        "cartes_jugador": 3,
        "senyes": False,
        "puntuacio_final": 24,
        "tipus_jugadors": {0: 0, 1: 1},
    }

    def demanar_config(self) -> dict:
        return self.CONFIG_PER_DEFECTE.copy()

    def mostrar_estat(self, estat: dict) -> None:
        pid = estat.get("id_jugador", "?")
        puntuacio = estat.get("puntuacio", [])
        ronda = (estat.get("comptador_ronda", 0)) + 1
        ma = estat.get("ma", 0)
        hist = estat.get("hist_cartes", [])
        ma_jugador = estat.get("ma_jugador", [])
        truc = estat.get("estat_truc", {})
        envit = estat.get("estat_envit", {})

        print(f"\n{'='*40}")
        print(f"  Puntuació: E0: {puntuacio[0]}  —  E1: {puntuacio[1]}")
        print(f"  Ronda: {ronda}  |  Mà: Jugador {ma}")
        if truc.get("level", 1) > 1:
            print(f"  Truc: {truc['level']}")
        if envit.get("level", 0) > 0:
            print(f"  Envit: {envit['level']}")

        if hist:
            print("\n  Taula:")
            for p, card in hist:
                print(f"    J{p} → {card}")

        print(f"\n  La teva mà (J{pid}): {ma_jugador}")

    def escollir_accio(self, accions_legals: list, estat: dict) -> int:
        print("\n  Accions disponibles:")
        options = {}
        for i, action_idx in enumerate(accions_legals):
            action_name = ACTION_LIST[action_idx]
            options[i] = action_idx
            print(f"    {i}: {action_name}")

        while True:
            try:
                user_input = input("  Selecciona una opció (número): ")
                choice = int(user_input)
                if choice in options:
                    return options[choice]
                print("  Opció no vàlida. Torna-ho a provar.")
            except ValueError:
                print("  Entrada no vàlida. Introdueix un número.")
            except EOFError:
                print("  Entrada finalitzada (EOF). Sortint...")
                sys.exit(0)
            except KeyboardInterrupt:
                print("\n  Interrupció per teclat. Sortint...")
                sys.exit(0)

    def mostrar_accio(self, jugador_id: int, nom_accio: str, es_bot: bool) -> None:
        prefix = "Bot" if es_bot else "Tu"
        print(f"  Jugador {jugador_id} ({prefix}): {nom_accio}")
        if es_bot:
            time.sleep(self.BOT_DELAY_S)

    def mostrar_fi_partida(self, score: list, payoffs: list) -> None:
        print(f"\n{'='*40}")
        print("  JOC ACABAT!")
        print(f"  Marcador: E0: {score[0]}  —  E1: {score[1]}")
        print(f"  Payoffs: {payoffs}")

    def demanar_repetir(self) -> bool:
        try:
            retry = input("\n  Vols jugar una altra partida? (s/n): ").strip().lower()
            return retry == "s"
        except (EOFError, KeyboardInterrupt):
            return False

    def mostrar_sortint(self) -> None:
        print("\n  Sortint...")


if __name__ == "__main__":
    from controlador import Controlador, ModelInteractiu

    vista = VistaConsola()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        controlador.bucle_principal()
    except KeyboardInterrupt:
        vista.mostrar_sortint()
