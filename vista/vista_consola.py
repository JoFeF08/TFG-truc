import sys
from entorn.cartes_accions import ACTION_LIST


class VistaConsola:
    """Vista per consola"""

    def escollir_accio(self, accions_legals: list, state: dict) -> int:
        """Mostra torn, mà, taula, puntuació; llista accions; demana input; retorna codi d'acció."""
        print(f"\n--- Torn del Jugador {state.get('id_jugador', '?')} ---")
        print(f"La teva mà: {state.get('ma_jugador', [])}")
        print(f"Cartes a la taula: {state.get('hist_cartes', [])}")
        print(f"Puntuació: {state.get('puntuacio', [])}")
        print("\nAccions disponibles:")

        options = {}
        for i, action_idx in enumerate(accions_legals):
            action_name = ACTION_LIST[action_idx]
            options[i] = action_idx
            print(f"  {i}: {action_name}")

        while True:
            try:
                user_input = input("Selecciona una opció (número): ")
                choice = int(user_input)
                if choice in options:
                    return options[choice]
                print("Opció no vàlida. Torna-ho a provar.")
            except ValueError:
                print("Entrada no vàlida. Introdueix un número.")
            except EOFError:
                print("Entrada finalitzada (EOF). Sortint...")
                sys.exit(0)
            except KeyboardInterrupt:
                print("\nInterrupció per teclat detectada. Sortint...")
                sys.exit(0)

    def demanar_config(self) -> dict:
        print("\n=== CONFIGURACIÓ DEL JOC ===\n")

        while True:
            try:
                num_jugadors = input("Número de jugadors (2): ").strip()
                num_jugadors = int(num_jugadors) if num_jugadors else 2
                if num_jugadors >= 2 and num_jugadors % 2 == 0:
                    break
                print("Ha de ser un número parell major o igual a 2.")
            except ValueError:
                print("Si us plau, introdueix un número vàlid.")

        print("\nTipus de jugadors disponibles:")
        print("  0: Humà")
        print("  1: Aleatori (Random)")

        tipus_jugadors = {}
        for i in range(num_jugadors):
            while True:
                try:
                    p_type = input(f"Tipus per al Jugador {i} (0=Humà, 1=Random): ").strip()
                    p_type = int(p_type) if p_type else 0
                    if p_type in (0, 1):
                        tipus_jugadors[i] = p_type
                        break
                    print("Opció no vàlida (0 o 1).")
                except ValueError:
                    print("Introdueix un número (0 o 1).")

        while True:
            try:
                cartes_jugador = input("Cartes per jugador (3): ").strip()
                cartes_jugador = int(cartes_jugador) if cartes_jugador else 3
                if cartes_jugador > 0:
                    break
                print("Ha de ser un número positiu!")
            except ValueError:
                print("Si us plau, introdueix un número vàlid.")

        if num_jugadors == 2:
            senyes = False
        else:
            while True:
                senyes_input = input("Activar senyes? (s/N): ").strip().lower()
                if senyes_input in ("s", "n", ""):
                    senyes = senyes_input == "s"
                    break
                print("Si us plau, respon 's' o 'n'.")

        while True:
            try:
                puntuacio_final = input("Puntuació final per guanyar (24): ").strip()
                puntuacio_final = int(puntuacio_final) if puntuacio_final else 24
                if puntuacio_final > 0:
                    break
                print("Ha de ser un número positiu!")
            except ValueError:
                print("Si us plau, introdueix un número vàlid.")

        return {
            "num_jugadors": num_jugadors,
            "cartes_jugador": cartes_jugador,
            "senyes": senyes,
            "puntuacio_final": puntuacio_final,
            "tipus_jugadors": tipus_jugadors,
        }

    def mostrar_inici(self) -> None:
    
        print("=== INICIANT DEMO TRUC (TERMINAL) ===")

    def mostrar_configuracio(
        self,
        num_jugadors: int,
        tipus_jugadors: dict,
        senyes: bool,
        cartes_jugador: int,
        puntuacio_final: int,
    ) -> None:
        """Mostra jugadors, senyes, cartes, puntuació final"""
        noms_tipus = {0: "Humà", 1: "Random"}
        print("Configuració del Joc:")
        print(f"  - Jugadors: {num_jugadors}")
        for pid in range(num_jugadors):
            tipus = tipus_jugadors.get(pid, 0)
            print(f"    - J{pid}: {noms_tipus.get(tipus, 'Humà')}")
        print(f"  - Senyes: {'Sí' if senyes else 'No'}")
        print(f"  - Cartes per jugador: {cartes_jugador}")
        print(f"  - Puntuació final: {puntuacio_final}")

    def mostrar_barrejant(self) -> None:
        print("Barrejant cartes...")

    def mostrar_taula(self, hist_cartes: list) -> None:
        """Mostra les cartes jugades a la taula."""
        if hist_cartes:
            print("\nTAULA (Històric):")
            for pid, card in hist_cartes:
                print(f"  J{pid} -> {card}")

    def mostrar_accio_executada(
        self, player_id: int, nom_jugador: str, action_name: str
    ) -> None:
        print(f"Jugador {player_id} ({nom_jugador}) executant: {action_name}...")

    def mostrar_fi_partida(self, score: list, payoffs: list) -> None:
        print("\n" + "=" * 40)
        print("JOC ACABAT!")
        print(f"Marcador Global: E0: {score[0]} - E1: {score[1]}")
        print(f"Payoffs (diferència de punts): {payoffs}")

    def demanar_repetir(self) -> bool:
        """Pregunta si es vol jugar una altra partida"""
        try:
            retry = input("Vols jugar una altra partida? (s/n): ").strip().lower()
            return retry == "s"
        except (EOFError, KeyboardInterrupt):
            return False

    def mostrar_sortint(self) -> None:
        print("\nSortint...")
