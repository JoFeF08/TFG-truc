import sys
from truc.rols.player.player import TrucPlayer
from truc.cartes_accions import ACTION_LIST

class HumanPlayer(TrucPlayer):
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)

    def triar_accio(self, state):
        """
        Demana al usuari quina acció vol realitzar donat l'estat actual.
        """
        print(f"\n--- Torn del Jugador {self.player_id} ---")
        print(f"La teva mà: {state['ma_jugador']}")
        print(f"Cartes a la taula: {state['hist_cartes']}")
        print(f"Puntuació: {state['puntuacio']}")
        
        accions_legals = state['accions_legals']
        print("\nAccions disponibles:")
        
        options = {}
        for i, action_idx in enumerate(accions_legals):
            action_name = ACTION_LIST[action_idx]
            options[i] = action_idx
            print(f"{i}: {action_name}")

        while True:
            try:
                user_input = input("Selecciona una opció (número): ")
                choice = int(user_input)
                if choice in options:
                    return options[choice]
                else:
                    print("Opció no vàlida. Torna-ho a provar.")
            except ValueError:
                print("Entrada no vàlida. Introdueix un número.")
            except EOFError:
                print("Entrada finalitzada (EOF). Sortint...")
                sys.exit(0)
            except KeyboardInterrupt:
                print("\nInterrupció per teclat detectada. Sortint...")
                sys.exit(0)
