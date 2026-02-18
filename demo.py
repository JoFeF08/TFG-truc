import random
import sys
from truc.env import TrucEnv
from truc.cartes_accions import ACTION_LIST
from truc.rols.player.player_huma import HumanPlayer
from truc.rols.player.player_random import RandomPlayer

def get_config_from_user():
    """Demana a l'usuari la configuració del joc"""
    print("\n=== CONFIGURACIÓ DEL JOC ===\n")
    
    # Número de jugadors
    while True:
        try:
            num_jugadors = input("Número de jugadors (2): ").strip()
            num_jugadors = int(num_jugadors) if num_jugadors else 2
            if num_jugadors > 0:
                break
            print("Ha de ser un número positiu!")
        except ValueError:
            print("Si us plau, introdueix un número vàlid.")
    
    # Tipus de jugadors
    player_classes = {}
    print("\nTipus de jugadors disponibles:")
    print("  0: Humà")
    print("  1: Aleatori (Random)")
    
    for i in range(num_jugadors):
        while True:
            try:
                p_type = input(f"Tipus per al Jugador {i} (0=Humà, 1=Random): ").strip()
                p_type = int(p_type) if p_type else 0
                
                if p_type == 0:
                    player_classes[i] = HumanPlayer
                    break
                elif p_type == 1:
                    player_classes[i] = RandomPlayer
                    break
                else:
                    print("Opció no vàlida (0 o 1).")
            except ValueError:
                print("Introdueix un número (0 o 1).")

    # Número de cartes per jugador
    while True:
        try:
            cartes_jugador = input("Cartes per jugador (3): ").strip()
            cartes_jugador = int(cartes_jugador) if cartes_jugador else 3
            if cartes_jugador > 0:
                break
            print("Ha de ser un número positiu!")
        except ValueError:
            print("Si us plau, introdueix un número vàlid.")
    
    # Senyes
    while True:
        senyes_input = input("Activar senyes? (s/N): ").strip().lower()
        if senyes_input in ['s', 'n', '']:
            senyes = (senyes_input == 's')
            break
        print("Si us plau, respon 's' o 'n'.")
    
    # Puntuació final
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
        'num_jugadors': num_jugadors,
        'cartes_jugador': cartes_jugador,
        'senyes': senyes,
        'puntuacio_final': puntuacio_final,
        'player_classes': player_classes
    }

def run_demo():
    print("=== INICIANT DEMO TRUC (TERMINAL) ===")
    
    # Obtenir configuració personalitzada
    user_config = get_config_from_user()
    
    # Configurar entorn amb els paràmetres de l'usuari
    env_config = {
        'allow_step_back': False, 
        'seed': random.randint(0, 100000),
        'player_class': user_config['player_classes'], # Diccionari de classes
        'num_jugadors': user_config['num_jugadors'],
        'cartes_jugador': user_config['cartes_jugador'],
        'senyes': user_config['senyes'],
        'puntuacio_final': user_config['puntuacio_final']
    }
    env = TrucEnv(env_config)

    print(f"Configuració del Joc:")
    print(f"  - Jugadors: {env.game.num_jugadors}")
    for pid, p_cls in user_config['player_classes'].items():
        print(f"    - J{pid}: {p_cls.__name__}")
    print(f"  - Senyes: {'Sí' if env.game.senyes else 'No'}")
    print(f"  - Cartes per jugador: {env.game.cartes_jugador}")
    print(f"  - Puntuació final: {env.game.puntuacio_final}")

    # Reiniciar joc
    print("Barrejant cartes...")
    state, player_id = env.reset()
    done = False
    
   
    while not done:
        game = env.game
        
        # Mostrar Taula
        if game.hist_cartes:
            print("\nTAULA (Històric):")
            for pid, card in game.hist_cartes:
                print(f"  J{pid} -> {card}")
        
        # Obtenir jugador actual
        player = env.game.players[player_id]
        
        action = player.triar_accio(state['raw_obs'])
        action_name = ACTION_LIST[action]

        print(f"Jugador {player_id} ({type(player).__name__}) executant: {action_name}...")
        state, next_player_id = env.step(action)
        
        if env.game.is_over():
            done = True
            break

        # Detectar següent jugador
        if isinstance(next_player_id, list):
            player_id = next_player_id[0]
        else:
            player_id = next_player_id
             

    # Final de partida
    print("\n" + "="*40)
    print("JOC ACABAT!")
    winner = 0 if game.score[0] > game.score[1] else 1 
    
    print(f"Marcador Global: E0: {game.score[0]} - E1: {game.score[1]}")
    
    payoffs = env.get_payoffs()
    print(f"Payoffs (diferència de punts): {payoffs}")

if __name__ == "__main__":
    while True:
        try:
            run_demo()
            print("\n")
            retry = input("Vols jugar un altra mà? (s/n): ")
            if retry.lower() != 's':
                break
        except KeyboardInterrupt:
            print("\nSortint...")
            sys.exit(0)
