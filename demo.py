import random
from truc.env import TrucEnv
from truc.cartes_accions import ACTION_LIST
from truc.rols.player.player_huma import HumanPlayer

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
        'puntuacio_final': puntuacio_final
    }

def run_demo():
    print("=== INICIANT DEMO TRUC (TERMINAL) ===")
    
    # Obtenir configuració personalitzada
    user_config = get_config_from_user()
    
    # Configurar entorn amb els paràmetres de l'usuari
    env_config = {
        'allow_step_back': False, 
        'seed': random.randint(0, 100000),
        'player_class': HumanPlayer,
        'num_jugadors': user_config['num_jugadors'],
        'cartes_jugador': user_config['cartes_jugador'],
        'senyes': user_config['senyes'],
        'puntuacio_final': user_config['puntuacio_final']
    }
    env = TrucEnv(env_config)

    print(f"Configuració del Joc:")
    print(f"  - Jugadors: {env.game.num_jugadors}")
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

        # Executar Pas
        print(f"Executant: {ACTION_LIST[action]}...")
        state, next_player_id = env.step(action)
        
        # Detectar final
        if isinstance(next_player_id, list):
             player_id = next_player_id[0]
        else:
             player_id = next_player_id
             
        if env.game.is_over():
            done = True

    # Final de partida
    print("\n" + "="*40)
    print("JOC ACABAT!")
    winner = 0 if game.score[0] > game.score[1] else 1 
    
    print(f"Marcador Final Mà: J0: {game.score[0]} - J1: {game.score[1]}")
    
    truc_win = game.judger.guanyador_ma(game.ronda_winners, game.ma)
    if truc_win != -1:
        print(f"Guanyador per Cartes: J{truc_win}")
    else:
        print("Final per Fold (No vull).")

if __name__ == "__main__":
    while True:
        run_demo()
        print("\n")
        retry = input("Vols jugar un altra mà? (s/n): ")
        if retry.lower() != 's':
            break
