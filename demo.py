import random
import sys

from entorn import ACTION_LIST, HumanPlayer, RandomPlayer, TrucEnv
from vista import VistaConsola


def run_demo(vista: VistaConsola) -> None:
    """Executa una partida; tota la I/O es fa a través de la vista."""
    config = vista.demanar_config()

    # Construir player_classes
    player_classes = {}
    for i in range(config["num_jugadors"]):
        if config["tipus_jugadors"].get(i, 0) == 0:
            player_classes[i] = lambda pid, rng, _v=vista: HumanPlayer(pid, rng, _v)
        else:
            player_classes[i] = RandomPlayer

    env_config = {
        "allow_step_back": False,
        "seed": random.randint(0, 100000),
        "player_class": player_classes,
        "num_jugadors": config["num_jugadors"],
        "cartes_jugador": config["cartes_jugador"],
        "senyes": config["senyes"],
        "puntuacio_final": config["puntuacio_final"],
    }
    env = TrucEnv(env_config)
    game = env.game

    vista.mostrar_inici()
    vista.mostrar_configuracio(
        num_jugadors=config["num_jugadors"],
        tipus_jugadors=config["tipus_jugadors"],
        senyes=config["senyes"],
        cartes_jugador=config["cartes_jugador"],
        puntuacio_final=config["puntuacio_final"],
    )
    vista.mostrar_barrejant()

    state, player_id = env.reset()
    done = False

    while not done:

        vista.mostrar_taula(game.hist_cartes)
        
        player = game.players[player_id]
        action = player.triar_accio(state["raw_obs"])
        action_name = ACTION_LIST[action]
        vista.mostrar_accio_executada(player_id, type(player).__name__, action_name)
        
        state, next_player_id = env.step(action)
        if game.is_over():
            done = True
            break
        player_id = next_player_id

    vista.mostrar_fi_partida(game.score, env.get_payoffs())


if __name__ == "__main__":
    vista = VistaConsola()
    while True:
        try:
            run_demo(vista)
            if not vista.demanar_repetir():
                break
        except KeyboardInterrupt:
            vista.mostrar_sortint()
            sys.exit(0)
