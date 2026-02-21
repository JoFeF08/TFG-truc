import os
import sys

# Concentrar tot el bytecode (__pycache__) dins la carpeta temp/ del projecte
_root = os.path.dirname(os.path.abspath(__file__))
_pycache_dir = os.path.join(_root, "temp", "pycache")
os.environ["PYTHONPYCACHEPREFIX"] = _pycache_dir
os.makedirs(_pycache_dir, exist_ok=True)

import argparse
import random
import time

from entorn import ACTION_LIST, HumanPlayer, RandomPlayer, TrucEnv
from vista import VistaConsola, VistaDesktop


def run_demo(vista) -> None:
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

    id_jugador_huma = next(
        (i for i in range(config["num_jugadors"]) if config["tipus_jugadors"].get(i, 0) == 0),
        None,
    )

    state, player_id = env.reset()
    done = False

    while not done:
        # Actualitzar taula al començament de cada torn (la que mostra les cartes és mostrar_taula)
        estat_inicial = env.get_estat_taula(id_jugador_huma) if id_jugador_huma is not None else None
        vista.mostrar_taula(game.hist_cartes, estat_inicial)

        player = game.players[player_id]
        action = player.triar_accio(state["raw_obs"])
        action_name = ACTION_LIST[action]
        vista.mostrar_accio_executada(player_id, type(player).__name__, action_name)
        if isinstance(player, RandomPlayer):
            time.sleep(player._time_ms / 1000.0)

        state, next_player_id = env.step(action)
        if id_jugador_huma is not None:
            state_taula = env.get_estat_taula(id_jugador_huma)
            vista.actualitzar_taula(state_taula)
            time.sleep(0.5)
        if game.is_over():
            done = True
            break
        player_id = next_player_id

    vista.mostrar_fi_partida(game.score, env.get_payoffs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo del joc Truc")
    parser.add_argument(
        "--vista",
        choices=["consola", "desktop"],
        default="consola",
        help="Vista a usar: consola (terminal) o desktop (Tkinter)",
    )
    args = parser.parse_args()
    if args.vista == "desktop":
        vista = VistaDesktop()
    else:
        vista = VistaConsola()
    while True:
        try:
            run_demo(vista)
            if not vista.demanar_repetir():
                break
        except KeyboardInterrupt:
            vista.mostrar_sortint()
            sys.exit(0)
