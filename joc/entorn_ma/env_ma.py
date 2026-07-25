import sys
import os
from collections import OrderedDict
from joc.entorn.env import BaseTrucEnv

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn_ma.game_ma import TrucGameMa
from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, ACTIONS_SIGNAL, init_joc_cartes
from joc.entorn.obs_builder import extract_obs, obs_shapes


class TrucEnvMa(BaseTrucEnv):
    """
    Entorn per entrenar per mans individuals.
    Cada episodi = una mà. Done=True quan la mà acaba.
    Reward = punts_truc + punts_envit guanyats/perduts (normalitzat per 24).
    """
    def __init__(self, config):
        self.name = 'truc_ma'
        self.num_jugadors = config.get('num_jugadors', 2)
        self.cartes_jugador = config.get('cartes_jugador', 3)
        self.puntuacio_final = config.get('puntuacio_final', 999)
        senyes = config.get('senyes', False)
        verbose = config.get('verbose', False)
        player_class = config.get('player_class', None)
        permetre_apostes = config.get('permetre_apostes', True)
        permetre_truc = config.get('permetre_truc', True)

        if player_class:
            self.game = TrucGameMa(
                num_jugadors=self.num_jugadors,
                cartes_jugador=self.cartes_jugador,
                senyes=senyes,
                puntuacio_final=self.puntuacio_final,
                player_class=player_class,
                verbose=verbose,
                permetre_apostes=permetre_apostes,
                permetre_truc=permetre_truc,
            )
        else:
            self.game = TrucGameMa(
                num_jugadors=self.num_jugadors,
                cartes_jugador=self.cartes_jugador,
                senyes=senyes,
                puntuacio_final=self.puntuacio_final,
                verbose=verbose,
                permetre_apostes=permetre_apostes,
                permetre_truc=permetre_truc,
            )

        config.setdefault('allow_step_back', False)
        config.setdefault('seed', None)
        super().__init__(config)

        self.cartes = init_joc_cartes()
        self.carta_map = {carta: i for i, carta in enumerate(self.cartes)}
        self.signal_map = {signal: i for i, signal in enumerate(ACTIONS_SIGNAL)}

        self.OBS_CARTES_SHAPE, self.OBS_CONTEXT_SIZE = obs_shapes(self.num_jugadors, senyes)

        self.state_size = (
            self.OBS_CARTES_SHAPE[0] * self.OBS_CARTES_SHAPE[1] * self.OBS_CARTES_SHAPE[2]
            + self.OBS_CONTEXT_SIZE
        )
        self.state_shape = [[self.state_size] for _ in range(self.num_jugadors)]
        self.action_shape = [[len(ACTION_LIST)] for _ in range(self.num_jugadors)]

    def _extract_state(self, state):
        obs_cartes, obs_context = extract_obs(
            state, self.num_jugadors, self.cartes_jugador, self.game.senyes
        )

        legal_actions_list = state['accions_legals']
        legal_actions = OrderedDict({a: None for a in legal_actions_list})

        extracted_state = {
            'obs': {'obs_cartes': obs_cartes, 'obs_context': obs_context},
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': [ACTION_LIST[a] for a in legal_actions_list],
            'action_record': self.action_recorder
        }
        return extracted_state

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_estat_taula(self, player_id):
        return self.game.get_state(player_id)

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        return self.game.get_legal_actions()
