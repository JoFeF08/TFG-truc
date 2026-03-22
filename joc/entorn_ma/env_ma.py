import sys
import os
import numpy as np
from collections import OrderedDict
from rlcard.envs.env import Env

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn_ma.game_ma import TrucGameMa
from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, ACTIONS_SIGNAL, PALS, NUMS, init_joc_cartes

_PAL_IDX = {p: i for i, p in enumerate(PALS)}
_NUM_IDX = {n: i for i, n in enumerate(NUMS)}


class TrucEnvMa(Env):
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

        if player_class:
            self.game = TrucGameMa(
                num_jugadors=self.num_jugadors,
                cartes_jugador=self.cartes_jugador,
                senyes=senyes,
                puntuacio_final=self.puntuacio_final,
                player_class=player_class,
                verbose=verbose
            )
        else:
            self.game = TrucGameMa(
                num_jugadors=self.num_jugadors,
                cartes_jugador=self.cartes_jugador,
                senyes=senyes,
                puntuacio_final=self.puntuacio_final,
                verbose=verbose
            )

        config.setdefault('allow_step_back', False)
        config.setdefault('seed', None)
        super().__init__(config)

        self.cartes = init_joc_cartes()
        self.carta_map = {carta: i for i, carta in enumerate(self.cartes)}
        self.signal_map = {signal: i for i, signal in enumerate(ACTIONS_SIGNAL)}

        self.OBS_CARTES_SHAPE = (6, 4, 9)
        self.OBS_CONTEXT_SIZE = 23

        self.state_size = (
            self.OBS_CARTES_SHAPE[0] * self.OBS_CARTES_SHAPE[1] * self.OBS_CARTES_SHAPE[2]
            + self.OBS_CONTEXT_SIZE
        )
        self.state_shape = [[self.state_size] for _ in range(self.num_jugadors)]
        self.action_shape = [[len(ACTION_LIST)] for _ in range(self.num_jugadors)]

    def _carta_a_idx(self, carta_str):
        pal = carta_str[0]
        rang = carta_str[1:]
        return _PAL_IDX.get(pal), _NUM_IDX.get(rang)

    def _extract_state(self, state):
        player_id = state['id_jugador']
        n = self.num_jugadors

        obs_cartes = np.zeros(self.OBS_CARTES_SHAPE, dtype=np.float32)

        def _marca_carta(canal, carta_str):
            f, c = self._carta_a_idx(carta_str)
            if f is not None and c is not None:
                obs_cartes[canal, f, c] = 1.0

        for carta in state['ma_jugador']:
            _marca_carta(0, carta)

        canal_per_jugador = {}
        for offset, canal in enumerate([1, 2, 3, 4]):
            pid = (player_id + offset) % n
            canal_per_jugador[pid] = canal

        for entrada in state['hist_cartes']:
            if len(entrada) == 3:
                pid, ronda, carta = entrada
            else:
                pid, carta = entrada
            canal = canal_per_jugador.get(pid)
            if canal is not None:
                _marca_carta(canal, carta)

        company_pid = (player_id + 2) % n
        SENYA_CARTA_MAP = {
            'senya_onze_bastos':    'B11',
            'senya_deu_ors':        'O10',
            'senya_as_espases':     'S1',
            'senya_as_bastos':      'B1',
            'senya_manilla_espases':'S7',
            'senya_manilla_ors':    'O7',
            'senya_tres':           None,
            'senya_as_bord':        None,
            'senya_cegas':          None,
        }
        for entry in state.get('hist_senyes', []):
            if len(entry) == 3:
                pid, ronda, senya = entry
            else:
                pid, senya = entry
            if pid == company_pid:
                carta_senya = SENYA_CARTA_MAP.get(senya)
                if carta_senya:
                    _marca_carta(5, carta_senya)

        obs_context = np.zeros(self.OBS_CONTEXT_SIZE, dtype=np.float32)

        equip_propi = player_id % 2
        equip_rival = 1 - equip_propi

        obs_context[0] = state['puntuacio'][equip_propi] / 24.0
        obs_context[1] = state['puntuacio'][equip_rival] / 24.0
        obs_context[2] = state['estat_truc']['level'] / 24.0
        obs_context[3] = state['estat_envit']['level'] / 24.0
        obs_context[4] = state['fase_torn']
        obs_context[5] = state['comptador_ronda'] / self.cartes_jugador

        ma_offset = (state['ma'] - player_id) % n
        if ma_offset < 4:
            obs_context[6 + ma_offset] = 1.0

        truc_owner = state['estat_truc']['owner']
        if truc_owner != -1:
            truc_offset = (truc_owner - player_id) % n
            if truc_offset < 4:
                obs_context[10 + truc_offset] = 1.0

        envit_owner = state['estat_envit']['owner']
        if envit_owner != -1:
            envit_offset = (envit_owner - player_id) % n
            if envit_offset < 3:
                obs_context[14 + envit_offset] = 1.0

        rondes_jo = sum(1 for w in state['ronda_winners'] if w is not None and w % 2 == equip_propi)
        rondes_rival = sum(1 for w in state['ronda_winners'] if w is not None and w % 2 == equip_rival)
        obs_context[17] = rondes_jo / 3.0
        obs_context[18] = rondes_rival / 3.0

        def _winner_rel(ronda_idx):
            rw = state['ronda_winners']
            if ronda_idx >= len(rw): return 0.0
            w = rw[ronda_idx]
            if w == -1: return 0.0
            return 1.0 if w % 2 == equip_propi else -1.0

        obs_context[19] = _winner_rel(0)
        obs_context[20] = _winner_rel(1)
        obs_context[21] = 1.0 if state['envit_accepted'] else 0.0
        rs_map = {0: 0.0, 1: 0.5, 2: 1.0}
        obs_context[22] = rs_map.get(state['response_state_val'], 0.0)

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
