import numpy as np
from collections import OrderedDict
from rlcard.envs import Env
from entorn.game import TrucGame
from entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, ACTIONS_SIGNAL, PALS, NUMS, init_joc_cartes

# Mapeig de pal i rang a índex de fila/columna en el tensor de cartes (6×4×9)
_PAL_IDX = {p: i for i, p in enumerate(PALS)}   # {'S':0, 'C':1, 'O':2, 'B':3}
_NUM_IDX = {n: i for i, n in enumerate(NUMS)}    # {'1':0,'3':1,'4':2,...,'12':8}

class TrucEnv(Env):
    """
    Entorn del joc del Truc per a RLCard.
    Aquesta classe connecta la lògica del joc (TrucGame) amb la interfície estàndard d'entorns de RLCard.
    """
    def __init__(self, config):
        self.name = 'truc'
        self.num_jugadors = config.get('num_jugadors', 2)
        self.cartes_jugador = config.get('cartes_jugador', 3)
        self.puntuacio_final = config.get('puntuacio_final', 24)
        senyes = config.get('senyes', False)
        verbose = config.get('verbose', False)
        player_class = config.get('player_class', None)

        if player_class:
            self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final,
                                 player_class=player_class,
                                 verbose=verbose)
        else:
             self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final,
                                 verbose=verbose)
        
        config.setdefault('allow_step_back', False)
        config.setdefault('seed', None)
        super().__init__(config)
        
        self.cartes = init_joc_cartes()
        self.carta_map = {carta: i for i, carta in enumerate(self.cartes)}
        self.signal_map = {signal: i for i, signal in enumerate(ACTIONS_SIGNAL)}

        # Format multi-entrada
        # obs_cartes: (6 canals, 4 pals, 9 rangs)
        # obs_context: (17,) — informació contextual
        self.OBS_CARTES_SHAPE = (6, 4, 9)
        self.OBS_CONTEXT_SIZE = 17

        # state_size i state_shape per compatibilitat amb RLCard
        self.state_size = (
            self.OBS_CARTES_SHAPE[0] * self.OBS_CARTES_SHAPE[1] * self.OBS_CARTES_SHAPE[2]
            + self.OBS_CONTEXT_SIZE
        )  # 6*4*9 + 17 = 233
        self.state_shape = [[self.state_size] for _ in range(self.num_jugadors)]
        self.action_shape = [[len(ACTION_LIST)] for _ in range(self.num_jugadors)]

    def _carta_a_idx(self, carta_str):
        pal = carta_str[0]
        rang = carta_str[1:]
        return _PAL_IDX.get(pal), _NUM_IDX.get(rang)

    def _extract_state(self, state):
        """
        Extreu l'estat del joc en format multi-entrada per a la xarxa neuronal

        Retorna un diccionari 'obs' amb:
          - 'obs_cartes'  : np.ndarray (6, 4, 9) float32
          - 'obs_context' : np.ndarray (17,)     float32
        """
        player_id = state['id_jugador']
        n = self.num_jugadors

        # Tensor de cartes (6 canals × 4 pals × 9 rangs)
        obs_cartes = np.zeros(self.OBS_CARTES_SHAPE, dtype=np.float32)

        def _marca_carta(canal, carta_str):
            f, c = self._carta_a_idx(carta_str)
            if f is not None and c is not None:
                obs_cartes[canal, f, c] = 1.0

        # Canal 0: Mà actual del jugador
        for carta in state['ma_jugador']:
            _marca_carta(0, carta)

        # Canals 1-4: Historial de cartes
        # Canal 1 = el propi jugador, Canal 2 = Rival 1, Canal 3 = Company, Canal 4 = Rival 2
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

        # Canal 5: Cartes assenyalades per les senyes del company
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
        for pid, ronda, senya in state.get('hist_senyes', []):
            if pid == company_pid:
                carta_senya = SENYA_CARTA_MAP.get(senya)
                if carta_senya:
                    _marca_carta(5, carta_senya)

        # Vector de context
        obs_context = np.zeros(self.OBS_CONTEXT_SIZE, dtype=np.float32)

        equip_propi = player_id % 2
        equip_rival = 1 - equip_propi

        obs_context[0] = state['puntuacio'][equip_propi] / 24.0
        obs_context[1] = state['puntuacio'][equip_rival] / 24.0
        obs_context[2] = state['estat_truc']['level'] / 24.0
        obs_context[3] = state['estat_envit']['level'] / 24.0
        obs_context[4] = state['fase_torn']                             # max=1 (2 fases: 0 i 1)
        obs_context[5] = state['comptador_ronda'] / self.cartes_jugador  # max=cartes_jugador

        # [6-9] One-hot relatiu: qui és "mà"
        ma_offset = (state['ma'] - player_id) % n
        if ma_offset < 4:
            obs_context[6 + ma_offset] = 1.0

        # [10-13] One-hot relatiu: qui ha cantat el Truc
        truc_owner = state['estat_truc']['owner']
        if truc_owner != -1:
            truc_offset = (truc_owner - player_id) % n
            if truc_offset < 4:
                obs_context[10 + truc_offset] = 1.0

        # [14-16] One-hot relatiu: qui ha cantat l'Envit
        envit_owner = state['estat_envit']['owner']
        if envit_owner != -1:
            envit_offset = (envit_owner - player_id) % n
            if envit_offset < 3:
                obs_context[14 + envit_offset] = 1.0

        # Accions legals
        legal_actions_list = state['accions_legals']
        legal_actions = OrderedDict({a: None for a in legal_actions_list})

        # Estat final per a l'agent
        extracted_state = {
            'obs': {'obs_cartes': obs_cartes, 'obs_context': obs_context},
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': [ACTION_LIST[a] for a in legal_actions_list],
            'action_record': self.action_recorder
        }
        return extracted_state

    def get_payoffs(self):
        score    = self.game.score
        objectiu = self.game.puntuacio_final
        payoffs  = []

        for pid in range(self.num_jugadors):
            oponent = 1 - pid
            diff = np.clip((score[pid] - score[oponent]) / objectiu, -1.0, 1.0)
            payoffs.append(diff)

        return payoffs

    def get_estat_taula(self, player_id):
        """
        Retorna l'estat brut del joc per mostrar la taula
        """
        return self.game.get_state(player_id)

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        return self.game.get_legal_actions()
