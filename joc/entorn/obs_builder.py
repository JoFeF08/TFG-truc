"""Font única de veritat per a la construcció de l'observació del Truc.

Tant `TrucEnv` (joc/entorn) com `TrucEnvMa` (joc/entorn_ma) han de cridar
`extract_obs()` per evitar que la lògica d'observació es dupliqui i dérivi
(com passava abans: dues còpies gairebé idèntiques amb els mateixos bugs).

Format de l'observació:
  - obs_cartes: tensor (C, 4, 9) amb C = 1 (mà pròpia) + num_jugadors (cartes
    jugades per cada jugador, en offset relatiu a qui observa) + 1 si senyes
    (cartes assenyalades pel company). Files=pal, columnes=rang.
  - obs_context: vector pla de 17 dimensions, sense slots morts, amb
    "propietari" de truc/envit codificat en relatiu-d'equip (+1 el meu
    equip, -1 el rival, 0 ningú) en lloc d'un one-hot per jugador.
"""

import numpy as np

from joc.entorn.cartes_accions import PALS, NUMS

_PAL_IDX = {p: i for i, p in enumerate(PALS)}
_NUM_IDX = {n: i for i, n in enumerate(NUMS)}

OBS_CONTEXT_SIZE = 17

SENYA_CARTA_MAP = {
    'senya_onze_bastos':     'B11',
    'senya_deu_ors':         'O10',
    'senya_as_espases':      'S1',
    'senya_as_bastos':       'B1',
    'senya_manilla_espases': 'S7',
    'senya_manilla_ors':     'O7',
    'senya_tres':            None,
    'senya_as_bord':         None,
    'senya_cegas':           None,
}


def _carta_a_idx(carta_str):
    pal = carta_str[0]
    rang = carta_str[1:]
    return _PAL_IDX.get(pal), _NUM_IDX.get(rang)


def n_card_channels(num_jugadors: int, senyes: bool) -> int:
    """Nombre de canals del tensor de cartes: mà pròpia + 1 per jugador + senyes."""
    return 1 + num_jugadors + (1 if senyes else 0)


def obs_shapes(num_jugadors: int, senyes: bool):
    """Retorna (obs_cartes_shape, obs_context_size) per a la configuració donada."""
    return (n_card_channels(num_jugadors, senyes), 4, 9), OBS_CONTEXT_SIZE


def extract_obs(state: dict, num_jugadors: int, cartes_jugador: int, senyes: bool):
    """Construeix (obs_cartes, obs_context) a partir de l'estat brut del joc."""
    player_id = state['id_jugador']
    n = num_jugadors

    cartes_shape, context_size = obs_shapes(num_jugadors, senyes)
    obs_cartes = np.zeros(cartes_shape, dtype=np.float32)

    def _marca_carta(canal, carta_str):
        f, c = _carta_a_idx(carta_str)
        if f is not None and c is not None:
            obs_cartes[canal, f, c] = 1.0

    # Canal 0: mà pròpia
    for carta in state['ma_jugador']:
        _marca_carta(0, carta)

    # Canals 1..n: cartes jugades per cada jugador, en offset relatiu a qui observa.
    # Cada offset 0..n-1 correspon a exactament un jugador (sense col·lisions).
    canal_per_jugador = {(player_id + offset) % n: 1 + offset for offset in range(n)}

    for entrada in state['hist_cartes']:
        if len(entrada) == 3:
            pid, _ronda, carta = entrada
        else:
            pid, carta = entrada
        canal = canal_per_jugador.get(pid)
        if canal is not None:
            _marca_carta(canal, carta)

    # Últim canal (només si senyes): cartes assenyalades pel company
    if senyes:
        canal_senyes = 1 + n
        company_pid = (player_id + 2) % n
        for entry in state.get('hist_senyes', []):
            if len(entry) == 3:
                pid, _ronda, senya = entry
            else:
                pid, senya = entry
            if pid == company_pid:
                carta_senya = SENYA_CARTA_MAP.get(senya)
                if carta_senya:
                    _marca_carta(canal_senyes, carta_senya)

    # Vector de context (17 dims, cap slot mort)
    obs_context = np.zeros(context_size, dtype=np.float32)

    equip_propi = player_id % 2
    equip_rival = 1 - equip_propi

    obs_context[0] = state['puntuacio'][equip_propi] / 24.0
    obs_context[1] = state['puntuacio'][equip_rival] / 24.0
    obs_context[2] = state['estat_truc']['level'] / 24.0
    obs_context[3] = state['estat_envit']['level'] / 24.0

    def _owner_equip_relatiu(owner_pid):
        if owner_pid == -1:
            return 0.0
        return 1.0 if owner_pid % 2 == equip_propi else -1.0

    obs_context[4] = _owner_equip_relatiu(state['estat_truc']['owner'])
    obs_context[5] = _owner_equip_relatiu(state['estat_envit']['owner'])
    obs_context[6] = 1.0 if state['envit_accepted'] else 0.0

    rs_val = state['response_state_val']
    obs_context[7] = 1.0 if rs_val == 1 else 0.0  # TRUC_PENDING
    obs_context[8] = 1.0 if rs_val == 2 else 0.0  # ENVIT_PENDING

    obs_context[9] = 1.0 if state['ma'] == player_id else 0.0
    obs_context[10] = 1.0 if state['ma'] % 2 == equip_propi else 0.0

    obs_context[11] = state['comptador_ronda'] / cartes_jugador

    rondes_jo = sum(1 for w in state['ronda_winners'] if w is not None and w != -1 and w % 2 == equip_propi)
    rondes_rival = sum(1 for w in state['ronda_winners'] if w is not None and w != -1 and w % 2 == equip_rival)
    obs_context[12] = rondes_jo / cartes_jugador
    obs_context[13] = rondes_rival / cartes_jugador

    def _winner_rel(ronda_idx):
        rw = state['ronda_winners']
        if ronda_idx >= len(rw):
            return 0.0
        w = rw[ronda_idx]
        if w == -1:
            return 0.0
        return 1.0 if w % 2 == equip_propi else -1.0

    obs_context[14] = _winner_rel(0)
    obs_context[15] = _winner_rel(1)
    obs_context[16] = float(state['fase_torn'])

    return obs_cartes, obs_context
