"""Subjoc d'aposta de l'ENVIT com a joc en forma extensiva per a CFR.

L'envit es puntua per la força d'envit de la mà inicial (`get_envit_ma`), NO
per les bases. Per tant l'únic que importa de cada mà és el seu valor d'envit
(0..35, 22 valors diferents). El node d'atzar reparteix les dues mans i en
resumim cadascuna al seu valor d'envit; el showdown és una comparació
determinista (empat → la mà, jugador 0, per `guanyador_envits`).

Interfície de joc (compartida amb el solver CFR i el best-response):
  chance_outcomes()               -> [((e0,e1), prob), ...]
  initial_state()                 -> state
  is_terminal(state)              -> bool
  current_player(state)           -> 0/1
  legal_actions(state)            -> [accions]
  apply(state, action)            -> state'
  info_key(state, chance, player) -> clau hashable de l'information set
  terminal_value(state, chance)   -> valor per al jugador 0 (suma zero)

Estat = tupla immutable `(status, to_act, level, owner, prev_level, extra)`.
`status`: 'P0_OPEN' | 'P1_OPEN' | 'RESPOND' | 'T_pass' | 'T_fold' | 'T_show'.

Estaca de falta = R = 24 − max(marcador). Per defecte 24 (context 0-0).
"""
from __future__ import annotations

import random

from joc.entorn.cartes_accions import init_joc_cartes
from joc.entorn.rols.judger import TrucJudger

# Accions de l'arbre d'aposta d'envit
PASSAR = 'pass'      # no obrir envit
OBRIR = 'open'       # obrir envit (nivell 0 -> 2)
ACCEPTAR = 'accept'  # vull_envit
PLEGAR = 'fold'      # fora_envit
PUJAR = 'raise'      # apostar_envit sobre un envit pendent


def _seguent_nivell(level: int, R: int) -> int:
    """0->2->4->6->falta(R)."""
    return {0: 2, 2: 4, 4: 6, 6: R}[level]


def _pot_pujar(level: int) -> bool:
    """Es pot pujar mentre el nivell és 2/4/6 (6->falta); a la falta, no."""
    return level in (2, 4, 6)


class EnvitGame:
    def __init__(self, R: int = 24):
        self.R = R  # valor de la falta (tots els punts que queden)

    # ---- Atzar: distribució conjunta de (envit0, envit1) ----
    def calcular_chance(self, n_mostres: int = 2_000_000, seed: int = 0):
        """Estima la distribució conjunta P(e0, e1) mostrejant repartiments
        reals (captura correctament la correlació per eliminació de cartes)."""
        judger = TrucJudger(None, n_cartes=3)
        deck = init_joc_cartes()
        rng = random.Random(seed)
        comptes: dict[tuple[int, int], int] = {}
        for _ in range(n_mostres):
            cartes = rng.sample(deck, 6)
            e0 = judger.get_envit_ma(cartes[:3])
            e1 = judger.get_envit_ma(cartes[3:])
            comptes[(e0, e1)] = comptes.get((e0, e1), 0) + 1
        total = float(n_mostres)
        self._chance = [((e0, e1), c / total) for (e0, e1), c in comptes.items()]
        return self._chance

    def chance_outcomes(self):
        return self._chance

    # ---- Arbre d'aposta ----
    def initial_state(self):
        return ('P0_OPEN', 0, 0, -1, 1, None)

    def is_terminal(self, state) -> bool:
        return state[0].startswith('T_')

    def current_player(self, state) -> int:
        return state[1]

    def legal_actions(self, state):
        status = state[0]
        if status in ('P0_OPEN', 'P1_OPEN'):
            return [PASSAR, OBRIR]
        if status == 'RESPOND':
            level = state[2]
            acc = [PLEGAR, ACCEPTAR]
            if _pot_pujar(level):
                acc.append(PUJAR)
            return acc
        return []

    def apply(self, state, action, chance=None):
        status, to_act, level, owner, prev_level, _ = state
        R = self.R

        if status == 'P0_OPEN':
            if action == PASSAR:
                return ('P1_OPEN', 1, 0, -1, 1, None)
            # OBRIR: 0 -> 2, owner 0, respon el jugador 1
            return ('RESPOND', 1, 2, 0, 1, None)

        if status == 'P1_OPEN':
            if action == PASSAR:
                return ('T_pass', 0, 0, -1, 0, None)
            return ('RESPOND', 0, 2, 1, 1, None)

        if status == 'RESPOND':
            if action == PLEGAR:
                # el proposant (owner) guanya prev_level
                return ('T_fold', to_act, level, owner, prev_level, None)
            if action == ACCEPTAR:
                # showdown al nivell actual
                return ('T_show', to_act, level, owner, prev_level, None)
            # PUJAR: nou nivell, jo passo a ser owner, respon l'altre
            nou = _seguent_nivell(level, R)
            return ('RESPOND', 1 - to_act, nou, to_act, level, None)

        raise ValueError(f"apply sobre estat terminal: {state}")

    def info_key(self, state, chance, player):
        e = chance[player]
        # públic: status, level, owner, prev_level, to_act; privat: e
        return (player, e, state[0], state[2], state[3], state[4])

    def terminal_value(self, state, chance):
        """Valor per al jugador 0 (suma zero)."""
        e0, e1 = chance
        status, _, level, owner, prev_level, _ = state
        if status == 'T_pass':
            return 0.0
        if status == 'T_fold':
            # owner guanya prev_level
            return float(prev_level) if owner == 0 else -float(prev_level)
        if status == 'T_show':
            # empat -> mà (jugador 0) guanya
            p0_guanya = e0 >= e1
            return float(level) if p0_guanya else -float(level)
        raise ValueError(f"terminal_value sobre estat no terminal: {state}")
