"""Subjoc d'aposta del TRUC com a joc en forma extensiva per a CFR.

El truc es puntua per guanyar la majoria de bases. El joc de cartes és
determinista (double-dummy), així que un repartiment es resumeix en
`(bucket0, bucket1, rw_full)`:
  - bucket0/bucket1: bucket d'equity (P guanyar majoria) de cada mà.
  - rw_full: seqüència de guanyadors de base sota joc òptim (defineix la
    progressió pública de les bases i el guanyador del showdown).

El node d'atzar mostreja `(bucket0, bucket1, rw_full)` amb la seva probabilitat
(precomputada). Entre decisions d'aposta les cartes s'avancen soles seguint
rw_full. L'estat i les transicions són les de `truc_tree` (esquelet públic),
però PLAY segueix la línia real en lloc de ramificar.

Estaca de "joc fora" = R = 24 − max(marcador). Per defecte 24 (context 0-0).
"""
from __future__ import annotations

from RL.cfr import truc_tree as T


class TrucGame:
    def __init__(self, chance, R: int = 24):
        # chance: llista [((bucket0, bucket1, rw_full), prob), ...]
        self._chance = list(chance)
        self.R = R

    def chance_outcomes(self):
        return self._chance

    def initial_state(self):
        return T.estat_inicial()

    def is_terminal(self, state) -> bool:
        return T.is_terminal(state)

    def current_player(self, state) -> int:
        return T.current_player(state)

    def legal_actions(self, state):
        return T.legal_actions(state)

    def apply(self, state, action, chance):
        phase, trick, rw, pc, to_act, level, owner, prev, resume = state
        rw_full = chance[2]

        if phase == 'BET':
            if action == T.JUGAR:
                if pc == 0:
                    # primera carta: juga l'altre a continuació
                    return ('BET', trick, rw, 1, 1 - to_act, level, owner, prev, None)
                # segona carta: la base es tanca amb el resultat real
                out = rw_full[trick]
                nou_rw = rw + (out,)
                if T.guanyador_ma(nou_rw) != -1:
                    return ('T_show', trick, nou_rw, pc, to_act, level, owner, prev, resume)
                nt = trick + 1
                return ('BET', nt, nou_rw, 0, T._lider(nt, nou_rw), level, owner, prev, None)
            if action == T.PUJAR:
                return ('RESP', trick, rw, pc, 1 - to_act, T._seguent_truc(level), to_act, level, to_act)
            if action == T.PLEGAR_JOC:
                return ('T_concede', trick, rw, pc, to_act, level, owner, prev, resume)

        elif phase == 'RESP':
            if action == T.ACCEPTAR:
                return ('BET', trick, rw, pc, resume, level, owner, prev, None)
            if action == T.PLEGAR_RESP:
                return ('T_fold', trick, rw, pc, to_act, level, owner, prev, resume)
            if action == T.PUJAR:
                return ('RESP', trick, rw, pc, 1 - to_act, T._seguent_truc(level), to_act, level, resume)

        raise ValueError(f"apply invàlid: {state} / {action}")

    def info_key(self, state, chance, player):
        bucket = chance[0] if player == 0 else chance[1]
        # tot l'estat és públic; el privat és el bucket del jugador
        return (player, bucket, state)

    def terminal_value(self, state, chance):
        """Valor per al jugador 0 (suma zero)."""
        status, trick, rw, pc, to_act, level, owner, prev, resume = state
        if status == 'T_show':
            w = T.guanyador_ma(rw)  # 0/1
            return float(level) if w == 0 else -float(level)
        if status == 'T_fold':
            # el proposant (owner) guanya prev_level
            return float(prev) if owner == 0 else -float(prev)
        if status == 'T_concede':
            # to_act concedeix: l'altre equip guanya el nivell actual
            guanyador = 1 - to_act
            return float(level) if guanyador == 0 else -float(level)
        raise ValueError(f"terminal_value sobre estat no terminal: {state}")
