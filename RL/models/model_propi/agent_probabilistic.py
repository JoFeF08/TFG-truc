"""Agent que calcula (no aprèn ni segueix llindars fets a mà) el joc
gairebé òptim per al subjoc de cartes del Truc sense apostes
(`permetre_apostes=False`, l'etapa "cartes" del currículum).

TÈCNICA: determinització exhaustiva (PIMC — perfect information Monte
Carlo, la mateixa idea que els solvers "double-dummy" de bridge/skat):
per a cada carta candidata, s'enumeren TOTES les mans que podria tenir
el rival donat el que ja s'ha vist, es resol cada hipòtesi amb minimax
d'informació perfecta, i es fa la mitjana. **No és un equilibri Bayesià/
CFR real** — dins de cada hipòtesi es tracta el rival com si conegués la
meva mà exacta. És una estimació "de seguretat"/near-optimal, vàlida i
honesta com a benchmark, perquè en el subjoc de cartes soles no hi ha
cap manera d'enganyar el rival triant una carta (les cartes es revelen
en jugar-se; no hi ha farol real, a diferència del truc/envit).

REUTILITZACIÓ FUTURA (CFR): `_minimax` calcula exactament el "valor de
continuació un cop les cartes es juguen" que un futur solver CFR del joc
complet (amb apostes/farol de veritat) necessitaria com a valor terminal
pel seu propi arbre de cerca de l'aposta — CFR només hauria de resoldre
l'arbre d'apostes, i podria cridar aquesta mateixa funció per avaluar què
passa un cop decidit el nivell de truc/envit i només queda jugar les
cartes. No és feina aïllada del subjoc actual.

Limitacions conegudes:
- Només pensat per a 2 jugadors i el subjoc sense apostes. Si alguna
  acció d'aposta (o de senyes) és legal, delega tota la decisió a un
  `AgentRegles` intern.
- v1: enumeració per identitat exacta de carta (no canonicalitzada per
  força). Correcte i prou ràpid per avaluació puntual (~300k nodes a la
  primera decisió, sub-segon a pocs segons); si mai cal fer-lo servir
  com a oponent massiu d'entrenament, caldria canonicalitzar per força
  (moltes cartes empaten en força i només la força importa aquí) per
  reduir la combinatòria un ordre de magnitud.
"""
from __future__ import annotations

import random
from itertools import combinations

from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, init_joc_cartes
from joc.entorn.rols.judger import TrucJudger
from RL.models.model_propi.agent_regles import AgentRegles

_PLAY_CARD_NAMES = {'play_card_0', 'play_card_1', 'play_card_2'}


class AgentProbabilistic:
    """Determinització exhaustiva + minimax per al subjoc de cartes sense apostes."""

    use_raw = False

    def __init__(self, seed=None, n_cartes: int = 3):
        self.rng = random.Random(seed)
        self.n_cartes = n_cartes
        self._judger = TrucJudger(None, n_cartes=n_cartes)
        self._fallback = AgentRegles(seed=seed, estil='equilibrat')
        self._cache: dict = {}

    def step(self, state):
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        legal = list(state['legal_actions'].keys())

        if any(ACTION_LIST[a] not in _PLAY_CARD_NAMES for a in legal):
            # Fora de l'abast (apostes/senyes actives): delega a AgentRegles.
            return self._fallback.eval_step(state)

        valors = self.avaluar_opcions(state)
        millor_valor = max(valors.values())
        millors_cartes = [c for c, v in valors.items() if abs(v - millor_valor) <= 1e-9]

        own_hand = list(state['raw_obs']['ma_jugador'])
        carta_triada = self.rng.choice(millors_cartes)
        idx = own_hand.index(carta_triada)
        action = ACTION_SPACE[f'play_card_{idx}']
        if action not in legal:
            action = legal[0]
        return action, {'valor': millor_valor, 'valors_totes': valors}

    def avaluar_opcions(self, state) -> dict:
        """Retorna {carta: valor_esperat} per a totes les cartes jugables a
        la mà actual (no només la millor) -- útil per analitzar decisions
        d'altres agents comparant-les amb el valor gairebé òptim d'aquest
        solver."""
        raw = state['raw_obs']
        root = raw['id_jugador']
        opp = 1 - root

        own_played = [c for p, c in raw['hist_cartes'] if p == root]
        opp_played = [c for p, c in raw['hist_cartes'] if p == opp]
        own_hand = list(raw['ma_jugador'])
        own_full = own_hand + own_played

        deck = init_joc_cartes()
        unseen = [c for c in deck if c not in own_full and c not in opp_played]
        opp_n = self.n_cartes - len(opp_played)

        ronda_winners = list(raw['ronda_winners'])
        ma = raw['ma']
        taula = list(raw['cartes_taula_actual'])

        self._cache = {}

        valors: dict = {}
        for card in own_hand:
            nova_ma = [c for c in own_hand if c != card]
            nova_taula = taula + [(root, card)]

            total = 0.0
            n_hip = 0
            for combo in combinations(unseen, opp_n):
                hands = {root: nova_ma, opp: list(combo)}
                if len(nova_taula) == 2:
                    guanyador = self._judger.guanyador_ronda(nova_taula)
                    nou_rw = ronda_winners + [guanyador if guanyador is not None else -1]
                    proper = guanyador if guanyador is not None else ma
                    valor = self._minimax(hands, nou_rw, proper, [], ma, root)
                else:
                    valor = self._minimax(hands, ronda_winners, opp, nova_taula, ma, root)
                total += valor
                n_hip += 1

            valors[card] = total / n_hip if n_hip else 0.0

        return valors

    def _minimax(self, hands, ronda_winners, turn_player, taula, ma, root):
        clau = (
            tuple(sorted(hands[root])), tuple(sorted(hands[1 - root])),
            tuple(ronda_winners), turn_player, tuple(taula),
        )
        if clau in self._cache:
            return self._cache[clau]

        guanyador_ma = self._judger.guanyador_ma(ronda_winners, ma)
        if guanyador_ma != -1:
            valor = 1.0 if guanyador_ma == root % 2 else -1.0
            self._cache[clau] = valor
            return valor

        millor = None
        for card in hands[turn_player]:
            noves_mans = dict(hands)
            noves_mans[turn_player] = [c for c in hands[turn_player] if c != card]
            nova_taula = taula + [(turn_player, card)]

            if len(nova_taula) == 2:
                guanyador = self._judger.guanyador_ronda(nova_taula)
                nou_rw = ronda_winners + [guanyador if guanyador is not None else -1]
                proper = guanyador if guanyador is not None else ma
                valor = self._minimax(noves_mans, nou_rw, proper, [], ma, root)
            else:
                valor = self._minimax(noves_mans, ronda_winners, 1 - turn_player, nova_taula, ma, root)

            if turn_player == root:
                millor = valor if millor is None else max(millor, valor)
            else:
                millor = valor if millor is None else min(millor, valor)

        self._cache[clau] = millor
        return millor
