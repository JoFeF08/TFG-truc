"""Agent que juga el joc complet combinant:
  - APOSTES (truc/envit): estratègia d'equilibri del CFR tabular (RL/cfr).
  - CARTES: joc quasi-òptim del solver (AgentProbabilistic).

Mapeja l'estat real del joc (interleaved) a les claus dels dos subjocs
d'aposta separats que va resoldre el CFR, mostreja l'acció d'aposta de la
política mitjana, i delega la tria de carta al solver.

Mateixa interfície que AgentProbabilistic/AgentRegles: eval_step(state) ->
(action_id, info).
"""
from __future__ import annotations

import os
import pickle
import random

from joc.entorn.cartes_accions import ACTION_SPACE
from joc.entorn.rols.judger import TrucJudger
from RL.models.model_propi.agent_probabilistic import AgentProbabilistic
from RL.cfr import envit_game as EG
from RL.cfr import truc_tree as TT
from RL.cfr.truc_equity import carregar_buckets

_POLICIES = os.path.join(os.path.dirname(__file__), '..', '..', 'cfr', 'policies')

_PREV_ENVIT = {2: 1, 4: 2, 6: 4}   # nivell -> nivell anterior (falta -> 6)
_PREV_TRUC = {1: 1, 3: 1, 6: 3, 9: 6, 24: 9}

_CACHE: dict = {}   # polítiques + buckets carregats un sol cop


def _carregar(policies_dir):
    if policies_dir not in _CACHE:
        with open(os.path.join(policies_dir, 'envit.pkl'), 'rb') as f:
            envit = pickle.load(f)['avg']
        with open(os.path.join(policies_dir, 'truc.pkl'), 'rb') as f:
            truc = pickle.load(f)['avg']
        buckets = carregar_buckets()['buckets']
        _CACHE[policies_dir] = (envit, truc, buckets)
    return _CACHE[policies_dir]


class AgentCFR:
    use_raw = False

    def __init__(self, policies_dir: str = _POLICIES, seed=None):
        self._envit, self._truc, self._buckets = _carregar(policies_dir)
        self._judger = TrucJudger(None, n_cartes=3)
        self._eg = EG.EnvitGame(R=24)
        self.rng = random.Random(seed)
        self.solver = AgentProbabilistic(seed=seed)

    # ---------- utilitats ----------
    def reset(self):
        if hasattr(self.solver, 'reset'):
            self.solver.reset()

    def _ma_inicial(self, raw, me):
        jugades = [e[-1] for e in raw['hist_cartes'] if e[0] == me]
        return list(raw['ma_jugador']) + jugades

    def _mostrejar(self, avg, key, legals, default):
        d = avg.get(key)
        if not d:
            return default
        parells = [(a, d.get(a, 0.0)) for a in legals]
        s = sum(p for _, p in parells)
        if s <= 0:
            return default
        r = self.rng.random() * s
        c = 0.0
        for a, p in parells:
            c += p
            if r <= c:
                return a
        return parells[-1][0]

    def _carta(self, state):
        """Delega la tria de carta al solver (evita el seu fallback cridant
        avaluar_opcions directament)."""
        raw = state['raw_obs']
        valors = self.solver.avaluar_opcions(state)
        millor = max(valors.values())
        candidates = [c for c, v in valors.items() if abs(v - millor) <= 1e-9]
        carta = self.rng.choice(candidates)
        idx = list(raw['ma_jugador']).index(carta)
        return ACTION_SPACE[f'play_card_{idx}']

    # ---------- decisions ----------
    def eval_step(self, state):
        raw = state['raw_obs']
        legal = set(state['legal_actions'].keys())
        me = raw['id_jugador']
        rsv = raw['response_state_val']

        if rsv == 2:
            return self._respondre_envit(raw, legal, me), {}
        if rsv == 1:
            return self._respondre_truc(raw, legal, me), {}

        # torn normal: primer decidim obrir envit (si es pot), després truc/carta
        if ACTION_SPACE['apostar_envit'] in legal:
            a = self._obrir_envit(raw, me)
            if a is not None:
                return a, {}
        return self._torn_truc(state, raw, legal, me), {}

    def _envit_pts(self, raw, me):
        return self._judger.get_envit_ma(self._ma_inicial(raw, me))

    def _respondre_envit(self, raw, legal, me):
        level = raw['estat_envit']['level']
        owner = raw['estat_envit']['owner']
        prev = _PREV_ENVIT.get(level, 6)
        e = self._envit_pts(raw, me)
        key = (me, e, 'RESPOND', level, owner, prev)
        st = ('RESPOND', me, level, owner, prev, None)
        legals = self._eg.legal_actions(st)
        a = self._mostrejar(self._envit, key, legals, EG.ACCEPTAR)
        mapa = {EG.ACCEPTAR: 'vull_envit', EG.PLEGAR: 'fora_envit', EG.PUJAR: 'apostar_envit'}
        return self._legalitzar(ACTION_SPACE[mapa[a]], legal)

    def _obrir_envit(self, raw, me):
        ma = raw['ma']
        status = 'P0_OPEN' if me == ma else 'P1_OPEN'
        e = self._envit_pts(raw, me)
        key = (me, e, status, 0, -1, 1)
        a = self._mostrejar(self._envit, key, [EG.PASSAR, EG.OBRIR], EG.PASSAR)
        if a == EG.OBRIR:
            return ACTION_SPACE['apostar_envit']
        return None

    def _truc_state(self, raw, me, phase):
        trick = raw['comptador_ronda']
        rw = tuple(raw['ronda_winners'])
        pc = len(raw.get('cartes_taula_actual', []))
        level = raw['estat_truc']['level']
        owner = raw['estat_truc']['owner']
        prev = _PREV_TRUC.get(level, 1)
        resume = owner if phase == 'RESP' else None
        return (phase, trick, rw, pc, me, level, owner, prev, resume)

    def _respondre_truc(self, raw, legal, me):
        bucket = self._buckets[tuple(sorted(self._ma_inicial(raw, me)))]
        st = self._truc_state(raw, me, 'RESP')
        key = (me, bucket, st)
        legals = TT.legal_actions(st)
        a = self._mostrejar(self._truc, key, legals, TT.ACCEPTAR)
        mapa = {TT.ACCEPTAR: 'vull_truc', TT.PLEGAR_RESP: 'fora_truc', TT.PUJAR: 'apostar_truc'}
        return self._legalitzar(ACTION_SPACE[mapa[a]], legal)

    def _torn_truc(self, state, raw, legal, me):
        bucket = self._buckets[tuple(sorted(self._ma_inicial(raw, me)))]
        st = self._truc_state(raw, me, 'BET')
        key = (me, bucket, st)
        legals = TT.legal_actions(st)
        a = self._mostrejar(self._truc, key, legals, TT.JUGAR)
        if a == TT.PUJAR and ACTION_SPACE['apostar_truc'] in legal:
            return ACTION_SPACE['apostar_truc']
        if a == TT.PLEGAR_JOC and ACTION_SPACE['fora_truc'] in legal:
            return ACTION_SPACE['fora_truc']
        return self._carta(state)  # JUGAR (o fallback)

    def _legalitzar(self, action, legal):
        return action if action in legal else next(iter(legal))

    def step(self, state):
        a, _ = self.eval_step(state)
        return a
