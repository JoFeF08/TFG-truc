"""Equity de truc d'una mà = P(guanyar la majoria de bases) i el seu bucket,
i el double-dummy (guanyador + línia de bases) d'un repartiment concret.

- L'equity (per bucketejar l'information set) es calcula com la mitjana del
  resultat double-dummy sobre mans rivals mostrejades (prou per bucketejar).
- El double-dummy d'un repartiment concret (per als valors terminals i la
  progressió pública `rw_full`) es calcula exacte amb minimax d'informació
  perfecta, reaprofitant TrucJudger.

NOMÉS força de carta per resoldre bases (via TrucJudger); MAI suma de forces
com a indicador de força de mà — l'equity és P(guanyar majoria).
"""
from __future__ import annotations

import os
import pickle
import random
from functools import lru_cache

from joc.entorn.cartes_accions import init_joc_cartes
from joc.entorn.rols.judger import TrucJudger

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

MA = 0
_JUDGER = TrucJudger(None, n_cartes=3)
_DECK = init_joc_cartes()


# ---------- double-dummy exacte d'un repartiment ----------

@lru_cache(maxsize=None)
def _dd(hands_key, rw, to_play, taula):
    """Minimax d'informació perfecta. `hands_key` = (tuple mà0, tuple mà1).
    Retorna el valor per a l'equip 0 (+1/-1). Memoitzat: la clau conté les dues
    mans senceres i tot l'estat, així que el cache global és segur."""
    w = _JUDGER.guanyador_ma(list(rw), MA)
    if w != -1:
        return 1.0 if w == 0 else -1.0
    hands = hands_key
    best = None
    for card in hands[to_play]:
        noves = list(hands)
        noves[to_play] = tuple(c for c in hands[to_play] if c != card)
        nova_taula = taula + ((to_play, card),)
        if len(nova_taula) == 2:
            g = _JUDGER.guanyador_ronda(list(nova_taula))
            nou_rw = rw + (g if g is not None else -1,)
            proper = g if g is not None else MA
            v = _dd(tuple(noves), nou_rw, proper, ())
        else:
            v = _dd(tuple(noves), rw, 1 - to_play, nova_taula)
        if to_play == 0:
            best = v if best is None else max(best, v)
        else:
            best = v if best is None else min(best, v)
    return best


def double_dummy_line(hand0, hand1):
    """Retorna (guanyador_equip, rw_full) sota joc òptim double-dummy per les
    dues bandes. rw_full = seqüència de guanyadors de base fins que es decideix
    la mà. Desempat de cartes: força més baixa (canònic i determinista)."""
    hands = (tuple(hand0), tuple(hand1))
    rw = ()
    to_play = MA
    taula = ()
    while _JUDGER.guanyador_ma(list(rw), MA) == -1:
        # tria la carta òptima per a l'equip de to_play (desempat: menys forta)
        millor_card, millor_val = None, None
        for card in sorted(hands[to_play], key=_JUDGER.get_forca_carta):
            noves = list(hands)
            noves[to_play] = tuple(c for c in hands[to_play] if c != card)
            nova_taula = taula + ((to_play, card),)
            if len(nova_taula) == 2:
                g = _JUDGER.guanyador_ronda(list(nova_taula))
                nou_rw = rw + (g if g is not None else -1,)
                proper = g if g is not None else MA
                v = _dd(tuple(noves), nou_rw, proper, ())
            else:
                v = _dd(tuple(noves), rw, 1 - to_play, nova_taula)
            millor = (millor_val is None
                      or (to_play == 0 and v > millor_val)
                      or (to_play == 1 and v < millor_val))
            if millor:
                millor_card, millor_val = card, v
        # aplica la carta triada
        noves = list(hands)
        noves[to_play] = tuple(c for c in hands[to_play] if c != millor_card)
        hands = tuple(noves)
        taula = taula + ((to_play, millor_card),)
        if len(taula) == 2:
            g = _JUDGER.guanyador_ronda(list(taula))
            rw = rw + (g if g is not None else -1,)
            to_play = g if g is not None else MA
            taula = ()
        else:
            to_play = 1 - to_play
    return _JUDGER.guanyador_ma(list(rw), MA), rw


# ---------- equity d'una mà (per bucketejar) ----------

def calcular_taula_equity(n_rivals: int = 400, seed: int = 0):
    """equity(mà) = fracció de mans rivals amb qui l'equip 0 guanya la majoria
    (double-dummy). Mostreja n_rivals mans rivals per cada mà. Retorna
    {tuple(sorted(hand)): equity}. Enumerar les 7140 mans."""
    from itertools import combinations
    rng = random.Random(seed)
    taula = {}
    hands = list(combinations(_DECK, 3))
    for h in hands:
        resta = [c for c in _DECK if c not in h]
        wins = 0
        for _ in range(n_rivals):
            opp = tuple(rng.sample(resta, 3))
            g, _rw = double_dummy_line(h, opp)
            if g == 0:
                wins += 1
        taula[tuple(sorted(h))] = wins / n_rivals
    return taula


def fer_buckets(taula_equity, n_buckets: int = 12):
    """Assigna cada mà a un bucket uniforme sobre [0,1] de la seva equity."""
    def bucket(eq):
        b = int(eq * n_buckets)
        return min(b, n_buckets - 1)
    return {h: bucket(eq) for h, eq in taula_equity.items()}


# ---------- cache a disc ----------

def calcular_i_desar_buckets(n_rivals: int = 300, n_buckets: int = 12, seed: int = 0):
    taula = calcular_taula_equity(n_rivals=n_rivals, seed=seed)
    buckets = fer_buckets(taula, n_buckets=n_buckets)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, 'buckets.pkl'), 'wb') as f:
        pickle.dump({'buckets': buckets, 'equity': taula, 'n_buckets': n_buckets,
                     'n_rivals': n_rivals}, f)
    return buckets, taula


def carregar_buckets():
    with open(os.path.join(CACHE_DIR, 'buckets.pkl'), 'rb') as f:
        return pickle.load(f)


def construir_chance_truc(bucket_table, n_deals: int = 500_000, seed: int = 1):
    """Mostreja repartiments reals i construeix la distribució abstracta
    P(bucket0, bucket1, rw_full) per al node d'atzar del joc de truc."""
    rng = random.Random(seed)
    comptes: dict = {}
    for _ in range(n_deals):
        cartes = rng.sample(_DECK, 6)
        b0 = bucket_table[tuple(sorted(cartes[:3]))]
        b1 = bucket_table[tuple(sorted(cartes[3:]))]
        _, rw_full = double_dummy_line(cartes[:3], cartes[3:])
        k = (b0, b1, rw_full)
        comptes[k] = comptes.get(k, 0) + 1
    total = float(n_deals)
    return [(k, c / total) for k, c in comptes.items()]


if __name__ == '__main__':
    import sys
    import time
    from collections import Counter
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t = time.time()
    buckets, taula = calcular_i_desar_buckets(n_rivals=nr)
    c = Counter(buckets.values())
    eqs = sorted(taula.values())
    print(f"Taula equity+buckets desada ({len(buckets)} mans, {nr} rivals) en {time.time()-t:.0f}s")
    print(f"equity: min={eqs[0]:.3f} max={eqs[-1]:.3f} mediana={eqs[len(eqs)//2]:.3f} "
          f"mitjana={sum(eqs)/len(eqs):.3f}")
    print("distribució de buckets:", dict(sorted(c.items())))

