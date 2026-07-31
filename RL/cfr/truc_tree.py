"""Esquelet PÚBLIC de l'arbre d'aposta del truc, per comptar information sets
(la porta go/no-go) i com a base del solver.

El truc s'aposta a qualsevol torn de qualsevol base, abans de jugar la carta.
El joc de cartes és determinista (solver), així que entre decisions d'aposta
les cartes s'avancen soles; l'únic que importa públicament és: en quina base
som, qui ha guanyat les bases anteriors (`rw`), quantes cartes s'han jugat en
la base actual, de qui és el torn, i l'estat de l'aposta de truc.

Aquí NOMÉS enumerem l'estructura pública (sense mans ni buckets): a les
transicions de "jugar carta que tanca una base" es ramifica sobre el resultat
possible de la base {0,1,-1} (sobre-aproximació), podant a boards no terminals.
El nombre d'information sets ≈ (#nodes de decisió públics) × (#buckets d'equity).

Estat públic (tupla immutable):
  (phase, trick, rw, play_count, to_act, level, owner, prev_level, resume)
  phase   : 'BET' (to_act pot pujar/plegar/jugar) | 'RESP' (to_act respon)
  trick   : índex de base 0..2
  rw      : tupla de guanyadors de bases completes, valors {0,1,-1}
  play_count: cartes jugades a la base actual (0 o 1)
  to_act  : jugador que decideix
  level   : nivell de truc {1,3,6,9,24}
  owner   : últim que va pujar {-1,0,1}
  prev_level: nivell abans de l'última pujada (per al pagament de fold)
  resume  : jugador que ha de jugar quan es resol una resposta ('BET')
"""
from __future__ import annotations

from joc.entorn.rols.judger import TrucJudger

_JUDGER = TrucJudger(None, n_cartes=3)
MA = 0  # la mà és sempre el jugador 0 en el subjoc per mà

# Accions
JUGAR = 'play'
PUJAR = 'raise'
PLEGAR_JOC = 'concede'   # fora_truc en joc normal: concedeix al nivell actual
ACCEPTAR = 'accept'      # vull_truc
PLEGAR_RESP = 'fold'     # fora_truc responent: paga prev_level


def _seguent_truc(level: int) -> int:
    return {1: 3, 3: 6, 6: 9, 9: 24}[level]


def _lider(trick: int, rw: tuple) -> int:
    if trick == 0:
        return MA
    w = rw[-1]
    return w if w != -1 else MA


def guanyador_ma(rw: tuple) -> int:
    """-1 si la mà encara no està decidida, si no la 0/1 de l'equip guanyador."""
    return _JUDGER.guanyador_ma(list(rw), MA)


def estat_inicial():
    # base 0, ningú ha jugat, lidera la mà; truc a 1 (base), sense owner
    return ('BET', 0, (), 0, _lider(0, ()), 1, -1, 1, None)


def is_terminal(state) -> bool:
    return state[0] in ('T_show', 'T_fold', 'T_concede')


def current_player(state) -> int:
    return state[4]


def legal_actions(state):
    phase, trick, rw, pc, to_act, level, owner, prev, resume = state
    if phase == 'BET':
        acc = [JUGAR]
        if owner != to_act and level < 24:
            acc.append(PUJAR)
        acc.append(PLEGAR_JOC)
        return acc
    if phase == 'RESP':
        acc = [ACCEPTAR, PLEGAR_RESP]
        if level < 24:
            acc.append(PUJAR)
        return acc
    return []


def _avancar_joc(trick, rw, pc, to_act):
    """Retorna la llista de (estat_seguent) després de JUGAR una carta en fase
    BET. Si la carta tanca la base, ramifica sobre el resultat {0,1,-1}."""
    if pc == 0:
        # primera carta de la base: juga l'altre a continuació
        return [('cont', trick, rw, 1, 1 - to_act)]
    # segona carta: la base es tanca; ramifica sobre el guanyador
    seguents = []
    for out in (0, 1, -1):
        nou_rw = rw + (out,)
        if guanyador_ma(nou_rw) != -1:
            seguents.append(('term', nou_rw))          # mà decidida -> showdown
        else:
            nt = trick + 1
            seguents.append(('cont', nt, nou_rw, 0, _lider(nt, nou_rw)))
    return seguents


def successors(state):
    """Retorna [(accio, estat_seguent), ...]. Els estats terminals es marquen
    amb phase 'T_*'. Per a JUGAR que tanca base, hi ha diverses branques (una
    per resultat) amb la mateixa acció JUGAR."""
    phase, trick, rw, pc, to_act, level, owner, prev, resume = state
    out = []
    if phase == 'BET':
        for a in legal_actions(state):
            if a == JUGAR:
                for nxt in _avancar_joc(trick, rw, pc, to_act):
                    if nxt[0] == 'term':
                        out.append((JUGAR, ('T_show', trick, nxt[1], pc, to_act, level, owner, prev, resume)))
                    else:
                        _, nt, nrw, npc, nta = nxt
                        out.append((JUGAR, ('BET', nt, nrw, npc, nta, level, owner, prev, None)))
            elif a == PUJAR:
                out.append((PUJAR, ('RESP', trick, rw, pc, 1 - to_act, _seguent_truc(level), to_act, level, to_act)))
            elif a == PLEGAR_JOC:
                out.append((PLEGAR_JOC, ('T_concede', trick, rw, pc, to_act, level, owner, prev, resume)))
    elif phase == 'RESP':
        for a in legal_actions(state):
            if a == ACCEPTAR:
                # es reprèn el joc: juga qui estava a punt de jugar (resume)
                out.append((ACCEPTAR, ('BET', trick, rw, pc, resume, level, owner, prev, None)))
            elif a == PLEGAR_RESP:
                out.append((PLEGAR_RESP, ('T_fold', trick, rw, pc, to_act, level, owner, prev, resume)))
            elif a == PUJAR:
                out.append((PUJAR, ('RESP', trick, rw, pc, 1 - to_act, _seguent_truc(level), to_act, level, resume)))
    return out


def comptar_nodes_publics():
    """DFS que recull tots els estats públics de DECISIÓ (BET/RESP) assolibles.
    Retorna (nombre de nodes de decisió, desglossament)."""
    vist = set()
    decisio = set()
    pila = [estat_inicial()]
    desglos = {'BET': 0, 'RESP': 0}
    while pila:
        s = pila.pop()
        if s in vist:
            continue
        vist.add(s)
        if is_terminal(s):
            continue
        decisio.add(s)
        desglos[s[0]] += 1
        for _, ns in successors(s):
            if ns not in vist:
                pila.append(ns)
    return len(decisio), desglos


if __name__ == '__main__':
    n_nodes, desglos = comptar_nodes_publics()
    for n_buckets in (8, 12, 16):
        print(f"Nodes de decisió públics: {n_nodes}  (BET={desglos['BET']}, RESP={desglos['RESP']})  "
              f"| × {n_buckets} buckets = {n_nodes * n_buckets} information sets (cota superior)")
