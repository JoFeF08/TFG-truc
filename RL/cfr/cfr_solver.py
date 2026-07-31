"""CFR+ vanilla (amb enumeració exacta de l'atzar) i best-response/exploitability
exactes, genèrics sobre la interfície de joc de `envit_game`/`truc_game`.

Un "joc" ha d'exposar:
  chance_outcomes()            -> [(chance, prob), ...]
  initial_state()              -> state
  is_terminal(state)           -> bool
  current_player(state)        -> 0/1
  legal_actions(state)         -> [accions]
  apply(state, action)         -> state'
  info_key(state, chance, p)   -> clau hashable
  terminal_value(state, chance)-> valor per al jugador 0 (suma zero)

CFR+ = regret-matching-plus (regrets no negatius) + mitjana lineal (pes = t).
L'atzar s'enumera exacte cada iteració, així que l'exploitability calculada
és exacta per al joc (abstracte) donat.
"""
from __future__ import annotations

from collections import defaultdict


def regret_matching_plus(regret_key: dict, actions):
    pos = [regret_key.get(a, 0.0) for a in actions]
    s = sum(pos)
    if s > 0:
        return [p / s for p in pos]
    return [1.0 / len(actions)] * len(actions)


def _cfr(game, state, chance, w, reach0, reach1, regrets, strat_sum, t):
    """CFR+ amb updates simultanis (regret-matching-plus + mitjana lineal).
    Retorna el valor per al jugador 0."""
    if game.is_terminal(state):
        return game.terminal_value(state, chance)
    p = game.current_player(state)
    key = game.info_key(state, chance, p)
    actions = game.legal_actions(state)
    strat = regret_matching_plus(regrets[key], actions)

    reach_p = reach0 if p == 0 else reach1
    ss = strat_sum[key]
    for i, a in enumerate(actions):
        ss[a] += t * w * reach_p * strat[i]

    util_a = {}
    node_util = 0.0
    for i, a in enumerate(actions):
        child = game.apply(state, a, chance)
        if p == 0:
            u = _cfr(game, child, chance, w, reach0 * strat[i], reach1, regrets, strat_sum, t)
        else:
            u = _cfr(game, child, chance, w, reach0, reach1 * strat[i], regrets, strat_sum, t)
        util_a[a] = u
        node_util += strat[i] * u

    cf = w * (reach1 if p == 0 else reach0)
    sign = 1.0 if p == 0 else -1.0
    node_u_p = sign * node_util
    reg = regrets[key]
    for a in actions:
        reg[a] = max(reg.get(a, 0.0) + cf * (sign * util_a[a] - node_u_p), 0.0)
    return node_util


def entrenar_cfr(game, iteracions: int, log_cada: int = 0):
    """Executa CFR+ i retorna (average_strategy, regrets, strat_sum).
    average_strategy: {key: {accio: prob}}."""
    regrets = defaultdict(dict)
    strat_sum = defaultdict(lambda: defaultdict(float))
    chance = game.chance_outcomes()

    for t in range(1, iteracions + 1):
        for ch, w in chance:
            _cfr(game, game.initial_state(), ch, w, 1.0, 1.0, regrets, strat_sum, t)
        if log_cada and (t % log_cada == 0 or t == 1):
            avg = _mitjana(strat_sum)
            expl = exploitability(game, avg)
            print(f"  iter {t:5d} | exploitability = {expl:.5f} pts/mà")

    return _mitjana(strat_sum), regrets, strat_sum


def _mitjana(strat_sum):
    avg = {}
    for key, ss in strat_sum.items():
        tot = sum(ss.values())
        if tot > 0:
            avg[key] = {a: v / tot for a, v in ss.items()}
        else:
            n = len(ss)
            avg[key] = {a: 1.0 / n for a in ss}
    return avg


# ---------------- Best response exacte / exploitability ----------------

def _estrategia_rival(avg, key, actions):
    d = avg.get(key)
    if d is None:
        return [1.0 / len(actions)] * len(actions)
    return [d.get(a, 0.0) for a in actions]


def _valor_avall(game, state, chance, br_player, avg, br_policy):
    """Valor per a br_player: br_player juga br_policy, el rival juga avg."""
    if game.is_terminal(state):
        v0 = game.terminal_value(state, chance)
        return v0 if br_player == 0 else -v0
    p = game.current_player(state)
    actions = game.legal_actions(state)
    if p == br_player:
        a = br_policy[game.info_key(state, chance, p)]
        return _valor_avall(game, game.apply(state, a, chance), chance, br_player, avg, br_policy)
    probs = _estrategia_rival(avg, game.info_key(state, chance, p), actions)
    return sum(pr * _valor_avall(game, game.apply(state, a, chance), chance, br_player, avg, br_policy)
               for pr, a in zip(probs, actions) if pr > 0)


def _recollir_nodes_br(game, state, chance, reach, depth, br_player, avg, nodes):
    if game.is_terminal(state):
        return
    p = game.current_player(state)
    actions = game.legal_actions(state)
    if p == br_player:
        key = game.info_key(state, chance, p)
        nodes.setdefault(key, []).append((state, chance, reach, depth))
        for a in actions:
            _recollir_nodes_br(game, game.apply(state, a, chance), chance, reach, depth + 1, br_player, avg, nodes)
    else:
        probs = _estrategia_rival(avg, game.info_key(state, chance, p), actions)
        for pr, a in zip(probs, actions):
            if pr > 0:
                _recollir_nodes_br(game, game.apply(state, a, chance), chance, reach * pr, depth + 1, br_player, avg, nodes)


def best_response_value(game, br_player, avg):
    """Valor òptim que br_player pot obtenir contra `avg` (rival), exacte."""
    # 1) recollir tots els nodes de decisió de br_player, amb reach rival*atzar
    nodes: dict = {}
    for ch, w in game.chance_outcomes():
        _recollir_nodes_br(game, game.initial_state(), ch, w, 0, br_player, avg, nodes)

    # 2) decidir la br_policy per information set, dels més profunds als menys
    br_policy: dict = {}
    for key in sorted(nodes, key=lambda k: -max(d for *_, d in nodes[k])):
        millor_a, millor_v = None, None
        # cfv[a] = sum sobre nodes de reach * valor_avall(apply(state,a))
        cfv = {}
        for state, chance, reach, _ in nodes[key]:
            for a in game.legal_actions(state):
                v = _valor_avall(game, game.apply(state, a, chance), chance, br_player, avg, br_policy)
                cfv[a] = cfv.get(a, 0.0) + reach * v
        for a, v in cfv.items():
            if millor_v is None or v > millor_v:
                millor_a, millor_v = a, v
        br_policy[key] = millor_a

    # 3) valor final des de l'arrel
    return sum(w * _valor_avall(game, game.initial_state(), ch, br_player, avg, br_policy)
               for ch, w in game.chance_outcomes())


def exploitability(game, avg):
    """NashConv/2 en punts/mà: com de lluny està `avg` de l'equilibri.
    0 = Nash exacte del joc (abstracte)."""
    v0 = best_response_value(game, 0, avg)
    v1 = best_response_value(game, 1, avg)
    return (v0 + v1) / 2.0
