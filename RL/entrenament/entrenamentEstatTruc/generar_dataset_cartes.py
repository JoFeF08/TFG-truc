"""Genera un dataset (observació, etiqueta tova de cartes, valor) etiquetat
per `AgentProbabilistic` per fer warm-start supervisat (SL) del cap de
cartes de la política MaskablePPO.

IDEA (vegeu el pla): en el subjoc de cartes sense apostes, `AgentProbabilistic`
(determinització/PIMC + minimax) és un expert quasi-òptim -- el valor que
retorna `avaluar_opcions` és literalment P(guanyar la mà) amb aquelles
cartes. Distil·lem aquest coneixement:

- ETIQUETA de política (tova): softmax sobre els valors de cada carta jugable
  (destil·lació, preserva empats), dins dels 3 slots `play_card_0/1/2`.
- ETIQUETA de valor: el millor valor de l'estat = max sobre les cartes =
  valor sota continuació òptima (per fer warm-start també del crític).

COBERTURA: etiquetem SEMPRE la decisió del jugador que mou amb el solver,
però AVANCEM el joc amb una política de comportament DIVERSA (solver / regles
en 4 estils / aleatori) perquè el dataset cobreixi també estats que un expert
sol no visitaria (evita mismatch de distribució a l'hora d'entrenar RL).

Només etiquetem nodes de card-play PUR (totes les accions legals són
`play_card_*`); si mai apareix un node de senyes/passar (no hauria, amb
`senyes=False`), s'avança sense etiquetar.

Sortida: un `.npz` amb `obs` (N,125) f32, `slot_values` (N,3) f32 (valor
gairebé òptim de cada slot `play_card_i`; NaN als slots il·legals) i
`slot_mask` (N,3) bool. Guardem els valors CRUS (no la softmax ja feta)
perquè la temperatura τ de l'etiqueta tova sigui un paràmetre barat de
l'entrenament SL, sense haver de regenerar (el solver és car). Es deduplica
per observació idèntica.
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import random

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from joc.entorn_ma.env_ma import TrucEnvMa
from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST

from RL.models.model_propi.agent_probabilistic import AgentProbabilistic
from RL.models.model_propi.agent_regles import AgentRegles
from RL.models.model_propi.random_agent import RandomAgent

PLAY_CARD_NAMES = {'play_card_0', 'play_card_1', 'play_card_2'}
ESTILS_LLISTA = ['conservador', 'equilibrat', 'agressiu', 'farol']
N_ACCIONS = len(ACTION_LIST)

# Config de generació: etapa "cartes" del currículum (sense apostes ni truc),
# sense senyes, 2 jugadors -- així tot node és card-play pur i el solver no
# cau mai al seu fallback d'AgentRegles.
ENV_CONFIG = {
    'num_jugadors': 2,
    'cartes_jugador': 3,
    'senyes': False,
    'permetre_apostes': False,
    'permetre_truc': False,
    'puntuacio_final': 999,
}


def clau_label(raw) -> tuple:
    """Clau completa de l'estat observable de la qual depèn `avaluar_opcions`
    (mà pròpia, cartes jugades per cadascú, rondes guanyades, mà/dealer,
    taula i perspectiva). Serveix per cachejar l'etiqueta i no recalcular el
    solver en estats repetits."""
    root = raw['id_jugador']
    opp = 1 - root
    own_played = tuple(sorted(c for p, c in raw['hist_cartes'] if p == root))
    opp_played = tuple(sorted(c for p, c in raw['hist_cartes'] if p == opp))
    own_hand = tuple(sorted(raw['ma_jugador']))
    taula = tuple((p, c) for p, c in raw['cartes_taula_actual'])
    return (root, own_hand, own_played, opp_played,
            tuple(raw['ronda_winners']), raw['ma'], taula)


def obs_flat(state) -> np.ndarray:
    """(125,) f32 = obs_cartes.flatten()(108) ⧺ obs_context(17), en el mateix
    ordre que espera `CosMultiInputSB3.forward`."""
    obs = state['obs']
    return np.concatenate([
        np.asarray(obs['obs_cartes'], dtype=np.float32).ravel(),
        np.asarray(obs['obs_context'], dtype=np.float32),
    ])


def construir_targets(state, valors):
    """Construeix (slot_values(3,), slot_mask(3,)) a partir del dict
    {carta: valor} del solver, mapejant carta→slot per la posició a la mà
    (slot i ↔ play_card_i ↔ own_hand[i]). Els slots il·legals queden a NaN.
    Es guarden els valors crus; la softmax (amb τ) es fa a l'entrenament."""
    raw = state['raw_obs']
    own_hand = list(raw['ma_jugador'])
    legal = set(state['legal_actions'].keys())

    slot_mask = np.zeros(3, dtype=bool)
    slot_values = np.full(3, np.nan, dtype=np.float32)
    for i in range(min(3, len(own_hand))):
        aid = ACTION_SPACE[f'play_card_{i}']
        if aid in legal:
            slot_mask[i] = True
            slot_values[i] = valors[own_hand[i]]
    return slot_values, slot_mask


def tria_comportament(state, valors, rng, regles_agents, rand_agent,
                      p_solver: float, p_regles: float):
    """Tria l'acció amb què AVANÇAR el joc (no afecta l'etiqueta, només la
    cobertura d'estats): p_solver% argmax del solver, p_regles% un estil de
    regles, resta aleatori."""
    r = rng.random()
    if r < p_solver:
        own_hand = list(state['raw_obs']['ma_jugador'])
        millor = max(valors.values())
        cands = [c for c, v in valors.items() if abs(v - millor) <= 1e-9]
        card = rng.choice(cands)
        return ACTION_SPACE[f'play_card_{own_hand.index(card)}']
    if r < p_solver + p_regles:
        agent = regles_agents[rng.choice(ESTILS_LLISTA)]
        action, _ = agent.eval_step(state)
        return action
    action, _ = rand_agent.eval_step(state)
    return action


def generar(n_mans: int, seed: int,
            p_solver: float, p_regles: float, verbose_cada: int):
    env = TrucEnvMa(config={**ENV_CONFIG, 'seed': seed})
    solver = AgentProbabilistic(seed=seed)
    regles_agents = {e: AgentRegles(num_actions=N_ACCIONS, seed=seed, estil=e)
                     for e in ESTILS_LLISTA}
    rand_agent = RandomAgent(num_actions=N_ACCIONS, seed=seed)
    rng = random.Random(seed)

    dades: dict[bytes, tuple] = {}   # dedup per observació
    label_cache: dict[tuple, dict] = {}
    temps_solver = []
    n_decisions = 0
    n_no_pur = 0

    t0 = time.time()
    for ma_idx in range(n_mans):
        state, pid = env.reset()
        while pid is not None:
            legal = list(state['legal_actions'].keys())
            es_pur = all(ACTION_LIST[a] in PLAY_CARD_NAMES for a in legal)

            if not es_pur:
                # Node de senyes/passar (no hauria de passar amb senyes=False):
                # avança sense etiquetar.
                n_no_pur += 1
                passar = ACTION_SPACE['passar']
                action = passar if passar in legal else legal[0]
                state, pid = env.step(action)
                continue

            raw = state['raw_obs']
            k = clau_label(raw)
            valors = label_cache.get(k)
            if valors is None:
                t = time.time()
                valors = solver.avaluar_opcions(state)
                temps_solver.append(time.time() - t)
                label_cache[k] = valors

            slot_values, slot_mask = construir_targets(state, valors)
            key_obs = obs_flat(state)
            dades.setdefault(key_obs.tobytes(),
                             (key_obs, slot_values, slot_mask))
            n_decisions += 1

            own_hand = list(raw['ma_jugador'])
            action = tria_comportament(state, valors, rng, regles_agents,
                                       rand_agent, p_solver, p_regles)

            # Asserció crítica: la decodificació play_card_i ha de retirar
            # exactament own_hand[i]. Si això falla, TOTES les etiquetes estan
            # mal alineades -> el dataset seria inservible.
            nom = ACTION_LIST[action]
            carta_esperada = None
            if nom.startswith('play_card_'):
                carta_esperada = own_hand[int(nom.rsplit('_', 1)[1])]

            actor_pid = pid
            state, pid = env.step(action)

            if carta_esperada is not None:
                entry = env.game.hist_cartes[-1]
                assert entry[0] == actor_pid and entry[-1] == carta_esperada, (
                    f"Desalineació slot↔carta: acció {nom} esperava "
                    f"{carta_esperada!r} del jugador {actor_pid}, però l'historial "
                    f"registra {entry!r}"
                )

        if verbose_cada and (ma_idx + 1) % verbose_cada == 0:
            print(f"  mà {ma_idx+1}/{n_mans} | estats únics={len(dades)} "
                  f"| decisions={n_decisions} | {time.time()-t0:.0f}s")

    return dades, {
        'n_decisions': n_decisions,
        'n_unics': len(dades),
        'n_no_pur': n_no_pur,
        'temps_solver_ms_mitja': (np.mean(temps_solver) * 1000) if temps_solver else 0.0,
        'temps_solver_ms_max': (np.max(temps_solver) * 1000) if temps_solver else 0.0,
        'temps_total_s': time.time() - t0,
        'n_solver_calls': len(temps_solver),
    }


def main():
    ap = argparse.ArgumentParser(description="Genera dataset SL de cartes etiquetat per AgentProbabilistic")
    ap.add_argument('--n_mans', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--p_solver', type=float, default=0.40, help="fracció d'accions d'avanç triades pel solver")
    ap.add_argument('--p_regles', type=float, default=0.40, help="fracció triada per AgentRegles (resta: aleatori)")
    ap.add_argument('--verbose_cada', type=int, default=250)
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'dades_sl', 'dataset_cartes.npz'))
    args = ap.parse_args()

    print(f"Generant dataset SL de cartes: n_mans={args.n_mans}, "
          f"comportament(solver={args.p_solver}, regles={args.p_regles}, "
          f"random={1-args.p_solver-args.p_regles:.2f})")

    dades, stats = generar(args.n_mans, args.seed,
                           args.p_solver, args.p_regles, args.verbose_cada)

    obs = np.stack([v[0] for v in dades.values()]).astype(np.float32)
    slot_values = np.stack([v[1] for v in dades.values()]).astype(np.float32)
    slot_mask = np.stack([v[2] for v in dades.values()]).astype(bool)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=obs, slot_values=slot_values, slot_mask=slot_mask)

    value_millor = np.nanmax(slot_values, axis=1)
    print("=" * 70)
    print(f"Estats únics desats: {stats['n_unics']}  (de {stats['n_decisions']} decisions)")
    print(f"Crides reals al solver: {stats['n_solver_calls']}  "
          f"(cache estalvia {stats['n_decisions'] - stats['n_solver_calls']})")
    print(f"Temps solver: mitjà={stats['temps_solver_ms_mitja']:.1f}ms, "
          f"màxim={stats['temps_solver_ms_max']:.1f}ms | total={stats['temps_total_s']:.0f}s")
    print(f"Nodes no-purs (senyes/passar) trobats: {stats['n_no_pur']}")
    print(f"obs={obs.shape}  slot_values={slot_values.shape}  slot_mask={slot_mask.shape}")
    print(f"valor millor slot: min={value_millor.min():.3f} max={value_millor.max():.3f} "
          f"mitjà={value_millor.mean():.3f}")
    print(f"Desat a: {args.out}")


if __name__ == '__main__':
    main()
