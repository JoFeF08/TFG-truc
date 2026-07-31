"""Driver del CFR: resol els dos subjocs d'aposta (envit i truc) i reporta
l'exploitability (resultat "Nash" de la tesi). Desa les estratègies mitjanes.

Ús:
    python -m RL.cfr.train_cfr --iters_envit 1500 --iters_truc 800

Requereix la taula d'equity/buckets precomputada:
    python -m RL.cfr.truc_equity 300
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

from RL.cfr.envit_game import EnvitGame
from RL.cfr.truc_game import TrucGame
from RL.cfr.truc_equity import carregar_buckets, construir_chance_truc
from RL.cfr.cfr_solver import entrenar_cfr, exploitability

POLICIES_DIR = os.path.join(os.path.dirname(__file__), 'policies')


def _desar(nom, avg, extra):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    with open(os.path.join(POLICIES_DIR, nom), 'wb') as f:
        pickle.dump({'avg': avg, **extra}, f)


def resoldre_envit(iters, n_mostres_chance, R, seed):
    print("=" * 70)
    print(f"ENVIT (R={R})")
    g = EnvitGame(R=R)
    t = time.time()
    g.calcular_chance(n_mostres=n_mostres_chance, seed=seed)
    print(f"  chance: {len(g.chance_outcomes())} outcomes en {time.time()-t:.1f}s")
    avg, _, ss = entrenar_cfr(g, iteracions=iters, log_cada=max(1, iters // 8))
    expl = exploitability(g, avg)
    print(f"  -> {len(ss)} infosets | exploitability = {expl:.5f} pts/mà")
    _desar('envit.pkl', avg, {'R': R, 'exploitability': expl})
    return expl


def resoldre_truc(iters, n_deals, R, seed):
    print("=" * 70)
    print(f"TRUC (R={R})")
    dades = carregar_buckets()
    bucket_table, n_buckets = dades['buckets'], dades['n_buckets']
    t = time.time()
    chance = construir_chance_truc(bucket_table, n_deals=n_deals, seed=seed)
    print(f"  chance: {len(chance)} outcomes (b0,b1,rw_full) de {n_deals} deals en {time.time()-t:.1f}s")
    g = TrucGame(chance, R=R)
    avg, _, ss = entrenar_cfr(g, iteracions=iters, log_cada=max(1, iters // 8))
    expl = exploitability(g, avg)
    print(f"  -> {len(ss)} infosets | exploitability = {expl:.5f} pts/mà")
    _desar('truc.pkl', avg, {'R': R, 'n_buckets': n_buckets, 'exploitability': expl})
    return expl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters_envit', type=int, default=1500)
    ap.add_argument('--iters_truc', type=int, default=800)
    ap.add_argument('--chance_envit', type=int, default=300_000)
    ap.add_argument('--deals_truc', type=int, default=500_000)
    ap.add_argument('--R', type=int, default=24, help="valor de falta/joc-fora = 24 - max(marcador)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nomes', choices=['envit', 'truc'], default=None)
    args = ap.parse_args()

    if args.nomes != 'truc':
        resoldre_envit(args.iters_envit, args.chance_envit, args.R, args.seed)
    if args.nomes != 'envit':
        resoldre_truc(args.iters_truc, args.deals_truc, args.R, args.seed)
    print("=" * 70)
    print(f"Polítiques desades a {POLICIES_DIR}")


if __name__ == '__main__':
    main()
