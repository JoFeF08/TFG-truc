"""
Mini Lliga — Round-Robin entre els 6 millors models PPO (run 1246h)
=======================================================================
Execució:
    python mini_lliga.py [--n N] [--run experiments_comparativa_26_03_1246h]

Cada parella juga N partides completes (per defecte N=100).
Una "partida" = una mà completa del Truc (episodi de TrucEnvMa).
"""

import sys, os, argparse, itertools
# UTF-8 per evitar errors d'encoding a Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import torch

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # TFG-truc/
sys.path.insert(0, str(ROOT))

from joc.entorn.env import TrucEnv
from RL.models.model_propi.ppo.cap_ppo_mlp    import PPOMlpNet
from RL.models.model_propi.ppo.agent_ppo_mlp  import PPOMlpAgent
from RL.models.model_propi.ppo_gru.cap_ppo_gru   import PPOGruNet
from RL.models.model_propi.ppo_gru.agent_ppo_gru import PPOGruAgent

# ── Configuració ──────────────────────────────────────────────────────────────

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--n',   type=int,   default=500,
                    help='Partides per parella (default 500)')
PARSER.add_argument('--run', type=str,
                    default='experiments_comparativa_26_03_1246h',
                    help='Directori de la run a avaluar')
ARGS = PARSER.parse_args()

RUN_DIR = ROOT / ARGS.run
N_PARTIDES = ARGS.n

MODELS_DEF = [
    ('GRU-scratch', 'gru', RUN_DIR / 'gru_baseline_scratch' / 'best.pt'),
    ('GRU-fase2',   'gru', RUN_DIR / 'gru_fase2_frozen'     / 'best.pt'),
    ('MLP-scratch', 'mlp', RUN_DIR / 'mlp_baseline_scratch' / 'best.pt'),
    ('MLP-fase2',   'mlp', RUN_DIR / 'mlp_fase2_frozen'     / 'best.pt'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_env():
    """Crea un entorn TrucEnv (partida completa fins a 24 punts)."""
    config = {
        'num_jugadors':    2,
        'cartes_jugador':  3,
        'puntuacio_final': 24,
        'senyes':          False,
        'seed':            None,
    }
    return TrucEnv(config)


def carrega_agent(nom, tipus, ruta, n_actions):
    """Carrega un agent des de best.pt."""
    device = 'cpu'
    ruta = str(ruta)
    if tipus == 'mlp':
        net = PPOMlpNet(n_actions=n_actions, device=device)
        net.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
        net.eval()
        return PPOMlpAgent(net, num_actions=n_actions, device=device)
    else:  # gru
        net = PPOGruNet(n_actions=n_actions, device=device)
        net.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
        net.eval()
        return PPOGruAgent(net, num_actions=n_actions, num_envs=1, device=device)


def reset_hidden_if_gru(agent):
    """Reinicia hidden state GRU entre partides."""
    if hasattr(agent, 'eval_hidden'):
        agent.eval_hidden = torch.zeros_like(agent.eval_hidden)


def jugar_partida(agent0, agent1, env):
    """
    Juga una partida completa via env.run() (patró rlcard estàndard).
    Retorna l'índex del jugador guanyador (0 o 1), o -1 en empat.
    """
    reset_hidden_if_gru(agent0)
    reset_hidden_if_gru(agent1)
    env.set_agents([agent0, agent1])
    _, payoffs = env.run(is_training=False)
    if payoffs[0] > payoffs[1]:
        return 0
    elif payoffs[1] > payoffs[0]:
        return 1
    else:
        return -1


# ── Càrrega de models ─────────────────────────────────────────────────────────

print("=" * 60)
print(f"Mini Lliga — {ARGS.run}")
print(f"Partides per parella: {N_PARTIDES}")
print("=" * 60)

env = build_env()
N_ACTIONS = env.num_actions
print(f"N_ACTIONS = {N_ACTIONS}\n")

agents = {}
for nom, tipus, ruta in MODELS_DEF:
    if not ruta.exists():
        print(f"[AVÍS] No trobat: {ruta} — s'omet {nom}")
        continue
    print(f"Carregant {nom} ({tipus}) ...", end=' ', flush=True)
    agents[nom] = (carrega_agent(nom, tipus, ruta, N_ACTIONS), tipus)
    print("OK")

noms = list(agents.keys())
n = len(noms)
print(f"\n{n} models carregats: {noms}\n")

# ── Round-Robin ───────────────────────────────────────────────────────────────

victòries  = {nom: 0 for nom in noms}
empats     = {nom: 0 for nom in noms}
partides   = {nom: 0 for nom in noms}

resultats  = {}   # (nomA, nomB) → [wins_A, wins_B, empats]

N_LOCAL    = N_PARTIDES // 2          # partides com a local (agent0)
N_VISITANT = N_PARTIDES - N_LOCAL     # partides com a visitant (agent1)
N_TOTAL    = N_LOCAL + N_VISITANT     # = N_PARTIDES

for nomA, nomB in itertools.combinations(noms, 2):
    agentA, _ = agents[nomA]
    agentB, _ = agents[nomB]
    wA = wB = emp = 0

    # Meitat de partides: A és local (agent0), B és visitant (agent1)
    for _ in range(N_LOCAL):
        guanyador = jugar_partida(agentA, agentB, env)
        if guanyador == 0:
            wA += 1
        elif guanyador == 1:
            wB += 1
        else:
            emp += 1

    # Altra meitat: B és local (agent0), A és visitant (agent1)
    for _ in range(N_VISITANT):
        guanyador = jugar_partida(agentB, agentA, env)
        if guanyador == 1:    # l'agent1 és A → A guanya
            wA += 1
        elif guanyador == 0:  # l'agent0 és B → B guanya
            wB += 1
        else:
            emp += 1

    resultats[(nomA, nomB)] = [wA, wB, emp]
    victòries[nomA] += wA
    victòries[nomB] += wB
    empats[nomA]    += emp
    empats[nomB]    += emp
    partides[nomA]  += N_TOTAL
    partides[nomB]  += N_TOTAL

    pct_A = 100 * wA / N_TOTAL
    pct_B = 100 * wB / N_TOTAL
    print(f"  {nomA:15s} vs {nomB:15s} -> {wA:3d}/{wB:3d} ({pct_A:.0f}%/{pct_B:.0f}%) [{N_LOCAL}L+{N_VISITANT}V]")

# ── Taula de resultats ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CLASSIFICACIÓ FINAL")
print("=" * 60)

files = []
for nom in noms:
    p = partides[nom]
    w = victòries[nom]
    e = empats[nom]
    pct = 100 * w / p if p > 0 else 0
    files.append([nom, w, e, p - w - e, p, f"{pct:.1f}%"])

files.sort(key=lambda x: -float(x[5].rstrip('%')))
for i, f in enumerate(files):
    f.insert(0, i + 1)

capçaleres = ['#', 'Model', 'Victòries', 'Empats', 'Derrotes', 'Total', 'Win%']
if HAS_TABULATE:
    print(tabulate(files, headers=capçaleres, tablefmt='github'))
else:
    print(f"{'#':<3} {'Model':<16} {'V':>5} {'E':>5} {'D':>5} {'Tot':>5} {'Win%':>6}")
    print("-" * 50)
    for f in files:
        print(f"{f[0]:<3} {f[1]:<16} {f[2]:>5} {f[3]:>5} {f[4]:>5} {f[5]:>5} {f[6]:>6}")

# ── Matriu de resultats ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MATRIU WIN% (fila=atacant, columna=rival)")
print("=" * 60)

header = [''] + noms
matriu = []
for nomA in noms:
    fila = [nomA]
    for nomB in noms:
        if nomA == nomB:
            fila.append('-')
        elif (nomA, nomB) in resultats:
            wA, wB, emp = resultats[(nomA, nomB)]
            fila.append(f"{100*wA/N_TOTAL:.0f}%")
        elif (nomB, nomA) in resultats:
            wA, wB, emp = resultats[(nomB, nomA)]
            fila.append(f"{100*wB/N_TOTAL:.0f}%")
        else:
            fila.append('N/A')
    matriu.append(fila)

if HAS_TABULATE:
    print(tabulate(matriu, headers=header, tablefmt='github'))
else:
    print('\t'.join(header))
    for fila in matriu:
        print('\t'.join(str(x) for x in fila))

print("\nFi de la Mini Lliga.")

# ── Export JSON ───────────────────────────────────────────────────────────────
import json

ranking_json = [
    {"model": f[1], "win_pct": float(f[6].rstrip('%'))}
    for f in files
]

matriu_json = {}
for nomA in noms:
    matriu_json[nomA] = {}
    for nomB in noms:
        if nomA == nomB:
            matriu_json[nomA][nomB] = None
        elif (nomA, nomB) in resultats:
            wA, wB, _ = resultats[(nomA, nomB)]
            matriu_json[nomA][nomB] = round(100 * wA / N_TOTAL)
        elif (nomB, nomA) in resultats:
            _, wB_inv, _ = resultats[(nomB, nomA)]
            matriu_json[nomA][nomB] = round(100 * wB_inv / N_TOTAL)
        else:
            matriu_json[nomA][nomB] = None

export = {
    "noms": noms,
    "n_partides": N_TOTAL,
    "run": ARGS.run,
    "ranking": ranking_json,
    "matriu": matriu_json,
}

json_path = Path(__file__).parent / "resultats_mini_lliga.json"
with open(json_path, 'w', encoding='utf-8') as fj:
    json.dump(export, fj, ensure_ascii=False, indent=2)
print(f"Resultats desats a: {json_path}")
