"""
Fase 2: Entrenament PPO amb Cos (CosMultiInput)
================================================
Compara 3 modes d'entrenament:
  - scratch:  cos aleatori, tot entrena des de zero
  - frozen:   cos SL pre-entrenat, congelat durant tot l'RL
  - finetune: cos SL pre-entrenat, congelat inicialment, descongelat al 15%

Condicions idèntiques a Fase 1 PPO paral·lel:
  - 48 entorns paral·lels (SubprocVecEnv)
  - Oponents: 5% Random, 65% AgentRegles, 30% Self-play
  - Mètrica: 0.25 × WR_random + 0.75 × WR_regles
  - 24M timesteps per defecte
"""

import sys
import os
import argparse
import csv
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm

try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn.parallel_env import SubprocVecEnv
from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import PPOMlpNet, COS_WEIGHTS_PATH, SPLIT, OBS_CONTEXT_SIZE
from RL.models.model_propi.model_ppo.ppo.agent_ppo_mlp import PPOMlpAgent
from RL.entrenament.entrenamentsPropis.ppo.buffers_ppo import RolloutBuffer
from RL.models.model_propi.model_ppo.ppo_loss import calcular_gae, calcular_perdua_ppo
from RL.models.model_propi.model_ppo.ppo_utils import extract_obs, evaluar_contra_random, evaluar_contra_regles
from RL.models.model_propi.agent_regles import AgentRegles
from joc.entorn.cartes_accions import ACTION_LIST
from rlcard.agents import RandomAgent

# Constants (idèntiques a Fase 1 PPO)
NUM_ENVS         = 48
NUM_STEPS        = 256
MINIBATCH_SIZE   = 1024
UPDATE_EPOCHS    = 7
TOTAL_TIMESTEPS  = 24_000_000
LR               = 3e-4
GAMMA            = 0.995
GAE_LAMBDA       = 0.95
CLIP_COEF        = 0.2
ENT_COEF         = 0.03
VF_COEF          = 0.5

# Fine-tune
FINETUNE_LR_COS  = 1e-5
FINETUNE_LR_MLP  = 1e-4
UNFREEZE_FRACTION = 0.15

# Oponents
PCT_RANDOM = 0.05
PCT_REGLES = 0.65

# Avaluació
EVAL_EVERY_STEPS  = 500_000
EVAL_GAMES_RANDOM = 50
EVAL_GAMES_REGLES = 100

N_ACTIONS = len(ACTION_LIST)
OBS_DIM   = SPLIT + OBS_CONTEXT_SIZE  # 239

ENV_CONFIG = {
    'num_jugadors': 2,
    'cartes_jugador': 3,
    'puntuacio_final': 24,
    'seed': 42,
}


def build_opponent_map(num_envs):
    opp_map = {}
    n_random = max(1, int(num_envs * PCT_RANDOM))
    n_regles = int(num_envs * PCT_REGLES)
    for i in range(num_envs):
        if i < n_random:
            opp_map[i] = {'type': 'random', 'pid': i % 2}
        elif i < n_random + n_regles:
            opp_map[i] = {'type': 'regles', 'pid': i % 2}
    return opp_map


def init_log(log_path):
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'step', 'games_played', 'loss',
            'eval_wr_random', 'eval_wr_regles', 'eval_metric', 'elapsed_s'
        ])


def append_log(log_path, step, games, loss, wr_r, wr_g, metric, elapsed):
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            step, games, f'{loss:.5f}' if loss is not None else '',
            f'{wr_r:.2f}', f'{wr_g:.2f}', f'{metric:.2f}', f'{elapsed:.1f}'
        ])


def main():
    parser = argparse.ArgumentParser(description='Fase 2: PPO amb Cos (scratch/frozen/finetune)')
    parser.add_argument('--mode', choices=['scratch', 'frozen', 'finetune'], required=True)
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--cos_weights', type=str, default=None,
                        help=f"Ruta als pesos SL del cos. 'none' per COS aleatori. Default: {os.path.basename(COS_WEIGHTS_PATH)}")
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--unfreeze_fraction', type=float, default=UNFREEZE_FRACTION)
    parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
    args = parser.parse_args()

    total_timesteps = args.total_timesteps
    num_envs = args.num_envs
    mode = args.mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{device.type.upper()}] Fase 2 — PPO+Cos mode={mode.upper()} | {total_timesteps/1e6:.0f}M steps | {num_envs} envs')

    # --- Model ---
    # Determinar pesos del cos
    if mode == 'scratch' or args.cos_weights == 'none':
        cos_w = 'none'  # No existeix → PPOMlpNet no carrega pesos
    elif args.cos_weights:
        cos_w = args.cos_weights
    else:
        cos_w = None  # Usa COS_WEIGHTS_PATH per defecte

    net = PPOMlpNet(n_actions=N_ACTIONS, hidden_size=256, ruta_weights=cos_w, device=device)

    # Post-init segons mode
    if mode == 'scratch':
        net.unfreeze_cos()  # Tot entrena des de zero
        optimizer = optim.Adam(net.parameters(), lr=LR, eps=1e-5)
        print(f'[Scratch] Cos aleatori, tot entrena. Params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}')
    elif mode == 'frozen':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
        print(f'[Frozen] Cos SL congelat. Trainable: {n_train:,} | Frozen: {n_frozen:,}')
    elif mode == 'finetune':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)
        unfreeze_step = int(total_timesteps * args.unfreeze_fraction)
        has_unfrozen = False
        print(f'[Finetune] Cos SL congelat fins step {unfreeze_step:,} ({args.unfreeze_fraction*100:.0f}%)')

    agent = PPOMlpAgent(net, N_ACTIONS, device=device)

    # --- Entorns paral·lels ---
    vec_env = SubprocVecEnv(num_envs, ENV_CONFIG)

    # --- Oponents ---
    rand_opp = RandomAgent(num_actions=N_ACTIONS)
    regles_opp = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    opp_map = build_opponent_map(num_envs)

    # --- Buffer ---
    buffer = RolloutBuffer(NUM_STEPS, num_envs, OBS_DIM, action_dim=N_ACTIONS, device=device)

    # --- Directori de sortida ---
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%dd_%mm_%H%Mh")
        save_dir = Path(__file__).parent / 'registres' / f'fase2_{mode}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    # --- Iniciar entorns ---
    results = vec_env.reset_all()
    current_states = [r[0] for r in results]

    num_updates = total_timesteps // (num_envs * NUM_STEPS)
    eval_every = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    global_step = 0
    games_played = 0
    best_metric = -1.0
    t0 = time.time()
    pg_loss_val = 0.0

    pbar = trange(1, num_updates + 1, desc=f'PPO-{mode}')
    for update in pbar:
        buffer.step = 0
        for step in range(NUM_STEPS):
            global_step += num_envs

            obs_t, masks_t = extract_obs(current_states)
            obs_t = obs_t.to(device)
            masks_t = masks_t.to(device)

            active_players = [s['raw_obs']['id_jugador'] for s in current_states]
            is_learning = torch.ones(num_envs, device=device)

            net.eval()
            with torch.no_grad():
                logits, value = net(obs_t)
                logits = logits.masked_fill(~masks_t, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            # Sobreescriure accions oponents
            for i, opp in opp_map.items():
                if active_players[i] == opp['pid']:
                    if opp['type'] == 'random':
                        a, _ = rand_opp.eval_step(current_states[i])
                        action[i] = a
                        is_learning[i] = 0.0
                    elif opp['type'] == 'regles':
                        a, _ = regles_opp.eval_step(current_states[i])
                        action[i] = a
                        is_learning[i] = 0.0

            actions_np = action.cpu().numpy()
            next_states_players, rewards_list, dones_list = vec_env.step(actions_np)

            step_rewards = []
            for i in range(num_envs):
                if dones_list[i]:
                    games_played += 1
                step_rewards.append(rewards_list[i][active_players[i]])

            rewards_t = torch.tensor(step_rewards, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones_list, dtype=torch.float32, device=device)

            buffer.add(obs_t, action, logprob, rewards_t, value.squeeze(-1),
                       dones_t, masks_t, is_learning)

            current_states = [sp[0] for sp in next_states_players]

        # GAE
        obs_last_t, _ = extract_obs(current_states)
        obs_last_t = obs_last_t.to(device)
        net.eval()
        with torch.no_grad():
            _, last_value = net(obs_last_t)
            last_value = last_value.squeeze(-1)

        advantages, returns = calcular_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_value, torch.zeros_like(last_value),
            GAMMA, GAE_LAMBDA,
        )

        b_obs, b_act, b_lp, b_adv, b_ret, b_masks, b_il = buffer.get(advantages, returns)
        b_inds = np.arange(num_envs * NUM_STEPS)

        # Finetune: descongelar cos
        if mode == 'finetune' and not has_unfrozen and global_step >= unfreeze_step:
            net.unfreeze_cos()
            param_groups = net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=FINETUNE_LR_MLP)
            optimizer = optim.Adam(param_groups, eps=1e-5)
            finetune_base_lrs = [pg['lr'] for pg in optimizer.param_groups]
            has_unfrozen = True
            print(f'\n[Finetune] Cos descongelat al step {global_step:,}. Optimizer reconstruït.')

        # LR scheduling: decay lineal cap a 0
        frac = 1.0 - (update - 1) / num_updates
        if mode == 'finetune' and has_unfrozen:
            for pg, base_lr in zip(optimizer.param_groups, finetune_base_lrs):
                pg['lr'] = base_lr * frac
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = LR * frac

        # PPO update
        net.train()
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, num_envs * NUM_STEPS, MINIBATCH_SIZE):
                mb = b_inds[start:start + MINIBATCH_SIZE]
                loss, pg_loss, v_loss, ent_loss = calcular_perdua_ppo(
                    agent, b_obs[mb], b_act[mb], b_lp[mb],
                    b_adv[mb], b_ret[mb], b_masks[mb],
                    is_learning=b_il[mb],
                    coef_retall=CLIP_COEF, coef_ent=ENT_COEF, coef_v=VF_COEF,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        pg_loss_val = pg_loss.item()

        # Avaluació periòdica
        if global_step % eval_every < (num_envs * NUM_STEPS):
            _, wr_r = evaluar_contra_random(agent, ENV_CONFIG, device)
            _, wr_g = evaluar_contra_regles(agent, regles_eval, ENV_CONFIG)
            metric = 0.25 * wr_r + 0.75 * wr_g
            elapsed = time.time() - t0

            append_log(log_path, global_step, games_played, pg_loss_val,
                       wr_r, wr_g, metric, elapsed)

            if metric > best_metric:
                best_metric = metric
                torch.save(net.state_dict(), save_dir / 'best.pt')
                tqdm.write(f'  [{mode}] step {global_step:,} | random={wr_r:.1f}% regles={wr_g:.1f}% metric={metric:.2f} >> NOU MILLOR!')
            else:
                tqdm.write(f'  [{mode}] step {global_step:,} | random={wr_r:.1f}% regles={wr_g:.1f}% metric={metric:.2f}')

        pbar.set_postfix({
            'step': f'{global_step/1e6:.1f}M',
            'games': games_played,
            'pg': f'{pg_loss_val:.3f}',
        })

        if update % 100 == 0:
            torch.cuda.empty_cache()

    # Finalització
    vec_env.close()
    torch.save(net.state_dict(), save_dir / 'final.pt')

    elapsed_total = time.time() - t0
    h = int(elapsed_total // 3600)
    m = int((elapsed_total % 3600) // 60)
    s = int(elapsed_total % 60)
    print(f'\n[{mode.upper()}] Completat en {h}h {m}m {s}s | Games: {games_played:,} | Best metric: {best_metric:.2f}')
    print(f'Resultats a: {save_dir}')


if __name__ == '__main__':
    main()
