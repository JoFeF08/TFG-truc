import sys
import os
import argparse
import csv
import time
import logging
import types
import contextlib
import io
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, root_path)
except Exception:
    pass

from rlcard.agents import DQNAgent, NFSPAgent, RandomAgent
from rlcard.utils import set_seed

from stable_baselines3.common.vec_env import SubprocVecEnv as SB3SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from joc.entorn import TrucEnv
from joc.entorn.cartes_accions import ACTION_LIST
from joc.entorn.gym_env import TrucGymEnv

from joc.entorn.parallel_env import SubprocVecEnv
from RL.models.model_propi.agent_regles import AgentRegles

# Silenciar logs
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)

# Constants generals
TOTAL_TIMESTEPS  = 24_000_000
NUM_STEPS        = 256
OBS_DIM          = 216 + 23
N_ACTIONS        = len(ACTION_LIST)
HIDDEN_SIZE      = 256
HIDDEN_LAYERS    = [256, 256]
SEED             = 42

# Parallelisme per algorisme
# PPO (on-policy): prefereix gran throughput
NUM_ENVS_PPO     = 48
# DQN/NFSP (off-policy): necessiten profunditat temporal
NUM_ENVS_DQN     = 16
NUM_ENVS_NFSP    = 16

EVAL_EVERY_STEPS  = 500_000
EVAL_GAMES_RANDOM = 50
EVAL_GAMES_REGLES = 100

ENV_CONFIG = {
    'num_jugadors': 2,
    'cartes_jugador': 3,
    'puntuacio_final': 24,
    'seed': SEED,
    'verbose': False,
}

# Distribució d'oponents
PCT_RANDOM = 0.05
PCT_REGLES = 0.65

# Hiperparàmetres PPO
PPO_LR          = 3e-4
PPO_GAMMA       = 0.995
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP        = 0.2
PPO_ENT         = 0.01
PPO_VF          = 0.5
PPO_EPOCHS      = 7
PPO_MINIBATCH   = 1024

# Hiperparàmetres DQN
DQN_LR           = 1e-4
DQN_BATCH        = 512
DQN_MEMORY       = 2_000_000
DQN_WARMUP       = 100_000
DQN_TRAIN_EVERY  = 128
DQN_UPDATE_TGT   = 2_000
DQN_EPS_MIN      = 0.05
DQN_POLYAK_TAU   = 0.05
DQN_POLYAK_FREQ  = 5_000

# Hiperparàmetres NFSP
NFSP_RL_LR         = 1e-4
NFSP_SL_LR         = 1e-4
NFSP_BATCH         = 512
NFSP_RESERVOIR     = 2_000_000
NFSP_Q_REPLAY      = 2_000_000
NFSP_Q_TRAIN_EVERY = 128
NFSP_Q_UPDATE      = 2_000
NFSP_WARMUP        = 100_000
NFSP_ETA           = 0.25
NFSP_EPS_MIN       = 0.05
NFSP_POLICY_SAMPLE_FREQ = 1_000



class SB3PPOEvalAgent:
    """
    Wrapper per fer un model MaskablePPO de SB3 compatible amb
    l'interface eval_step() de RLCard (usada per evaluar_agent).
    """
    use_raw = False

    def __init__(self, model, n_actions: int = N_ACTIONS):
        self.model = model
        self.num_actions = n_actions

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        else:
            obs_flat = np.asarray(obs, dtype=np.float32)

        legal = list(state['legal_actions'].keys())
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[legal] = True

        # predict() necessita dimensió de batch: (1, obs_dim) i (1, n_actions)
        action, _ = self.model.predict(
            obs_flat[np.newaxis],
            action_masks=mask[np.newaxis],
            deterministic=True,
        )
        return int(action[0]), {}



# Funcions compartides
def flatten_obs(state):
    """Aplana l'observació dict→array (239 dims)."""
    obs = state['obs']
    if isinstance(obs, dict):
        return np.concatenate(
            [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


def make_flat_state(state):
    """Retorna un dict amb 'obs' pla + 'legal_actions' + 'raw_obs' originals."""
    return {
        'obs': flatten_obs(state),
        'legal_actions': state['legal_actions'],
        'raw_obs': state.get('raw_obs', {}),
    }


def wrap_env_aplanat(env):
    """Posa un monkey-patch a TrucEnv perquè retorni obs planes (per agents RLCard)."""
    original = env._extract_state

    def _extract_patched(self, state):
        extracted = original(state)
        if isinstance(extracted.get('obs'), dict):
            extracted['obs'] = np.concatenate([
                extracted['obs']['obs_cartes'].flatten(),
                extracted['obs']['obs_context'],
            ], axis=0).astype(np.float32)
        return extracted

    env._extract_state = types.MethodType(_extract_patched, env)
    return env


def build_opponent_map(num_envs):
    """Retorna un dict {env_idx: {'type': str, 'pid': int}} per als entorns."""
    opp_map = {}
    n_random = max(1, int(num_envs * PCT_RANDOM))
    n_regles = int(num_envs * PCT_REGLES)
    for i in range(num_envs):
        if i < n_random:
            opp_map[i] = {'type': 'random', 'pid': i % 2}
        elif i < n_random + n_regles:
            opp_map[i] = {'type': 'regles', 'pid': i % 2}
        else:
            opp_map[i] = {'type': 'selfplay', 'pid': i % 2}
    return opp_map


def evaluar_agent(agent, env_config, regles_agent,
                  n_random=EVAL_GAMES_RANDOM, n_regles=EVAL_GAMES_REGLES):
    """
    Avalua l'agent contra Random i AgentRegles.
    Retorna (wr_random%, wr_regles%, metric).
    Funciona per DQNAgent, NFSPAgent i SimpleActorCriticAgent (tots tenen eval_step).
    """
    eval_env = wrap_env_aplanat(TrucEnv(env_config))

    # vs Random
    rand_opp = RandomAgent(num_actions=N_ACTIONS)
    wins_r = 0
    for i in range(n_random):
        if i % 2 == 0:
            eval_env.set_agents([agent, rand_opp])
            pid = 0
        else:
            eval_env.set_agents([rand_opp, agent])
            pid = 1
        _, payoffs = eval_env.run(is_training=False)
        if payoffs[pid] > 0:
            wins_r += 1
    wr_random = 100.0 * wins_r / n_random

    # vs AgentRegles
    wins_g = 0
    for i in range(n_regles):
        if i % 2 == 0:
            eval_env.set_agents([agent, regles_agent])
            pid = 0
        else:
            eval_env.set_agents([regles_agent, agent])
            pid = 1
        _, payoffs = eval_env.run(is_training=False)
        if payoffs[pid] > 0:
            wins_g += 1
    wr_regles = 100.0 * wins_g / n_regles

    metric = 0.25 * wr_random + 0.75 * wr_regles
    return wr_random, wr_regles, metric


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


"""
Entrenament Comparatiu: DQN / NFSP / PPO
=========================================
Tots tres algorismes amb condicions idèntiques:
  - Observació plana de 239 dims (sense COS), des de zero (scratch)
  - Xarxa [256, 256] hidden layers per a tots
  - 48 entorns paral·lels (SubprocVecEnv)
  - Mateixa distribució d'oponents: 5% Random, 65% AgentRegles, 30% Self-play
  - Mateixa avaluació: win rate vs Random + vs AgentRegles
  - 24M timesteps totals
"""

# DQN
def run_dqn(save_dir, total_timesteps, device, num_envs_override=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    num_envs = num_envs_override or NUM_ENVS_DQN
    eval_every = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    eps_decay = int(total_timesteps * 0.8)
    memory_size = min(DQN_MEMORY, 500_000) if num_envs <= 4 else DQN_MEMORY
    warmup = min(DQN_WARMUP, memory_size // 10)

    dqn = DQNAgent(
        num_actions=N_ACTIONS,
        state_shape=OBS_DIM,
        mlp_layers=HIDDEN_LAYERS,
        learning_rate=DQN_LR,
        batch_size=DQN_BATCH,
        replay_memory_size=memory_size,
        replay_memory_init_size=warmup,
        update_target_estimator_every=DQN_UPDATE_TGT,
        epsilon_decay_steps=eps_decay,
        epsilon_end=DQN_EPS_MIN,
        train_every=DQN_TRAIN_EVERY,
        device=device,
    )

    # Oponent self-play: còpia Polyak del DQN principal
    dqn_polyak = DQNAgent(
        num_actions=N_ACTIONS,
        state_shape=OBS_DIM,
        mlp_layers=HIDDEN_LAYERS,
        learning_rate=DQN_LR,
        batch_size=DQN_BATCH,
        replay_memory_size=1000,
        replay_memory_init_size=100,
        update_target_estimator_every=DQN_UPDATE_TGT,
        epsilon_decay_steps=1,
        epsilon_end=0.0,
        train_every=99999,
        device=device,
    )
    # Copiem pesos inicials
    dqn_polyak.q_estimator.qnet.load_state_dict(dqn.q_estimator.qnet.state_dict())

    rand_opp   = RandomAgent(num_actions=N_ACTIONS)
    regles_opp = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)

    opp_map = build_opponent_map(num_envs)
    vec_env = SubprocVecEnv(num_envs, ENV_CONFIG)

    results = vec_env.reset_all()
    current_states = [r[0] for r in results]

    num_updates = total_timesteps // (num_envs * NUM_STEPS)
    global_step = 0
    games_played = 0
    best_metric = -1.0
    t0 = time.time()
    last_loss = None

    # Transició pendent de p0 per a cada env (persistent entre steps)
    last_p0_sa = [None] * num_envs

    pbar = trange(1, num_updates + 1, desc='DQN')
    for update in pbar:
        for step in range(NUM_STEPS):
            global_step += num_envs

            actions = np.zeros(num_envs, dtype=np.int32)
            prev_flat = [None] * num_envs
            is_learning = np.zeros(num_envs, dtype=bool)

            for i in range(num_envs):
                s = current_states[i]
                active_pid = s['raw_obs']['id_jugador']
                opp = opp_map[i]
                learner_pid = 1 - opp['pid']

                if active_pid == learner_pid:
                    flat = make_flat_state(s)
                    prev_flat[i] = flat
                    actions[i] = dqn.step(flat)
                    is_learning[i] = True
                else:
                    if opp['type'] == 'random':
                        actions[i], _ = rand_opp.eval_step(s)
                    elif opp['type'] == 'regles':
                        actions[i], _ = regles_opp.eval_step(s)
                    else:
                        flat = make_flat_state(s)
                        actions[i] = dqn_polyak.step(flat)

            next_states_players, rewards_list, dones_list = vec_env.step(actions)

            devnull = io.StringIO()
            for i in range(num_envs):
                opp = opp_map[i]
                learner_pid = 1 - opp['pid']
                flat_next = make_flat_state(next_states_players[i][0])
                reward_p0 = rewards_list[i][learner_pid]
                done = dones_list[i]

                if done:
                    games_played += 1
                    # Joc acabat en torn de l'oponent → completar la transició pendent de p0
                    if not is_learning[i] and last_p0_sa[i] is not None:
                        prev_obs, prev_act = last_p0_sa[i]
                        with contextlib.redirect_stdout(devnull):
                            dqn.feed((prev_obs, prev_act, reward_p0, flat_next, True))
                    last_p0_sa[i] = None

                if is_learning[i]:
                    with contextlib.redirect_stdout(devnull):
                        dqn.feed((prev_flat[i], actions[i], reward_p0, flat_next, done))
                    last_p0_sa[i] = None if done else (prev_flat[i], actions[i])

            current_states = [sp[0] for sp in next_states_players]

            # Actualització Polyak
            if global_step % DQN_POLYAK_FREQ < num_envs:
                with torch.no_grad():
                    for p, tp in zip(
                        dqn.q_estimator.qnet.parameters(),
                        dqn_polyak.q_estimator.qnet.parameters()
                    ):
                        tp.data.lerp_(p.data, DQN_POLYAK_TAU)

        # Avaluació
        if global_step % eval_every < (num_envs * NUM_STEPS):
            wr_r, wr_g, metric = evaluar_agent(dqn, ENV_CONFIG, regles_eval)
            elapsed = time.time() - t0
            append_log(log_path, global_step, games_played, last_loss, wr_r, wr_g, metric, elapsed)
            if metric > best_metric:
                best_metric = metric
                torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'best.pt')
                tqdm.write(f'[DQN step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
            else:
                tqdm.write(f'[DQN step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}%')

        pbar.set_postfix({'step': global_step, 'WR_g': f'{wr_g if "wr_g" in locals() else 0.0:.1f}%'})

    vec_env.close()
    # Guardar model
    best = save_dir / 'best.pt'
    if best.exists():
        shutil.copy2(best, save_dir / 'final.pt')
    else:
        torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'final.pt')
    print(f'[DQN] Entrenament complet. Millor metric: {best_metric:.2f}%')


# NFSP
def run_nfsp(save_dir, total_timesteps, device, num_envs_override=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    num_envs = num_envs_override or NUM_ENVS_NFSP
    eval_every = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    eps_decay = int(total_timesteps * 0.8)
    q_memory = min(NFSP_Q_REPLAY, 500_000) if num_envs <= 4 else NFSP_Q_REPLAY
    reservoir = min(NFSP_RESERVOIR, 500_000) if num_envs <= 4 else NFSP_RESERVOIR
    warmup = min(NFSP_WARMUP, q_memory // 10)

    def make_nfsp():
        return NFSPAgent(
            num_actions=N_ACTIONS,
            state_shape=OBS_DIM,
            hidden_layers_sizes=HIDDEN_LAYERS,
            q_mlp_layers=HIDDEN_LAYERS,
            rl_learning_rate=NFSP_RL_LR,
            sl_learning_rate=NFSP_SL_LR,
            batch_size=NFSP_BATCH,
            reservoir_buffer_capacity=reservoir,
            q_replay_memory_size=q_memory,
            q_update_target_estimator_every=NFSP_Q_UPDATE,
            anticipatory_param=NFSP_ETA,
            q_epsilon_decay_steps=eps_decay,
            q_epsilon_end=NFSP_EPS_MIN,
            q_replay_memory_init_size=warmup,
            q_batch_size=NFSP_BATCH,
            q_train_every=NFSP_Q_TRAIN_EVERY,
            device=device,
        )

    p0 = make_nfsp()  # learner principal
    p1 = make_nfsp()  # oponent self-play

    rand_opp    = RandomAgent(num_actions=N_ACTIONS)
    regles_opp  = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)

    opp_map = build_opponent_map(num_envs)
    vec_env = SubprocVecEnv(num_envs, ENV_CONFIG)

    results = vec_env.reset_all()
    current_states = [r[0] for r in results]

    # Inicialitzar polítiques
    p0.sample_episode_policy()
    p1.sample_episode_policy()

    num_updates = total_timesteps // (num_envs * NUM_STEPS)
    global_step = 0
    games_played = 0
    best_metric = -1.0
    t0 = time.time()
    wr_g = 0.0

    # Transicions pendents (persistent entre steps) per capturar recompenses terminals
    last_p0_sa = [None] * num_envs
    last_p1_sa = [None] * num_envs   # ídem per p1 (selfplay envs)

    pbar = trange(1, num_updates + 1, desc='NFSP')
    for update in pbar:
        for step in range(NUM_STEPS):
            global_step += num_envs

            actions = np.zeros(num_envs, dtype=np.int32)
            prev_flat_p0 = [None] * num_envs
            prev_flat_p1 = [None] * num_envs
            step_for_p0  = np.zeros(num_envs, dtype=bool)
            step_for_p1  = np.zeros(num_envs, dtype=bool)

            for i in range(num_envs):
                s = current_states[i]
                active_pid = s['raw_obs']['id_jugador']
                opp = opp_map[i]
                learner_pid = 1 - opp['pid']

                if opp['type'] == 'selfplay':
                    if active_pid == learner_pid:
                        flat = make_flat_state(s)
                        prev_flat_p0[i] = flat
                        actions[i] = p0.step(flat)
                        step_for_p0[i] = True
                    else:
                        flat = make_flat_state(s)
                        prev_flat_p1[i] = flat
                        actions[i] = p1.step(flat)
                        step_for_p1[i] = True
                else:
                    if active_pid == learner_pid:
                        flat = make_flat_state(s)
                        prev_flat_p0[i] = flat
                        actions[i] = p0.step(flat)
                        step_for_p0[i] = True
                    else:
                        if opp['type'] == 'random':
                            actions[i], _ = rand_opp.eval_step(s)
                        else:
                            actions[i], _ = regles_opp.eval_step(s)

            next_states_players, rewards_list, dones_list = vec_env.step(actions)

            devnull = io.StringIO()
            for i in range(num_envs):
                opp = opp_map[i]
                learner_pid = 1 - opp['pid']
                opp_pid     = opp['pid']
                flat_next   = make_flat_state(next_states_players[i][0])
                reward_p0   = rewards_list[i][learner_pid]
                reward_p1   = rewards_list[i][opp_pid]
                done        = dones_list[i]

                if done:
                    games_played += 1
                    # Completar transició pendent de p0 si el joc va acabar en torn de l'oponent
                    if not step_for_p0[i] and last_p0_sa[i] is not None:
                        prev_obs, prev_act = last_p0_sa[i]
                        with contextlib.redirect_stdout(devnull):
                            p0.feed((prev_obs, prev_act, reward_p0, flat_next, True))
                    last_p0_sa[i] = None
                    # Ídem per p1 (selfplay)
                    if opp['type'] == 'selfplay' and not step_for_p1[i] and last_p1_sa[i] is not None:
                        prev_obs, prev_act = last_p1_sa[i]
                        with contextlib.redirect_stdout(devnull):
                            p1.feed((prev_obs, prev_act, reward_p1, flat_next, True))
                    last_p1_sa[i] = None
                    # Resample política NFSP per al nou episodi
                    p0.sample_episode_policy()
                    if opp['type'] == 'selfplay':
                        p1.sample_episode_policy()

                if step_for_p0[i]:
                    with contextlib.redirect_stdout(devnull):
                        p0.feed((prev_flat_p0[i], actions[i], reward_p0, flat_next, done))
                    last_p0_sa[i] = None if done else (prev_flat_p0[i], actions[i])

                if step_for_p1[i]:
                    with contextlib.redirect_stdout(devnull):
                        p1.feed((prev_flat_p1[i], actions[i], reward_p1, flat_next, done))
                    last_p1_sa[i] = None if done else (prev_flat_p1[i], actions[i])

            current_states = [sp[0] for sp in next_states_players]

        # Avaluació
        if global_step % eval_every < (num_envs * NUM_STEPS):
            wr_r, wr_g, metric = evaluar_agent(p0, ENV_CONFIG, regles_eval)
            elapsed = time.time() - t0
            append_log(log_path, global_step, games_played, None, wr_r, wr_g, metric, elapsed)
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
                    'sl_net': p0.policy_network.state_dict(),
                }, save_dir / 'best.pt')
                tqdm.write(f'[NFSP step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
            else:
                tqdm.write(f'[NFSP step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}%')

        pbar.set_postfix({'step': global_step, 'WR_g': f'{wr_g:.1f}%'})

    vec_env.close()
    best = save_dir / 'best.pt'
    if best.exists():
        shutil.copy2(best, save_dir / 'final.pt')
    else:
        torch.save({'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
                    'sl_net': p0.policy_network.state_dict()},
                   save_dir / 'final.pt')
    print(f'[NFSP] Entrenament complet. Millor metric: {best_metric:.2f}%')


# PPO (Stable-Baselines3 MaskablePPO)
def run_ppo(save_dir, total_timesteps, device, num_envs_override=None):
    """
    Entrenament PPO usant MaskablePPO de sb3_contrib.

    Es creen NUM_ENVS_PPO entorns Gymnasium (TrucGymEnv) amb ActionMasker,
    alternant la posició de l'agent aprenent (learner_pid 0/1) i amb la
    distribució d'oponents: 5% Random, 95% AgentRegles.
    El self-play s'implementa alternant learner_pid perquè el model aprengui
    les dues posicions simultàniament.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    num_envs = num_envs_override or NUM_ENVS_PPO
    n_steps = min(NUM_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)

    n_random = max(1, int(num_envs * PCT_RANDOM))

    # Factories d'entorns Gymnasium amb ActionMasker per a MaskablePPO.
    # - n_random entorns amb Random opponent
    # - resta amb AgentRegles opponent
    # - learner_pid alterna per aprendre les dues posicions (self-play implícit)
    def _make_env_fn(opponent_type: str, learner_pid: int, seed: int):
        def _init():
            import sys, os
            sys.path.insert(0, os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..')
            ))
            from joc.entorn.gym_env import TrucGymEnv
            from sb3_contrib.common.wrappers import ActionMasker
            from rlcard.agents import RandomAgent as _RandomAgent
            from RL.models.model_propi.agent_regles import AgentRegles as _AgentRegles

            cfg = ENV_CONFIG.copy()
            cfg['seed'] = seed
            if opponent_type == 'random':
                opp = _RandomAgent(num_actions=N_ACTIONS)
            else:
                opp = _AgentRegles(num_actions=N_ACTIONS, seed=seed + 1000)
            env = TrucGymEnv(cfg, opponent=opp, learner_pid=learner_pid)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    env_fns = []
    for i in range(num_envs):
        opp_type = 'random' if i < n_random else 'regles'
        learner_pid = i % 2  # alterna posició aprenent
        env_fns.append(_make_env_fn(opp_type, learner_pid, SEED + i))

    vec_env = SB3SubprocVecEnv(env_fns)

    # Arquitectura de la política: 2 capes ocultes [256, 256] per a actor i crític
    policy_kwargs = dict(
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        'MlpPolicy',
        vec_env,
        learning_rate=PPO_LR,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=PPO_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP,
        ent_coef=PPO_ENT,
        vf_coef=PPO_VF,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
    )

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]  # mutable per al callback
    t0 = time.time()

    class _EvalLogCallback(BaseCallback):
        """Avalua el model cada eval_every passos i registra mètriques al CSV."""

        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= eval_every:
                self._last_eval = self.num_timesteps
                eval_agent = SB3PPOEvalAgent(self.model)
                wr_r, wr_g, metric = evaluar_agent(eval_agent, ENV_CONFIG, regles_eval)
                elapsed = time.time() - t0
                # loss no disponible directament; usem ep_rew_mean si existeix
                ep_rew = (self.locals.get('infos') or [{}])[0].get('episode', {}).get('r', None)
                append_log(log_path, self.num_timesteps, 0, ep_rew,
                           wr_r, wr_g, metric, elapsed)
                if metric > best_metric[0]:
                    best_metric[0] = metric
                    self.model.save(str(save_dir / 'best'))
                    print(f'[PPO step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
                else:
                    print(f'[PPO step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}%')
            return True

    model.learn(
        total_timesteps=total_timesteps,
        callback=_EvalLogCallback(),
        progress_bar=True,
    )

    vec_env.close()

    # Guardar model final
    model.save(str(save_dir / 'final'))
    best_path = save_dir / 'best.zip'
    if not best_path.exists():
        model.save(str(save_dir / 'best'))
    print(f'[PPO SB3] Entrenament complet. Millor metric: {best_metric[0]:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Entrenament Comparatiu DQN/NFSP/PPO')
    parser.add_argument('--agent', choices=['dqn', 'nfsp', 'ppo'], required=True)
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--num_envs', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f'[GPU] {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('[CPU] GPU no disponible.')

    if args.save_dir:
        save_dir = args.save_dir
    else:
        ts = datetime.now().strftime('%d%m_%H%Mh')
        save_dir = str(Path(__file__).parent / 'registres' / f'{args.agent}_{ts}')

    print(f'[{args.agent.upper()}] Timesteps: {args.total_timesteps:,} | Guardat a: {save_dir}')

    t_start = time.time()
    if args.agent == 'dqn':
        run_dqn(save_dir, args.total_timesteps, device, args.num_envs)
    elif args.agent == 'nfsp':
        run_nfsp(save_dir, args.total_timesteps, device, args.num_envs)
    elif args.agent == 'ppo':
        run_ppo(save_dir, args.total_timesteps, device, args.num_envs)

    total_time = time.time() - t_start
    print(f'\nTemps total: {total_time:.0f}s ({total_time/3600:.2f}h)')


if __name__ == '__main__':
    main()
