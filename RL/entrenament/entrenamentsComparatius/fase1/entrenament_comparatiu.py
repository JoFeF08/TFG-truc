"""
Script d'entrenament comparatiu (Fase 1)
-----------------------------------------
Cada algorisme s'entrena amb les tècniques que millor s'adapten a la seva naturalesa.

  DQN RLCard  → off-policy, loop seqüencial sobre TrucEnv, replay buffer gran + Polyak self-play
  NFSP RLCard → off-policy + imitació, loop seqüencial, reservoir buffer + self-play natiu
  DQN SB3     → off-policy, TrucGymEnv + SB3, replica del DQN però amb implementació SB3
  PPO SB3     → on-policy, 48 TrucGymEnv paral·lels via SB3SubprocVecEnv

El nexe comú que permet la comparació:
  - evaluar_agent()  → mateixa funció per als quatre (vs Random + vs AgentRegles)
  - training_log.csv → format idèntic: step, games_played, loss, wr_random, wr_regles, metric, elapsed_s
  - metric = 0.25 * wr_random + 0.75 * wr_regles
  - Mateixa arquitectura de xarxa [256, 256] per a tots
  - Mateixa dimensió d'observació (calculada dinàmicament)
"""

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
from stable_baselines3 import PPO, DQN as SB3DQN

from joc.entorn import TrucEnv
from joc.entorn.cartes_accions import ACTION_LIST
from joc.entorn.gym_env import TrucGymEnv

from RL.models.model_propi.agent_regles import AgentRegles

# Silenciar logs de llibreries externes
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)

# Constants compartides
TOTAL_TIMESTEPS = 24_000_000
SEED            = 42

# Càlcul dinàmic obs
_tmp_env  = TrucEnv({'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 24, 'seed': SEED})
_dummy_st, _ = _tmp_env.reset()
OBS_DIM   = np.concatenate(
    [_dummy_st['obs']['obs_cartes'].flatten(), _dummy_st['obs']['obs_context']], axis=0
).shape[0]

N_ACTIONS     = len(ACTION_LIST)
HIDDEN_LAYERS = [256, 256]   # igual per als quatre algorismes

ENV_CONFIG = {
    'num_jugadors':    2,
    'cartes_jugador':  3,
    'puntuacio_final': 24,
    'seed':            SEED,
    'verbose':         False,
}

# Param avaluació 
EVAL_EVERY_STEPS  = 500_000
EVAL_GAMES_RANDOM = 50
EVAL_GAMES_REGLES = 100

# DQN RLCard (off-policy)
DQN_LR          = 1e-4
DQN_BATCH       = 512
DQN_MEMORY      = 2_000_000
DQN_WARMUP      = 100_000
DQN_TRAIN_EVERY = 128
DQN_UPDATE_TGT  = 2_000
DQN_EPS_MIN     = 0.05
DQN_POLYAK_TAU  = 0.05
DQN_POLYAK_FREQ = 5_000

# Distribució d'oponents
DQN_PCT_RANDOM  = 0.10
DQN_PCT_REGLES  = 0.60
# restant ~30% → self-play Polyak



# NFSP RLCard (off-policy + imitació)
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

# Distribució d'oponents per NFSP
NFSP_PCT_RANDOM    = 0.10
NFSP_PCT_REGLES    = 0.60
# restant → self-play NFSP natiu



# DQN SB3 (off-policy, TrucGymEnv)
SB3DQN_LR            = 1e-4
SB3DQN_BATCH         = 512
SB3DQN_MEMORY        = 2_000_000
SB3DQN_WARMUP        = 100_000
SB3DQN_TRAIN_FREQ    = 4        # actualitza cada 4 steps
SB3DQN_TARGET_UPDATE = 2_000
SB3DQN_EPS_START     = 1.0
SB3DQN_EPS_END       = 0.05
SB3DQN_EPS_FRACTION  = 0.8      # fracció de timesteps per al decay
SB3DQN_GAMMA         = 0.99

# PPO (on-policy, SB3)
NUM_ENVS_PPO    = 48

PPO_LR          = 3e-4
PPO_GAMMA       = 0.995
PPO_GAE_LAMBDA  = 0.95
PPO_CLIP        = 0.2
PPO_ENT         = 0.01
PPO_VF          = 0.5
PPO_EPOCHS      = 7
PPO_MINIBATCH   = 1024
PPO_N_STEPS     = 256

PPO_PCT_RANDOM  = 0.05


# Adaptador SB3
class SB3EvalAgent:
    """
    Adapta qualsevol model SB3 (PPO, DQN) a la interfície eval_step()
    que usa evaluar_agent() per avaluar qualsevol agent de forma uniforme.
    """
    use_raw = False

    def __init__(self, model, n_actions: int = N_ACTIONS):
        self.model       = model
        self.num_actions = n_actions

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        else:
            obs_flat = np.asarray(obs, dtype=np.float32)

        action, _ = self.model.predict(obs_flat[np.newaxis], deterministic=True)
        return int(action[0]), {}


def flatten_obs(state) -> np.ndarray:
    obs = state['obs']
    if isinstance(obs, dict):
        return np.concatenate(
            [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


def make_flat_state(state) -> dict:
    return {
        'obs':           flatten_obs(state),
        'legal_actions': state['legal_actions'],
        'raw_obs':       state.get('raw_obs', {}),
    }


def wrap_env_aplanat(env):
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


def _pick_opponent(rng, rand_opp, regles_opp, polyak_agent, pct_random, pct_regles):
    """Tria l'agent oponent per a un episodi basat en les probabilitats configurades."""
    r = rng.random()
    if r < pct_random:
        return rand_opp, 'random'
    elif r < pct_random + pct_regles:
        return regles_opp, 'regles'
    else:
        return polyak_agent, 'selfplay'


# avaluació comú
def evaluar_agent(agent, env_config: dict, regles_agent,
                  n_random: int = EVAL_GAMES_RANDOM,
                  n_regles: int = EVAL_GAMES_REGLES):
    """
    Avalua qualsevol agent que implementi eval_step()
    (DQNAgent, NFSPAgent, SB3EvalAgent) contra RandomAgent i AgentRegles.

    Retorna: (wr_random%, wr_regles%, metric)
    metric = 0.25 * wr_random + 0.75 * wr_regles
    """
    eval_env = wrap_env_aplanat(TrucEnv(env_config))
    rand_opp = RandomAgent(num_actions=N_ACTIONS)

    wins_r = 0
    for i in range(n_random):
        if i % 2 == 0:
            eval_env.set_agents([agent, rand_opp]); pid = 0
        else:
            eval_env.set_agents([rand_opp, agent]); pid = 1
        _, payoffs = eval_env.run(is_training=False)
        if payoffs[pid] > 0:
            wins_r += 1
    wr_random = 100.0 * wins_r / n_random

    wins_g = 0
    for i in range(n_regles):
        if i % 2 == 0:
            eval_env.set_agents([agent, regles_agent]); pid = 0
        else:
            eval_env.set_agents([regles_agent, agent]); pid = 1
        _, payoffs = eval_env.run(is_training=False)
        if payoffs[pid] > 0:
            wins_g += 1
    wr_regles = 100.0 * wins_g / n_regles

    metric = 0.25 * wr_random + 0.75 * wr_regles
    return wr_random, wr_regles, metric


# loggs
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


# entorn gym
def _make_gym_env_fn(opponent_type: str, learner_pid: int, seed: int):
    """Retorna una funció factory per SB3SubprocVecEnv / Monitor."""
    def _init():
        import sys, os
        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..')
        ))
        from joc.entorn.gym_env import TrucGymEnv
        from rlcard.agents import RandomAgent as _RandomAgent
        from RL.models.model_propi.agent_regles import AgentRegles as _AgentRegles

        cfg = ENV_CONFIG.copy()
        cfg['seed'] = seed
        opp = _RandomAgent(num_actions=N_ACTIONS) if opponent_type == 'random' \
            else _AgentRegles(num_actions=N_ACTIONS, seed=seed + 1000)
        return TrucGymEnv(cfg, opponent=opp, learner_pid=learner_pid)
    return _init


# DQN RLCard
def run_dqn(save_dir, total_timesteps, device):
    """
    DQN de RLCard sobre un sol TrucEnv seqüencial.
    El replay buffer gran garanteix diversitat de mostres sense paral·lelisme.
    Oponent per episodi: 10% Random, 60% AgentRegles, 30% self-play Polyak.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    eps_decay = int(total_timesteps * 0.8)

    dqn = DQNAgent(
        num_actions=N_ACTIONS,
        state_shape=OBS_DIM,
        mlp_layers=HIDDEN_LAYERS,
        learning_rate=DQN_LR,
        batch_size=DQN_BATCH,
        replay_memory_size=DQN_MEMORY,
        replay_memory_init_size=DQN_WARMUP,
        update_target_estimator_every=DQN_UPDATE_TGT,
        epsilon_decay_steps=eps_decay,
        epsilon_end=DQN_EPS_MIN,
        train_every=DQN_TRAIN_EVERY,
        device=device,
    )

    # Oponent self-play Polyak
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
    dqn_polyak.q_estimator.qnet.load_state_dict(dqn.q_estimator.qnet.state_dict())

    rand_opp    = RandomAgent(num_actions=N_ACTIONS)
    regles_opp  = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    rng         = np.random.default_rng(SEED)

    env          = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_every   = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    global_step  = 0
    games_played = 0
    best_metric  = -1.0
    wr_r = wr_g  = 0.0
    t0           = time.time()
    devnull      = io.StringIO()

    pbar = trange(total_timesteps, desc='DQN-RLCard')
    while global_step < total_timesteps:
        # Triar oponent per a aquest episodi
        opp, opp_type = _pick_opponent(rng, rand_opp, regles_opp, dqn_polyak,
                                       DQN_PCT_RANDOM, DQN_PCT_REGLES)
        learner_pid = rng.integers(0, 2)   # alterna posició

        if learner_pid == 0:
            env.set_agents([dqn, opp])
        else:
            env.set_agents([opp, dqn])

        # Loop manual: control total del format de transicions per a dqn.feed()
        state, player_id = env.reset()
        n_ts = 0
        pending = None   # (flat, action) pendent de tancar quan acabi l'oponent

        while player_id is not None:
            flat = make_flat_state(state)

            if player_id == learner_pid:
                action = dqn.step(flat)
                next_state, next_pid = env.step(action)
                done = (next_pid is None)
                reward = float(env.game.get_payoffs()[learner_pid]) if done else 0.0
                next_flat = flat if done else make_flat_state(next_state)
                with contextlib.redirect_stdout(devnull):
                    dqn.feed((flat, action, reward, next_flat, done))
                pending = None
                n_ts += 1
            else:
                opp_action, _ = opp.eval_step(state)
                next_state, next_pid = env.step(opp_action)
                done = (next_pid is None)
                if done and pending is not None:
                    prev_flat, prev_action = pending
                    reward = float(env.game.get_payoffs()[learner_pid])
                    with contextlib.redirect_stdout(devnull):
                        dqn.feed((prev_flat, prev_action, reward, prev_flat, True))
                    n_ts += 1
                pending = None

            state = next_state
            player_id = next_pid

        global_step += n_ts
        pbar.update(n_ts)
        games_played += 1

        # Actualització Polyak
        if global_step % DQN_POLYAK_FREQ < len(learner_traj):
            with torch.no_grad():
                for p, tp in zip(
                    dqn.q_estimator.qnet.parameters(),
                    dqn_polyak.q_estimator.qnet.parameters()
                ):
                    tp.data.lerp_(p.data, DQN_POLYAK_TAU)

        # Avaluació
        if global_step % eval_every < len(learner_traj):
            wr_r, wr_g, metric = evaluar_agent(dqn, ENV_CONFIG, regles_eval)
            elapsed = time.time() - t0
            append_log(log_path, global_step, games_played, None, wr_r, wr_g, metric, elapsed)
            if metric > best_metric:
                best_metric = metric
                torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'best.pt')
                tqdm.write(f'[DQN-RLCard step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
            else:
                tqdm.write(f'[DQN-RLCard step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}%')

        pbar.set_postfix({'WR_rand': f'{wr_r:.1f}%', 'WR_regl': f'{wr_g:.1f}%'})

    pbar.close()
    best = save_dir / 'best.pt'
    if best.exists():
        shutil.copy2(best, save_dir / 'final.pt')
    else:
        torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'final.pt')
    print(f'[DQN-RLCard] Entrenament complet. Millor metric: {best_metric:.2f}%')


# NFSP RLCard
def run_nfsp(save_dir, total_timesteps, device):
    """
    NFSP sobre un sol TrucEnv seqüencial.
    Inclou self-play intern, reservoir buffer i anticipatory parameter.
    Oponent per episodi: 10% Random, 60% AgentRegles, 30% self-play natiu.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    eps_decay = int(total_timesteps * 0.8)

    def make_nfsp():
        return NFSPAgent(
            num_actions=N_ACTIONS,
            state_shape=OBS_DIM,
            hidden_layers_sizes=HIDDEN_LAYERS,
            q_mlp_layers=HIDDEN_LAYERS,
            rl_learning_rate=NFSP_RL_LR,
            sl_learning_rate=NFSP_SL_LR,
            batch_size=NFSP_BATCH,
            reservoir_buffer_capacity=NFSP_RESERVOIR,
            q_replay_memory_size=NFSP_Q_REPLAY,
            q_update_target_estimator_every=NFSP_Q_UPDATE,
            anticipatory_param=NFSP_ETA,
            q_epsilon_decay_steps=eps_decay,
            q_epsilon_end=NFSP_EPS_MIN,
            q_replay_memory_init_size=NFSP_WARMUP,
            q_batch_size=NFSP_BATCH,
            q_train_every=NFSP_Q_TRAIN_EVERY,
            device=device,
        )

    p0 = make_nfsp()  # learner principal
    p1 = make_nfsp()  # oponent self-play

    rand_opp    = RandomAgent(num_actions=N_ACTIONS)
    regles_opp  = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    rng         = np.random.default_rng(SEED)

    env          = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_every   = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    global_step  = 0
    games_played = 0
    best_metric  = -1.0
    wr_r = wr_g  = 0.0
    t0           = time.time()
    devnull      = io.StringIO()

    pbar = trange(total_timesteps, desc='NFSP')
    while global_step < total_timesteps:
        r = rng.random()
        learner_pid = int(rng.integers(0, 2))

        if r < NFSP_PCT_RANDOM:
            opp = rand_opp
        elif r < NFSP_PCT_RANDOM + NFSP_PCT_REGLES:
            opp = regles_opp
        else:
            opp = p1   # self-play natiu

        if learner_pid == 0:
            env.set_agents([p0, opp])
        else:
            env.set_agents([opp, p0])

        p0.sample_episode_policy()
        if opp is p1:
            p1.sample_episode_policy()

        # Loop manual
        state, player_id = env.reset()
        n_ts = 0

        while player_id is not None:
            flat = make_flat_state(state)

            if player_id == learner_pid:
                action = p0.step(flat)
                next_state, next_pid = env.step(action)
                done = (next_pid is None)
                reward = float(env.game.get_payoffs()[learner_pid]) if done else 0.0
                next_flat = flat if done else make_flat_state(next_state)
                with contextlib.redirect_stdout(devnull):
                    p0.feed((flat, action, reward, next_flat, done))
                n_ts += 1
            else:
                if opp is p1:
                    opp_action = p1.step(flat)
                else:
                    opp_action, _ = opp.eval_step(state)
                next_state, next_pid = env.step(opp_action)
                done = (next_pid is None)
                if opp is p1:
                    reward_p1 = float(env.game.get_payoffs()[1 - learner_pid]) if done else 0.0
                    next_flat_p1 = flat if done else make_flat_state(next_state)
                    with contextlib.redirect_stdout(devnull):
                        p1.feed((flat, opp_action, reward_p1, next_flat_p1, done))

            state = next_state
            player_id = next_pid

        global_step += n_ts
        pbar.update(n_ts)
        games_played += 1

        # Avaluació
        if global_step % eval_every < max(n_ts, 1):
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

        pbar.set_postfix({'WR_rand': f'{wr_r:.1f}%', 'WR_regl': f'{wr_g:.1f}%'})

    pbar.close()
    best = save_dir / 'best.pt'
    if best.exists():
        shutil.copy2(best, save_dir / 'final.pt')
    else:
        torch.save({
            'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
            'sl_net': p0.policy_network.state_dict(),
        }, save_dir / 'final.pt')
    print(f'[NFSP] Entrenament complet. Millor metric: {best_metric:.2f}%')


# DQN SB3
def run_dqn_sb3(save_dir, total_timesteps, device):
    """
    DQN de Stable-Baselines3 sobre TrucGymEnv.
    Permet comparar directament la implementació de RLCard vs la de SB3
    sota la mateixa arquitectura i protocol d'avaluació.
    Oponent: 5% Random, 95% AgentRegles. Posició aprenent alterna per episodi.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    eval_every  = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    eps_decay_f = SB3DQN_EPS_FRACTION

    env = _make_gym_env_fn('regles', learner_pid=0, seed=SEED)()

    policy_kwargs = dict(net_arch=HIDDEN_LAYERS)

    model = SB3DQN(
        'MlpPolicy', env,
        learning_rate=SB3DQN_LR,
        batch_size=SB3DQN_BATCH,
        buffer_size=SB3DQN_MEMORY,
        learning_starts=SB3DQN_WARMUP,
        train_freq=SB3DQN_TRAIN_FREQ,
        target_update_interval=SB3DQN_TARGET_UPDATE,
        exploration_fraction=eps_decay_f,
        exploration_initial_eps=SB3DQN_EPS_START,
        exploration_final_eps=SB3DQN_EPS_END,
        gamma=SB3DQN_GAMMA,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
    )

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    t0          = time.time()

    class _EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= eval_every:
                self._last_eval = self.num_timesteps
                eval_agent      = SB3EvalAgent(self.model)
                wr_r, wr_g, metric = evaluar_agent(eval_agent, ENV_CONFIG, regles_eval)
                elapsed = time.time() - t0
                append_log(log_path, self.num_timesteps, 0, None, wr_r, wr_g, metric, elapsed)
                if metric > best_metric[0]:
                    best_metric[0] = metric
                    self.model.save(str(save_dir / 'best'))
                    print(f'[DQN-SB3 step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
                else:
                    print(f'[DQN-SB3 step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}%')
            return True

    model.learn(
        total_timesteps=total_timesteps,
        callback=_EvalCallback(),
        progress_bar=True,
    )

    env.close()
    model.save(str(save_dir / 'final'))
    if not (save_dir / 'best.zip').exists():
        model.save(str(save_dir / 'best'))
    print(f'[DQN-SB3] Entrenament complet. Millor metric: {best_metric[0]:.2f}%')


# PPO SB3
def run_ppo(save_dir, total_timesteps, device, num_envs_override=None):
    """
    PPO on-policy via Stable-Baselines3.
    Molts entorns paral·lels (SB3SubprocVecEnv + TrucGymEnv).
    Oponent gestionat internament: 5% Random, 95% AgentRegles.
    Posició aprenent alterna (learner_pid 0/1) per aprendre les dues posicions.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.csv'
    init_log(log_path)

    num_envs   = num_envs_override or NUM_ENVS_PPO
    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = min(EVAL_EVERY_STEPS, total_timesteps // 20)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)
    n_random   = max(1, int(num_envs * PPO_PCT_RANDOM))

    env_fns = [
        _make_gym_env_fn('random' if i < n_random else 'regles', i % 2, SEED + i)
        for i in range(num_envs)
    ]

    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        'MlpPolicy', vec_env,
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
    best_metric = [-1.0]
    t0          = time.time()

    class _EvalLogCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= eval_every:
                self._last_eval = self.num_timesteps
                eval_agent      = SB3EvalAgent(self.model)
                wr_r, wr_g, metric = evaluar_agent(eval_agent, ENV_CONFIG, regles_eval)
                elapsed = time.time() - t0
                ep_rew  = (self.locals.get('infos') or [{}])[0].get('episode', {}).get('r', None)
                append_log(log_path, self.num_timesteps, 0, ep_rew, wr_r, wr_g, metric, elapsed)
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
    model.save(str(save_dir / 'final'))
    if not (save_dir / 'best.zip').exists():
        model.save(str(save_dir / 'best'))
    print(f'[PPO] Entrenament complet. Millor metric: {best_metric[0]:.2f}%')



def main():
    parser = argparse.ArgumentParser(description='Entrenament Comparatiu DQN/NFSP/PPO')
    parser.add_argument('--agent',
                        choices=['dqn', 'dqn_sb3', 'nfsp', 'ppo'],
                        required=True,
                        help='dqn=RLCard DQN (seqüencial) | dqn_sb3=SB3 DQN | nfsp=NFSP RLCard | ppo=SB3 PPO')
    parser.add_argument('--total_timesteps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--num_envs',        type=int, default=None,
                        help='Només aplica a PPO (per defecte 48)')
    parser.add_argument('--save_dir',        type=str, default=None)
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
        ts       = datetime.now().strftime('%d%m_%H%Mh')
        save_dir = str(Path(__file__).parent / 'registres' / f'{args.agent}_{ts}')

    print(f'[{args.agent.upper()}] Timesteps: {args.total_timesteps:,} | Guardat a: {save_dir}')

    t_start = time.time()
    if args.agent == 'dqn':
        run_dqn(save_dir, args.total_timesteps, device)
    elif args.agent == 'dqn_sb3':
        run_dqn_sb3(save_dir, args.total_timesteps, device)
    elif args.agent == 'nfsp':
        run_nfsp(save_dir, args.total_timesteps, device)
    elif args.agent == 'ppo':
        run_ppo(save_dir, args.total_timesteps, device, args.num_envs)

    total_time = time.time() - t_start
    print(f'\nTemps total: {total_time:.0f}s ({total_time/3600:.2f}h)')


if __name__ == '__main__':
    main()
