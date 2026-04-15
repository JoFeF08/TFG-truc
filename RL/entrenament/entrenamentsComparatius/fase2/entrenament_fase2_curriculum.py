"""
Script d'entrenament comparatiu (Fase 2 — Curriculum Learning)
---------------------------------------------------------------
Investiga l'impacte del curriculum learning (mans → partides) sobre els 4 agents.

Modes:
  control    → 24M steps directament en partides, sense self-play
  curriculum → 12M steps en mans aïllades + 12M steps finetune en partides

Diferències respecte Fase 1:
  - Sense self-play per a cap agent (oponents: 10% Random + 90% AgentRegles)
  - En mode curriculum, els agents pre-entrenats en mans es carreguen per al finetune
  - DQN SB3: epsilon reduït al finetune (política ja inicialitzada)
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
from RL.tools.obs_utils import flatten_obs
from joc.entorn_ma.env_ma import TrucEnvMa
from joc.entorn_ma.gym_env_ma import TrucGymEnvMa

from RL.models.model_propi.agent_regles import AgentRegles

logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)

# Constants compartides
SEED = 42

_tmp_env  = TrucEnv({'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 24, 'seed': SEED})
_dummy_st, _ = _tmp_env.reset()
OBS_DIM = np.concatenate(
    [_dummy_st['obs']['obs_cartes'].flatten(), _dummy_st['obs']['obs_context']], axis=0
).shape[0]

N_ACTIONS     = len(ACTION_LIST)
HIDDEN_LAYERS = [256, 256]

ENV_CONFIG = {
    'num_jugadors':    2,
    'cartes_jugador':  3,
    'puntuacio_final': 24,
    'seed':            SEED,
    'verbose':         False,
}

ENV_CONFIG_MA = {
    'num_jugadors':    2,
    'cartes_jugador':  3,
    'puntuacio_final': 999,
    'seed':            SEED,
    'verbose':         False,
}

EVAL_EVERY_STEPS  = 500_000
EVAL_GAMES_RANDOM = 50
EVAL_GAMES_REGLES = 100

# tots els agents igual
PCT_RANDOM = 0.10
PCT_REGLES = 0.90

# DQN RLCard 
DQN_LR          = 1e-4
DQN_BATCH       = 512
DQN_MEMORY      = 2_000_000
DQN_WARMUP      = 100_000
DQN_TRAIN_EVERY = 128
DQN_UPDATE_TGT  = 2_000
DQN_EPS_MIN     = 0.05

# NFSP RLCard
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

# DQN SB3
SB3DQN_LR            = 1e-4
SB3DQN_BATCH         = 512
SB3DQN_MEMORY        = 2_000_000
SB3DQN_WARMUP        = 100_000
SB3DQN_TRAIN_FREQ    = 4
SB3DQN_TARGET_UPDATE = 2_000
SB3DQN_EPS_START     = 1.0
SB3DQN_EPS_END       = 0.05
SB3DQN_EPS_FRACTION  = 0.8
SB3DQN_GAMMA         = 0.99
SB3DQN_EPS_FINETUNE  = 0.10
SB3DQN_WARMUP_FT     = 10_000

# PPO SB3
NUM_ENVS_PPO  = 32
PPO_LR        = 3e-4
PPO_GAMMA     = 0.995
PPO_GAE       = 0.95
PPO_CLIP      = 0.2
PPO_ENT       = 0.01
PPO_VF        = 0.5
PPO_EPOCHS    = 7
PPO_MINIBATCH = 1024
PPO_N_STEPS   = 256


def make_flat_state(state) -> dict:
    return {
        'obs':           flatten_obs(state['obs']),
        'legal_actions': state['legal_actions'],
        'raw_obs':       state.get('raw_obs', {}),
    }


def wrap_env_aplanat(env):
    original = env._extract_state

    def _extract_patched(self, state):
        extracted = original(state)
        extracted['obs'] = flatten_obs(extracted['obs'])
        return extracted

    env._extract_state = types.MethodType(_extract_patched, env)
    return env


def init_log(log_path: Path) -> None:
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['step', 'games_played', 'loss',
             'eval_wr_random', 'eval_wr_regles', 'eval_metric', 'elapsed_s']
        )


def append_log(log_path, step, games, loss, wr_r, wr_g, metric, elapsed) -> None:
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            step, games,
            f'{loss:.6f}' if loss is not None else '',
            f'{wr_r:.2f}', f'{wr_g:.2f}', f'{metric:.2f}',
            f'{elapsed:.1f}',
        ])


def evaluar_agent(agent, env_config: dict, regles_agent,
                  n_random: int = EVAL_GAMES_RANDOM,
                  n_regles: int = EVAL_GAMES_REGLES):
    """Avalua l'agent contra Random i AgentRegles. Retorna (wr_random%, wr_regles%, metric)."""
    rand_eval = RandomAgent(num_actions=N_ACTIONS)
    wins_r = wins_g = 0

    def _agent_action(flat, raw_state):
        if hasattr(agent, 'step'):
            return agent.step(flat)
        if hasattr(agent, 'eval_step'):
            a, _ = agent.eval_step(raw_state)
            return a
        return _sb3_step(agent, flat, raw_state)

    for _ in range(n_random):
        env = TrucEnv(env_config)
        env = wrap_env_aplanat(env)
        pid = 0
        env.set_agents([agent, rand_eval])
        state, player_id = env.reset()
        while player_id is not None:
            flat = make_flat_state(state)
            if player_id == pid:
                action = _agent_action(flat, state)
                state, player_id = env.step(action)
            else:
                a, _ = rand_eval.eval_step(state)
                state, player_id = env.step(a)
        payoffs = env.game.get_payoffs()
        if payoffs[pid] > 0:
            wins_r += 1

    for _ in range(n_regles):
        env = TrucEnv(env_config)
        env = wrap_env_aplanat(env)
        pid = 0
        env.set_agents([agent, regles_agent])
        state, player_id = env.reset()
        while player_id is not None:
            flat = make_flat_state(state)
            if player_id == pid:
                action = _agent_action(flat, state)
                state, player_id = env.step(action)
            else:
                a, _ = regles_agent.eval_step(state)
                state, player_id = env.step(a)
        payoffs = env.game.get_payoffs()
        if payoffs[pid] > 0:
            wins_g += 1

    wr_r = 100.0 * wins_r / n_random
    wr_g = 100.0 * wins_g / n_regles
    metric = 0.25 * wr_r + 0.75 * wr_g
    return wr_r, wr_g, metric


def _sb3_step(model, flat_state: dict, raw_state: dict) -> int:
    """Crida predict() d'un model SB3 i retorna una acció legal."""
    obs = flat_state['obs']
    legal = list(raw_state['legal_actions'].keys())
    action, _ = model.predict(obs[np.newaxis], deterministic=True)
    action = int(action[0])
    if action not in legal:
        action = legal[0]
    return action


class SB3EvalAgent:
    """Adapta models SB3 a la interfície eval_step() per a evaluar_agent()."""
    use_raw = False

    def __init__(self, model):
        self.model = model
        self.num_actions = N_ACTIONS

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        else:
            obs_flat = np.asarray(obs, dtype=np.float32)
        legal = list(state['legal_actions'].keys())
        action, _ = self.model.predict(obs_flat[np.newaxis], deterministic=True)
        action = int(action[0])
        if action not in legal:
            action = legal[0]
        return action, {}


def _make_gym_env_fn(opponent_type: str, learner_pid: int, seed: int):
    """Retorna una factory per a TrucGymEnv (partides)."""
    def _init():
        cfg = ENV_CONFIG.copy()
        cfg['seed'] = seed
        from rlcard.agents import RandomAgent as _RA
        opp = _RA(num_actions=N_ACTIONS) if opponent_type == 'random' \
            else AgentRegles(num_actions=N_ACTIONS, seed=seed + 1000)
        return TrucGymEnv(cfg, opponent=opp, learner_pid=learner_pid)
    return _init


def _make_gym_env_ma_fn(opponent_type: str, learner_pid: int, seed: int):
    """Retorna una factory per a TrucGymEnvMa (mans)."""
    def _init():
        cfg = ENV_CONFIG_MA.copy()
        cfg['seed'] = seed
        from rlcard.agents import RandomAgent as _RA
        opp = _RA(num_actions=N_ACTIONS) if opponent_type == 'random' \
            else AgentRegles(num_actions=N_ACTIONS, seed=seed + 1000)
        return TrucGymEnvMa(cfg, opponent=opp, learner_pid=learner_pid)
    return _init


# DQN RLCard
def _dqn_rlcard_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
                     log_name: str, label: str, load_path: Path | None = None):
    """Loop d'entrenament genèric per DQN RLCard (mans o partides)."""
    log_path = save_dir / log_name
    init_log(log_path)

    eps_decay = int(min(timesteps, 24_000_000) * 0.8)

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

    if load_path is not None and load_path.exists():
        dqn.q_estimator.qnet.load_state_dict(
            torch.load(load_path, map_location=device)
        )
        print(f'[DQN-RLCard] Pesos carregats des de {load_path}')

    rand_opp    = RandomAgent(num_actions=N_ACTIONS)
    regles_opp  = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    rng         = np.random.default_rng(SEED)

    EnvClass = TrucEnvMa if use_mans else TrucEnv
    cfg      = ENV_CONFIG_MA if use_mans else ENV_CONFIG
    env      = wrap_env_aplanat(EnvClass(cfg))

    eval_every   = EVAL_EVERY_STEPS
    global_step  = 0
    games_played = 0
    best_metric  = -1.0
    wr_r = wr_g  = 0.0
    t0           = time.time()
    devnull      = io.StringIO()

    pbar = trange(timesteps, desc=label)
    while global_step < timesteps:
        r_opp = rng.random()
        opp = rand_opp if r_opp < PCT_RANDOM else regles_opp
        learner_pid = int(rng.integers(0, 2))

        if learner_pid == 0:
            env.set_agents([dqn, opp])
        else:
            env.set_agents([opp, dqn])

        state, player_id = env.reset()
        n_ts   = 0
        pending = None

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

        global_step  += n_ts
        games_played += 1
        pbar.update(n_ts)

        if global_step % eval_every < max(n_ts, 1):
            wr_r, wr_g, metric = evaluar_agent(dqn, ENV_CONFIG, regles_eval)
            elapsed = time.time() - t0
            append_log(log_path, global_step, games_played, None, wr_r, wr_g, metric, elapsed)
            best_pt = save_dir / f'best_{log_name.replace("training_log_", "").replace(".csv","")}.pt'
            if metric > best_metric:
                best_metric = metric
                torch.save(dqn.q_estimator.qnet.state_dict(), best_pt)
                tqdm.write(f'[{label} step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
            else:
                tqdm.write(f'[{label} step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}%')

        pbar.set_postfix({'WR_rand': f'{wr_r:.1f}%', 'WR_regl': f'{wr_g:.1f}%'})

    pbar.close()
    best_pt = save_dir / f'best_{log_name.replace("training_log_", "").replace(".csv","")}.pt'
    if not best_pt.exists():
        torch.save(dqn.q_estimator.qnet.state_dict(), best_pt)
    print(f'[{label}] Complet. Millor metric: {best_metric:.2f}%')
    return dqn


def run_dqn(save_dir, mans_steps: int, partides_steps: int, device, mode: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'control':
        dqn = _dqn_rlcard_loop(save_dir, mans_steps + partides_steps, device,
                               use_mans=False, log_name='training_log.csv',
                               label='DQN-RLCard-ctrl')
        torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'final.pt')
    else:
        _dqn_rlcard_loop(save_dir, mans_steps, device,
                         use_mans=True, log_name='training_log_mans.csv',
                         label='DQN-RLCard-mans')
        load_path = save_dir / 'best_mans.pt'
        dqn = _dqn_rlcard_loop(save_dir, partides_steps, device,
                               use_mans=False, log_name='training_log_partides.csv',
                               label='DQN-RLCard-partides', load_path=load_path)
        torch.save(dqn.q_estimator.qnet.state_dict(), save_dir / 'final.pt')


# NFSP RLCard
def _nfsp_rlcard_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
                      log_name: str, label: str, load_path: Path | None = None):
    """Loop d'entrenament genèric per NFSP RLCard (mans o partides)."""
    log_path = save_dir / log_name
    init_log(log_path)

    eps_decay = int(min(timesteps, 24_000_000) * 0.8)

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

    p0 = make_nfsp()

    if load_path is not None and load_path.exists():
        checkpoint = torch.load(load_path, map_location=device)
        p0._rl_agent.q_estimator.qnet.load_state_dict(checkpoint['q_net'])
        p0.policy_network.load_state_dict(checkpoint['sl_net'])
        print(f'[NFSP] Pesos carregats des de {load_path}')

    rand_opp    = RandomAgent(num_actions=N_ACTIONS)
    regles_opp  = AgentRegles(num_actions=N_ACTIONS, seed=456)
    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    rng         = np.random.default_rng(SEED)

    EnvClass = TrucEnvMa if use_mans else TrucEnv
    cfg      = ENV_CONFIG_MA if use_mans else ENV_CONFIG
    env      = wrap_env_aplanat(EnvClass(cfg))

    eval_every   = EVAL_EVERY_STEPS
    global_step  = 0
    games_played = 0
    best_metric  = -1.0
    wr_r = wr_g  = 0.0
    t0           = time.time()
    devnull      = io.StringIO()

    pbar = trange(timesteps, desc=label)
    while global_step < timesteps:
        r_opp = rng.random()
        opp = rand_opp if r_opp < PCT_RANDOM else regles_opp
        learner_pid = int(rng.integers(0, 2))

        if learner_pid == 0:
            env.set_agents([p0, opp])
        else:
            env.set_agents([opp, p0])

        p0.sample_episode_policy()

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
                opp_action, _ = opp.eval_step(state)
                next_state, next_pid = env.step(opp_action)

            state = next_state
            player_id = next_pid

        global_step  += n_ts
        games_played += 1
        pbar.update(n_ts)

        if global_step % eval_every < max(n_ts, 1):
            wr_r, wr_g, metric = evaluar_agent(p0, ENV_CONFIG, regles_eval)
            elapsed = time.time() - t0
            append_log(log_path, global_step, games_played, None, wr_r, wr_g, metric, elapsed)
            best_pt = save_dir / f'best_{log_name.replace("training_log_","").replace(".csv","")}.pt'
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
                    'sl_net': p0.policy_network.state_dict(),
                }, best_pt)
                tqdm.write(f'[{label} step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
            else:
                tqdm.write(f'[{label} step {global_step}] random={wr_r:.1f}% regles={wr_g:.1f}%')

        pbar.set_postfix({'WR_rand': f'{wr_r:.1f}%', 'WR_regl': f'{wr_g:.1f}%'})

    pbar.close()
    best_pt = save_dir / f'best_{log_name.replace("training_log_","").replace(".csv","")}.pt'
    if not best_pt.exists():
        torch.save({
            'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
            'sl_net': p0.policy_network.state_dict(),
        }, best_pt)
    print(f'[{label}] Complet. Millor metric: {best_metric:.2f}%')
    return p0


def run_nfsp(save_dir, mans_steps: int, partides_steps: int, device, mode: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'control':
        p0 = _nfsp_rlcard_loop(save_dir, mans_steps + partides_steps, device,
                               use_mans=False, log_name='training_log.csv',
                               label='NFSP-ctrl')
        torch.save({
            'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
            'sl_net': p0.policy_network.state_dict(),
        }, save_dir / 'final.pt')
    else:
        _nfsp_rlcard_loop(save_dir, mans_steps, device,
                         use_mans=True, log_name='training_log_mans.csv',
                         label='NFSP-mans')
        load_path = save_dir / 'best_mans.pt'
        p0 = _nfsp_rlcard_loop(save_dir, partides_steps, device,
                               use_mans=False, log_name='training_log_partides.csv',
                               label='NFSP-partides', load_path=load_path)
        torch.save({
            'q_net': p0._rl_agent.q_estimator.qnet.state_dict(),
            'sl_net': p0.policy_network.state_dict(),
        }, save_dir / 'final.pt')


# DQN SB3
def _dqn_sb3_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
                  log_name: str, label: str, load_path: Path | None = None):
    """Loop d'entrenament genèric per DQN SB3 (mans o partides)."""
    log_path = save_dir / log_name
    init_log(log_path)

    env_fn  = _make_gym_env_ma_fn if use_mans else _make_gym_env_fn
    env     = env_fn('regles', learner_pid=0, seed=SEED)()

    policy_kwargs = dict(net_arch=HIDDEN_LAYERS)

    if load_path is not None and load_path.exists():
        model = SB3DQN.load(str(load_path), env=env, device=device)
        model.learning_starts    = SB3DQN_WARMUP_FT
        model.exploration_rate   = SB3DQN_EPS_FINETUNE
        print(f'[DQN-SB3] Pesos carregats des de {load_path}')
    else:
        model = SB3DQN(
            'MlpPolicy', env,
            learning_rate=SB3DQN_LR,
            batch_size=SB3DQN_BATCH,
            buffer_size=SB3DQN_MEMORY,
            learning_starts=SB3DQN_WARMUP,
            train_freq=SB3DQN_TRAIN_FREQ,
            target_update_interval=SB3DQN_TARGET_UPDATE,
            exploration_fraction=SB3DQN_EPS_FRACTION,
            exploration_initial_eps=SB3DQN_EPS_START,
            exploration_final_eps=SB3DQN_EPS_END,
            gamma=SB3DQN_GAMMA,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_pt     = save_dir / f'best_{log_name.replace("training_log_","").replace(".csv","")}'
    t0          = time.time()

    class _EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= EVAL_EVERY_STEPS:
                self._last_eval = self.num_timesteps
                eval_agent = SB3EvalAgent(self.model)
                wr_r, wr_g, metric = evaluar_agent(eval_agent, ENV_CONFIG, regles_eval)
                elapsed = time.time() - t0
                append_log(log_path, self.num_timesteps, 0, None, wr_r, wr_g, metric, elapsed)
                if metric > best_metric[0]:
                    best_metric[0] = metric
                    self.model.save(str(best_pt))
                    print(f'[{label} step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
                else:
                    print(f'[{label} step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}%')
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCallback())
    env.close()
    model.save(str(save_dir / f'final_{log_name.replace("training_log_","").replace(".csv","")}'))
    if not (Path(str(best_pt) + '.zip')).exists():
        model.save(str(best_pt))
    print(f'[{label}] Complet. Millor metric: {best_metric[0]:.2f}%')
    return model


def run_dqn_sb3(save_dir, mans_steps: int, partides_steps: int, device, mode: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'control':
        _dqn_sb3_loop(save_dir, mans_steps + partides_steps, device,
                      use_mans=False, log_name='training_log.csv', label='DQN-SB3-ctrl')
    else:
        _dqn_sb3_loop(save_dir, mans_steps, device,
                      use_mans=True, log_name='training_log_mans.csv', label='DQN-SB3-mans')
        load_path = save_dir / 'best_mans.zip'
        _dqn_sb3_loop(save_dir, partides_steps, device,
                      use_mans=False, log_name='training_log_partides.csv',
                      label='DQN-SB3-partides',
                      load_path=load_path if load_path.exists() else None)


# PPO SB3
def _ppo_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
              log_name: str, label: str, num_envs: int = NUM_ENVS_PPO,
              load_path: Path | None = None):
    """Loop d'entrenament genèric per PPO SB3 (mans o partides)."""
    log_path = save_dir / log_name
    init_log(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)
    n_random   = max(1, int(num_envs * 0.05))  # ~5% random

    env_fn = _make_gym_env_ma_fn if use_mans else _make_gym_env_fn
    env_fns = [
        env_fn('random' if i < n_random else 'regles', i % 2, SEED + i)
        for i in range(num_envs)
    ]
    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

    if load_path is not None and load_path.exists():
        model = PPO.load(str(load_path), env=vec_env, device=device)
        print(f'[PPO] Pesos carregats des de {load_path}')
    else:
        model = PPO(
            'MlpPolicy', vec_env,
            learning_rate=PPO_LR,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=PPO_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE,
            clip_range=PPO_CLIP,
            ent_coef=PPO_ENT,
            vf_coef=PPO_VF,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_zip    = save_dir / f'best_{log_name.replace("training_log_","").replace(".csv","")}'
    t0          = time.time()

    class _EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= eval_every:
                self._last_eval = self.num_timesteps
                eval_agent = SB3EvalAgent(self.model)
                wr_r, wr_g, metric = evaluar_agent(eval_agent, ENV_CONFIG, regles_eval)
                elapsed = time.time() - t0
                append_log(log_path, self.num_timesteps, 0, None, wr_r, wr_g, metric, elapsed)
                if metric > best_metric[0]:
                    best_metric[0] = metric
                    self.model.save(str(best_zip))
                    print(f'[{label} step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!')
                else:
                    print(f'[{label} step {self.num_timesteps}] '
                          f'random={wr_r:.1f}% regles={wr_g:.1f}%')
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCallback())
    vec_env.close()
    model.save(str(save_dir / f'final_{log_name.replace("training_log_","").replace(".csv","")}'))
    if not (Path(str(best_zip) + '.zip')).exists():
        model.save(str(best_zip))
    print(f'[{label}] Complet. Millor metric: {best_metric[0]:.2f}%')
    return model


def run_ppo(save_dir, mans_steps: int, partides_steps: int, device, mode: str,
            num_envs: int = NUM_ENVS_PPO):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'control':
        _ppo_loop(save_dir, mans_steps + partides_steps, device,
                  use_mans=False, log_name='training_log.csv',
                  label='PPO-ctrl', num_envs=num_envs)
    else:
        _ppo_loop(save_dir, mans_steps, device,
                  use_mans=True, log_name='training_log_mans.csv',
                  label='PPO-mans', num_envs=num_envs)
        load_path = save_dir / 'best_mans.zip'
        _ppo_loop(save_dir, partides_steps, device,
                  use_mans=False, log_name='training_log_partides.csv',
                  label='PPO-partides', num_envs=num_envs,
                  load_path=load_path if load_path.exists() else None)




def main():
    parser = argparse.ArgumentParser(description='Fase 2: Curriculum Learning')
    parser.add_argument('--agent',
                        choices=['dqn', 'nfsp', 'dqn_sb3', 'ppo'],
                        required=True)
    parser.add_argument('--mode',
                        choices=['control', 'curriculum'],
                        default='control',
                        help='control=24M partides directes | curriculum=12M mans + 12M partides')
    parser.add_argument('--mans_steps',     type=int, default=12_000_000)
    parser.add_argument('--partides_steps', type=int, default=12_000_000)
    parser.add_argument('--num_envs',       type=int, default=None,
                        help='Nombre d\'entorns paral·lels (PPO, defecte 48)')
    parser.add_argument('--save_dir',       type=str, default=None)
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
        save_dir = str(Path(__file__).parent / 'registres' / f'{args.agent}_{args.mode}_{ts}')

    total = args.mans_steps + args.partides_steps
    print(f'[{args.agent.upper()}] mode={args.mode} | '
          f'mans={args.mans_steps:,} partides={args.partides_steps:,} '
          f'(total={total:,}) | save_dir={save_dir}')

    num_envs = args.num_envs or NUM_ENVS_PPO
    t_start  = time.time()

    if args.agent == 'dqn':
        run_dqn(save_dir, args.mans_steps, args.partides_steps, device, args.mode)
    elif args.agent == 'nfsp':
        run_nfsp(save_dir, args.mans_steps, args.partides_steps, device, args.mode)
    elif args.agent == 'dqn_sb3':
        run_dqn_sb3(save_dir, args.mans_steps, args.partides_steps, device, args.mode)
    elif args.agent == 'ppo':
        run_ppo(save_dir, args.mans_steps, args.partides_steps, device, args.mode, num_envs)

    total_time = time.time() - t_start
    print(f'\nTemps total: {total_time:.0f}s ({total_time/3600:.2f}h)')


if __name__ == '__main__':
    main()
