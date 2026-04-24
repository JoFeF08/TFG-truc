"""
Script d'entrenament Fase 4 — Memòria d'Oponent (RecurrentPPO + Sessions Multi-Partida)
----------------------------------------------------------------------------------------
Entrena un agent PPO amb COS frozen contra un pool divers d'AgentRegles en sessions
de N partides consecutives. 
"""

import sys
import os
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.insert(0, root_path)
except Exception:
    pass

from rlcard.utils import set_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv as SB3SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError as e:
    RecurrentPPO = None

from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.sb3.sb3_lstm_eval_agent import SB3LSTMEvalAgent
from RL.models.model_propi.agent_regles import AgentRegles

from RL.entrenament.entrenamentsComparatius.fase2.entrenament_fase2_curriculum import (
    SEED,
    N_ACTIONS,
    HIDDEN_LAYERS,
    ENV_CONFIG_MA,
    EVAL_EVERY_STEPS,
    NUM_ENVS_PPO, PPO_LR, PPO_GAMMA, PPO_GAE, PPO_CLIP, PPO_ENT, PPO_VF,
    PPO_EPOCHS, PPO_MINIBATCH, PPO_N_STEPS,
    init_log, append_log, SB3EvalAgent,
    evaluar_agent, ENV_CONFIG, EVAL_GAMES_RANDOM, EVAL_GAMES_REGLES,
)

from joc.entorn.gym_env import TrucGymEnv
from joc.entorn_ma.gym_env_sessio import TrucGymEnvSessio
from RL.entrenament.entrenamentsComparatius.fase4.pool_oponents import (
    sample_oponent, crear_oponent, NOMS_VARIANTS, POOL_OPONENTS,
)


N_PARTIDES_SESSIO_DEFAULT = 5
N_SESSIONS_EVAL = 20  # per cada tipus d'oponent (random + pool)


def _make_sessio_env_fn(n_partides: int, learner_pid: int, seed: int):
    """Factory per a TrucGymEnvSessio (pool divers d'AgentRegles)."""
    def _init():
        cfg = ENV_CONFIG_MA.copy()
        cfg['seed'] = seed
        return TrucGymEnvSessio(
            cfg,
            opponent_pool_fn=sample_oponent,
            n_partides=n_partides,
            learner_pid=learner_pid,
            seed=seed,
        )
    return _init


def _aplicar_frozen(model, pesos_cos: str, lr: float):
    """Carrega pesos preentrenats al COS i el congela. Reconstrueix optimitzador."""
    if not Path(pesos_cos).exists():
        raise FileNotFoundError(f"No existeix: {pesos_cos}")

    extractor = getattr(model.policy, "features_extractor", None)
    if extractor is None:
        raise RuntimeError("model.policy no té features_extractor")

    extractor.carregar_pesos_preentrenats(pesos_cos)
    extractor.to(model.policy.device)
    extractor.congelar_cos()

    entrenables = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = torch.optim.Adam(entrenables, lr=lr)

    n_tot = sum(p.numel() for p in model.policy.parameters())
    n_ent = sum(p.numel() for p in entrenables)
    print(f"[fase4] COS carregat i congelat. Params entrenables: {n_ent:,}/{n_tot:,}")


def init_log_sessions(path: Path):
    """Log amb mètrica estàndard (comparable amb altres fases) + dades de sessions."""
    header = (
        "step,"
        "wr_random,wr_regles,metric,"
        "wr_random_pool,wr_regles_pool,metric_pool,"
        "wr_pos_1,wr_pos_2,wr_pos_3,wr_pos_4,wr_pos_5,"
        "elapsed\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)


def append_log_sessions(path: Path, step: int,
                        wr_r_std: float, wr_g_std: float, metric_std: float,
                        wr_r_pool: float, wr_g_pool: float, metric_pool: float,
                        wr_pos: list[float], elapsed: float):
    cols = [
        str(step),
        f"{wr_r_std:.4f}", f"{wr_g_std:.4f}", f"{metric_std:.4f}",
        f"{wr_r_pool:.4f}", f"{wr_g_pool:.4f}", f"{metric_pool:.4f}",
    ]
    cols += [f"{w:.4f}" for w in wr_pos]
    cols.append(f"{elapsed:.2f}")
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")


def evaluar_sessions(eval_agent, n_partides_sessio: int,
                     n_sessions_random: int, n_sessions_pool: int, seed: int = 12345):
    """
    Juga sessions de N partides i retorna (wr_random, wr_pool, metric, wr_per_posicio[N]).

    Per cada sessió: reset LSTM al principi, N partides consecutives contra el
    mateix oponent, sense reset LSTM entre partides. WR per posició agrega
    resultats segons l'índex de partida dins la sessió.
    """
    from rlcard.agents import RandomAgent
    from joc.entorn.env import TrucEnv
    from RL.tools.obs_utils import flatten_obs
    from joc.entorn.cartes_accions import ACTION_LIST

    rng = random.Random(seed)
    from joc.entorn_ma.env_ma import TrucEnvMa  # no usat directament, mantenim simetria

    # Acumuladors: posicio_wins[i] = victòries a la i-èsima partida de cada sessió
    pos_wins_tot = [0] * n_partides_sessio
    pos_games_tot = [0] * n_partides_sessio
    wins_random = games_random = 0
    wins_pool = games_pool = 0

    def _jugar_sessio(oponent, is_random: bool):
        nonlocal wins_random, games_random, wins_pool, games_pool
        if hasattr(eval_agent, 'reset'):
            eval_agent.reset()
        for idx in range(n_partides_sessio):
            cfg = ENV_CONFIG_MA.copy()  # evaluem amb mateix protocol (mans)
            # Per avaluar partides senceres comparables amb F3.5, usem ENV_CONFIG (partides)
            from RL.entrenament.entrenamentsComparatius.fase2.entrenament_fase2_curriculum import ENV_CONFIG
            cfg = ENV_CONFIG.copy()
            cfg['seed'] = rng.randint(0, 2**31 - 1)
            env = TrucEnv(cfg)
            env.set_agents([eval_agent, oponent])
            state, player_id = env.reset()
            while player_id is not None:
                if player_id == 0:
                    a, _ = eval_agent.eval_step(state)
                else:
                    a, _ = oponent.eval_step(state)
                state, player_id = env.step(a)
            payoffs = env.game.get_payoffs()
            guanyat = payoffs[0] > 0
            pos_wins_tot[idx] += int(guanyat)
            pos_games_tot[idx] += 1
            if is_random:
                games_random += 1
                wins_random += int(guanyat)
            else:
                games_pool += 1
                wins_pool += int(guanyat)

    # Sessions contra RandomAgent
    rand_agent = RandomAgent(num_actions=N_ACTIONS)
    for _ in range(n_sessions_random):
        _jugar_sessio(rand_agent, is_random=True)

    # Sessions contra el pool d'oponents (distribució uniforme)
    n_per_variant = max(1, n_sessions_pool // len(NOMS_VARIANTS))
    for nom in NOMS_VARIANTS:
        for _ in range(n_per_variant):
            opp = crear_oponent(nom, seed=rng.randint(0, 2**31 - 1))
            _jugar_sessio(opp, is_random=False)

    wr_r = 100.0 * wins_random / max(1, games_random)
    wr_g = 100.0 * wins_pool / max(1, games_pool)
    metric = 0.25 * wr_r + 0.75 * wr_g
    wr_pos = [100.0 * pos_wins_tot[i] / max(1, pos_games_tot[i])
              for i in range(n_partides_sessio)]
    return wr_r, wr_g, metric, wr_pos


def _ppo_ablacio(save_dir: Path, timesteps: int, device,
                 pesos_cos: str, num_envs: int, n_partides: int):
    """F4-ablació: PPO estàndard + COS frozen + pool divers + sessions."""
    log_path = save_dir / "training_log.csv"
    init_log_sessions(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)

    env_fns = [_make_sessio_env_fn(n_partides, i % 2, SEED + i) for i in range(num_envs)]
    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=PPO_LR, n_steps=n_steps, batch_size=batch_size,
        n_epochs=PPO_EPOCHS, gamma=PPO_GAMMA, gae_lambda=PPO_GAE,
        clip_range=PPO_CLIP, ent_coef=PPO_ENT, vf_coef=PPO_VF,
        policy_kwargs=policy_kwargs, verbose=0, device=device,
    )

    _aplicar_frozen(model, pesos_cos, lr=PPO_LR)

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_zip    = save_dir / "best"
    t0          = time.time()
    label       = "PPO-ABLACIO"

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last >= eval_every:
                self._last = self.num_timesteps
                agent = SB3EvalAgent(self.model)
                wr_r_std, wr_g_std, metric_std = evaluar_agent(
                    agent, ENV_CONFIG, regles_eval,
                    n_random=EVAL_GAMES_RANDOM, n_regles=EVAL_GAMES_REGLES,
                )
                wr_r_pool, wr_g_pool, metric_pool, wr_pos = evaluar_sessions(
                    agent, n_partides, N_SESSIONS_EVAL, N_SESSIONS_EVAL * 2,
                )
                elapsed = time.time() - t0
                append_log_sessions(log_path, self.num_timesteps,
                                    wr_r_std, wr_g_std, metric_std,
                                    wr_r_pool, wr_g_pool, metric_pool,
                                    wr_pos, elapsed)
                if metric_std > best_metric[0]:
                    best_metric[0] = metric_std
                    self.model.save(str(best_zip))
                    print(f"[{label} step {self.num_timesteps}] "
                          f"std={metric_std:.1f}% (rand={wr_r_std:.1f}% reg={wr_g_std:.1f}%) "
                          f"pool={wr_g_pool:.1f}% pos=[{','.join(f'{w:.0f}' for w in wr_pos)}] → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"std={metric_std:.1f}% (rand={wr_r_std:.1f}% reg={wr_g_std:.1f}%) "
                          f"pool={wr_g_pool:.1f}% pos=[{','.join(f'{w:.0f}' for w in wr_pos)}]")
            return True

    model.learn(total_timesteps=timesteps, callback=_Cb())
    vec_env.close()
    model.save(str(save_dir / "final"))
    if not (Path(str(best_zip) + ".zip")).exists():
        model.save(str(best_zip))
    print(f"[{label}] Complet. Millor metric (std): {best_metric[0]:.2f}%")
    return model


def _ppo_lstm_complet(save_dir: Path, timesteps: int, device,
                      pesos_cos: str, num_envs: int, n_partides: int):
    """F4-complet: RecurrentPPO + COS frozen + pool divers + sessions."""
    if RecurrentPPO is None:
        raise ImportError("sb3-contrib no disponible. Executa: pip install sb3-contrib")

    log_path = save_dir / "training_log.csv"
    init_log_sessions(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    # RecurrentPPO: batch_size en seqüències. 8 seqs × n_steps transicions/minibatch.
    batch_size_seqs = max(1, min(num_envs // 4, 8))

    env_fns = [_make_sessio_env_fn(n_partides, i % 2, SEED + i) for i in range(num_envs)]
    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        enable_critic_lstm=True,
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.ReLU,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy", vec_env,
        learning_rate=PPO_LR, n_steps=n_steps, batch_size=batch_size_seqs,
        n_epochs=PPO_EPOCHS, gamma=PPO_GAMMA, gae_lambda=PPO_GAE,
        clip_range=PPO_CLIP, ent_coef=PPO_ENT, vf_coef=PPO_VF,
        policy_kwargs=policy_kwargs, verbose=0, device=device,
    )

    _aplicar_frozen(model, pesos_cos, lr=PPO_LR)

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_zip    = save_dir / "best"
    t0          = time.time()
    label       = "PPO-LSTM-COMPLET"

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last >= eval_every:
                self._last = self.num_timesteps
                agent = SB3LSTMEvalAgent(self.model, num_actions=N_ACTIONS)
                wr_r_std, wr_g_std, metric_std = evaluar_agent(
                    agent, ENV_CONFIG, regles_eval,
                    n_random=EVAL_GAMES_RANDOM, n_regles=EVAL_GAMES_REGLES,
                )
                wr_r_pool, wr_g_pool, metric_pool, wr_pos = evaluar_sessions(
                    agent, n_partides, N_SESSIONS_EVAL, N_SESSIONS_EVAL * 2,
                )
                elapsed = time.time() - t0
                append_log_sessions(log_path, self.num_timesteps,
                                    wr_r_std, wr_g_std, metric_std,
                                    wr_r_pool, wr_g_pool, metric_pool,
                                    wr_pos, elapsed)
                if metric_std > best_metric[0]:
                    best_metric[0] = metric_std
                    self.model.save(str(best_zip))
                    print(f"[{label} step {self.num_timesteps}] "
                          f"std={metric_std:.1f}% (rand={wr_r_std:.1f}% reg={wr_g_std:.1f}%) "
                          f"pool={wr_g_pool:.1f}% pos=[{','.join(f'{w:.0f}' for w in wr_pos)}] → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"std={metric_std:.1f}% (rand={wr_r_std:.1f}% reg={wr_g_std:.1f}%) "
                          f"pool={wr_g_pool:.1f}% pos=[{','.join(f'{w:.0f}' for w in wr_pos)}]")
            return True

    model.learn(total_timesteps=timesteps, callback=_Cb())
    vec_env.close()
    model.save(str(save_dir / "final"))
    if not (Path(str(best_zip) + ".zip")).exists():
        model.save(str(best_zip))
    print(f"[{label}] Complet. Millor metric (std): {best_metric[0]:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fase 4: Memòria d'Oponent")
    parser.add_argument("--variant", choices=["ablacio", "complet"], required=True)
    parser.add_argument("--pesos_cos", type=str, required=True,
                        help="Ruta al best_pesos_cos_truc.pth")
    parser.add_argument("--n_partides", type=int, default=N_PARTIDES_SESSIO_DEFAULT)
    parser.add_argument("--steps", type=int, default=12_000_000)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS_PPO)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU] GPU no disponible.")

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        ts = datetime.now().strftime("%d%m_%H%Mh")
        save_dir = Path(__file__).parent / "registres" / f"ppo_{args.variant}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PPO-{args.variant.upper()}] steps={args.steps:,} "
          f"n_partides/sessio={args.n_partides} save_dir={save_dir}")

    t_start = time.time()
    if args.variant == "ablacio":
        _ppo_ablacio(save_dir, args.steps, device, args.pesos_cos,
                     num_envs=args.num_envs, n_partides=args.n_partides)
    else:
        _ppo_lstm_complet(save_dir, args.steps, device, args.pesos_cos,
                          num_envs=args.num_envs, n_partides=args.n_partides)
    total_time = time.time() - t_start
    print(f"\nTemps total: {total_time:.0f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    main()
