"""
Script d'entrenament Fase 3 — Valor del Feature Extractor COS
-------------------------------------------------------------
Compara DQN-SB3 i PPO-SB3 amb i sense el feature extractor CosMultiInput,
en tres protocols:
  control    → 24M steps directament sobre partides (TrucGymEnv)
  curriculum → 12M steps sobre mans (TrucGymEnvMa) + 12M finetune sobre partides
  mans       → 24M steps sobre mans (TrucGymEnvMa)
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, root_path)
except Exception:
    pass

from rlcard.utils import set_seed
from stable_baselines3 import PPO, DQN as SB3DQN
from stable_baselines3.common.vec_env import SubprocVecEnv as SB3SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.model_propi.agent_regles import AgentRegles

from RL.entrenament.entrenamentsComparatius.fase2.entrenament_fase2_curriculum import (
    SEED, N_ACTIONS, HIDDEN_LAYERS, ENV_CONFIG, EVAL_EVERY_STEPS,
    SB3DQN_LR, SB3DQN_BATCH, SB3DQN_MEMORY, SB3DQN_WARMUP,
    SB3DQN_TRAIN_FREQ, SB3DQN_TARGET_UPDATE,
    SB3DQN_EPS_START, SB3DQN_EPS_END, SB3DQN_EPS_FRACTION, SB3DQN_GAMMA,
    SB3DQN_EPS_FINETUNE, SB3DQN_WARMUP_FT,
    NUM_ENVS_PPO, PPO_LR, PPO_GAMMA, PPO_GAE, PPO_CLIP, PPO_ENT, PPO_VF,
    PPO_EPOCHS, PPO_MINIBATCH, PPO_N_STEPS,
    init_log, append_log, evaluar_agent, SB3EvalAgent,
    _make_gym_env_fn, _make_gym_env_ma_fn,
)


def _policy_kwargs_dqn(use_cos: bool) -> dict:
    if use_cos:
        return dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=HIDDEN_LAYERS,
        )
    return dict(net_arch=HIDDEN_LAYERS)


def _policy_kwargs_ppo(use_cos: bool) -> dict:
    if use_cos:
        return dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
            activation_fn=nn.ReLU,
        )
    return dict(
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )


def _dqn_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
              policy_kwargs: dict, log_name: str, label: str,
              load_path: Path | None = None, use_cos: bool = False) -> None:
    log_path = save_dir / log_name
    init_log(log_path)

    env_fn = _make_gym_env_ma_fn if use_mans else _make_gym_env_fn
    env = env_fn("regles", learner_pid=0, seed=SEED)()

    if load_path is not None and load_path.exists():
        co = {"policy_kwargs": policy_kwargs} if use_cos else None
        model = SB3DQN.load(
            str(load_path), env=env, device=device,
            **({"custom_objects": co} if co else {}),
        )
        model.learning_starts = SB3DQN_WARMUP_FT
        model.exploration_rate = SB3DQN_EPS_FINETUNE
        print(f"[{label}] Model carregat des de {load_path}")
    else:
        model = SB3DQN(
            "MlpPolicy", env,
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
    stem = log_name.replace("training_log_", "").replace("training_log", "").replace(".csv", "")
    best_pt = save_dir / (f"best_{stem}" if stem else "best")
    t0 = time.time()

    class _EvalCB(BaseCallback):
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
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}%")
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCB())
    env.close()
    model.save(str(save_dir / (f"final_{stem}" if stem else "final")))
    if not (Path(str(best_pt) + ".zip")).exists():
        model.save(str(best_pt))
    print(f"[{label}] Complet. Millor metric: {best_metric[0]:.2f}%")


def _ppo_loop(save_dir: Path, timesteps: int, device, use_mans: bool,
              policy_kwargs: dict, log_name: str, label: str,
              num_envs: int = NUM_ENVS_PPO,
              load_path: Path | None = None, use_cos: bool = False) -> None:
    log_path = save_dir / log_name
    init_log(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)
    n_random   = max(1, int(num_envs * 0.05))

    env_fn = _make_gym_env_ma_fn if use_mans else _make_gym_env_fn
    env_fns = [
        env_fn("random" if i < n_random else "regles", i % 2, SEED + i)
        for i in range(num_envs)
    ]
    vec_env = SB3SubprocVecEnv(env_fns)

    if load_path is not None and load_path.exists():
        co = {"policy_kwargs": policy_kwargs} if use_cos else None
        model = PPO.load(
            str(load_path), env=vec_env, device=device,
            **({"custom_objects": co} if co else {}),
        )
        print(f"[{label}] Model carregat des de {load_path}")
    else:
        model = PPO(
            "MlpPolicy", vec_env,
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
    stem = log_name.replace("training_log_", "").replace("training_log", "").replace(".csv", "")
    best_zip = save_dir / (f"best_{stem}" if stem else "best")
    t0 = time.time()

    class _EvalCB(BaseCallback):
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
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}%")
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCB())
    vec_env.close()
    model.save(str(save_dir / (f"final_{stem}" if stem else "final")))
    if not (Path(str(best_zip) + ".zip")).exists():
        model.save(str(best_zip))
    print(f"[{label}] Complet. Millor metric: {best_metric[0]:.2f}%")


def _run_dqn(save_dir: Path, timesteps: int, device, protocol: str, use_cos: bool) -> None:
    pk = _policy_kwargs_dqn(use_cos)
    tag = "cos" if use_cos else "mlp"

    if protocol == "curriculum":
        half = timesteps // 2
        _dqn_loop(save_dir, half, device, use_mans=True, policy_kwargs=pk,
                  log_name="training_log_mans.csv", label=f"DQN-mans-{tag}", use_cos=use_cos)
        load_path = save_dir / "best_mans.zip"
        _dqn_loop(save_dir, half, device, use_mans=False, policy_kwargs=pk,
                  log_name="training_log_partides.csv", label=f"DQN-partides-{tag}",
                  load_path=load_path if load_path.exists() else None, use_cos=use_cos)
    elif protocol == "control":
        _dqn_loop(save_dir, timesteps, device, use_mans=False, policy_kwargs=pk,
                  log_name="training_log.csv", label=f"DQN-ctrl-{tag}", use_cos=use_cos)
    else:  # mans
        _dqn_loop(save_dir, timesteps, device, use_mans=True, policy_kwargs=pk,
                  log_name="training_log.csv", label=f"DQN-mans-{tag}", use_cos=use_cos)


def _run_ppo(save_dir: Path, timesteps: int, device, protocol: str,
             use_cos: bool, num_envs: int) -> None:
    pk = _policy_kwargs_ppo(use_cos)
    tag = "cos" if use_cos else "mlp"

    if protocol == "curriculum":
        half = timesteps // 2
        _ppo_loop(save_dir, half, device, use_mans=True, policy_kwargs=pk,
                  log_name="training_log_mans.csv", label=f"PPO-mans-{tag}",
                  num_envs=num_envs, use_cos=use_cos)
        load_path = save_dir / "best_mans.zip"
        _ppo_loop(save_dir, half, device, use_mans=False, policy_kwargs=pk,
                  log_name="training_log_partides.csv", label=f"PPO-partides-{tag}",
                  num_envs=num_envs,
                  load_path=load_path if load_path.exists() else None, use_cos=use_cos)
    elif protocol == "control":
        _ppo_loop(save_dir, timesteps, device, use_mans=False, policy_kwargs=pk,
                  log_name="training_log.csv", label=f"PPO-ctrl-{tag}",
                  num_envs=num_envs, use_cos=use_cos)
    else:  # mans
        _ppo_loop(save_dir, timesteps, device, use_mans=True, policy_kwargs=pk,
                  log_name="training_log.csv", label=f"PPO-mans-{tag}",
                  num_envs=num_envs, use_cos=use_cos)


def main():
    parser = argparse.ArgumentParser(description="Fase 3: Valor del Feature Extractor COS")
    parser.add_argument("--agent",    choices=["dqn_sb3", "ppo_sb3"], required=True)
    parser.add_argument("--protocol", choices=["control", "curriculum", "mans"], required=True)
    parser.add_argument("--cos",      action="store_true", default=False,
                        help="Usa CosMultiInputSB3 (scratch). Ometre per MLP estàndard.")
    parser.add_argument("--steps",    type=int, default=24_000_000)
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

    tag = "cos" if args.cos else "mlp"
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        ts = datetime.now().strftime("%d%m_%H%Mh")
        save_dir = Path(__file__).parent / "registres" / f"{args.agent}_{args.protocol}_{tag}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fase3] agent={args.agent} protocol={args.protocol} cos={args.cos} "
          f"steps={args.steps:,} save_dir={save_dir}")

    t_start = time.time()
    if args.agent == "dqn_sb3":
        _run_dqn(save_dir, args.steps, device, args.protocol, args.cos)
    else:
        _run_ppo(save_dir, args.steps, device, args.protocol, args.cos, args.num_envs)

    total = time.time() - t_start
    print(f"\nTemps total: {total:.0f}s ({total/3600:.2f}h)")


if __name__ == "__main__":
    main()
