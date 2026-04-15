"""
Script d'entrenament Fase 3 — Feature Extractor Preentrenat
------------------------------------------------------------
Integra CosMultiInput (preentrenat amb preentrenar_cos.py) com a features extractor
dels dos models SB3 seleccionats al Checkpoint 1 (DQN-SB3 i PPO-SB3).

Tres variants per a cada algorisme:
  scratch  → CosMultiInput amb pesos aleatoris (SB3 init), tot entrenable.
  frozen   → CosMultiInput amb pesos preentrenats, cos congelat (requires_grad=False).
  finetune → CosMultiInput amb pesos preentrenats, tot entrenable.
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
    SEED,
    N_ACTIONS,
    HIDDEN_LAYERS,
    ENV_CONFIG,
    EVAL_EVERY_STEPS,
    SB3DQN_LR, SB3DQN_BATCH, SB3DQN_MEMORY, SB3DQN_WARMUP,
    SB3DQN_TRAIN_FREQ, SB3DQN_TARGET_UPDATE,
    SB3DQN_EPS_START, SB3DQN_EPS_END, SB3DQN_EPS_FRACTION, SB3DQN_GAMMA,
    NUM_ENVS_PPO, PPO_LR, PPO_GAMMA, PPO_GAE, PPO_CLIP, PPO_ENT, PPO_VF,
    PPO_EPOCHS, PPO_MINIBATCH, PPO_N_STEPS,
    init_log, append_log, evaluar_agent, SB3EvalAgent,
    _make_gym_env_ma_fn,
)


def _aplicar_variant(model, variant: str, pesos_cos: str | None, lr: float):
    """Carrega pesos i/o congela el cos segons la variant escollida."""
    # Per a DQN l'extractor és a q_net i q_net_target, per a PPO està a model.policy
    if isinstance(model, SB3DQN):
        extractors = [model.policy.q_net.features_extractor, model.policy.q_net_target.features_extractor]
    else:
        # PPO i altres
        extractors = [getattr(model.policy, "features_extractor", None)]

    if variant in ("frozen", "finetune"):
        if pesos_cos is None:
            raise ValueError(f"variant={variant!r} requereix --pesos_cos")
        if not Path(pesos_cos).exists():
            raise FileNotFoundError(f"No existeix: {pesos_cos}")
        
        for extractor in extractors:
            if extractor is not None:
                extractor.carregar_pesos_preentrenats(pesos_cos)
                # Moure al mateix device que la policy.
                extractor.to(model.policy.device)
        
        print(f"[fase3] Pesos preentrenats carregats: {pesos_cos}")

    if variant == "frozen":
        for extractor in extractors:
            if extractor is not None:
                extractor.congelar_cos()
        
        entrenables = [p for p in model.policy.parameters() if p.requires_grad]
        model.policy.optimizer = torch.optim.Adam(entrenables, lr=lr)
        n_tot = sum(p.numel() for p in model.policy.parameters())
        n_ent = sum(p.numel() for p in entrenables)
        print(f"[fase3] Cos congelat. Params entrenables: {n_ent:,}/{n_tot:,}")


def _dqn_sb3_fase3(save_dir: Path, timesteps: int, device, variant: str,
                   pesos_cos: str | None):
    log_path = save_dir / "training_log.csv"
    init_log(log_path)

    env = _make_gym_env_ma_fn("regles", learner_pid=0, seed=SEED)()

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=HIDDEN_LAYERS,
    )

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

    _aplicar_variant(model, variant, pesos_cos, lr=SB3DQN_LR)

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_pt     = save_dir / "best"
    t0          = time.time()
    label       = f"DQN-SB3-{variant}"

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
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}%")
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCallback())
    env.close()
    model.save(str(save_dir / "final"))
    if not (Path(str(best_pt) + ".zip")).exists():
        model.save(str(best_pt))
    print(f"[{label}] Complet. Millor metric: {best_metric[0]:.2f}%")
    return model


def _ppo_sb3_fase3(save_dir: Path, timesteps: int, device, variant: str,
                   pesos_cos: str | None, num_envs: int):
    log_path = save_dir / "training_log.csv"
    init_log(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)
    n_random   = max(1, int(num_envs * 0.05))  # ~5% random

    env_fns = [
        _make_gym_env_ma_fn("random" if i < n_random else "regles", i % 2, SEED + i)
        for i in range(num_envs)
    ]
    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

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

    _aplicar_variant(model, variant, pesos_cos, lr=PPO_LR)

    regles_eval = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric = [-1.0]
    best_zip    = save_dir / "best"
    t0          = time.time()
    label       = f"PPO-SB3-{variant}"

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
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}% → nou millor!")
                else:
                    print(f"[{label} step {self.num_timesteps}] "
                          f"random={wr_r:.1f}% regles={wr_g:.1f}%")
            return True

    model.learn(total_timesteps=timesteps, callback=_EvalCallback())
    vec_env.close()
    model.save(str(save_dir / "final"))
    if not (Path(str(best_zip) + ".zip")).exists():
        model.save(str(best_zip))
    print(f"[{label}] Complet. Millor metric: {best_metric[0]:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fase 3: Feature Extractor Preentrenat")
    parser.add_argument("--agent",    choices=["dqn_sb3", "ppo_sb3"], required=True)
    parser.add_argument("--variant",  choices=["scratch", "frozen", "finetune"], required=True)
    parser.add_argument("--pesos_cos", type=str, default=None,
                        help="Ruta al best_pesos_cos_truc.pth (obligatori per frozen/finetune)")
    parser.add_argument("--steps",    type=int, default=24_000_000)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS_PPO,
                        help="Nombre d'entorns paral·lels per PPO (defecte 48)")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.variant in ("frozen", "finetune") and args.pesos_cos is None:
        parser.error(f"--variant {args.variant} requereix --pesos_cos")

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
        save_dir = Path(__file__).parent / "registres" / f"{args.agent}_{args.variant}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.agent.upper()}-{args.variant}] steps={args.steps:,} save_dir={save_dir}")

    t_start = time.time()
    if args.agent == "dqn_sb3":
        _dqn_sb3_fase3(save_dir, args.steps, device, args.variant, args.pesos_cos)
    else:
        _ppo_sb3_fase3(save_dir, args.steps, device, args.variant, args.pesos_cos,
                       num_envs=args.num_envs)

    total_time = time.time() - t_start
    print(f"\nTemps total: {total_time:.0f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    main()
