"""
F4-ablació 48M — baseline extès per comparació amb F5-selfplay.
Idèntic a _ppo_ablacio de entrenament_fase4.py però amb logging complet
de variants (wr_conservador, ..., metric_robust) igual que F5.
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

from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.model_propi.agent_regles import AgentRegles

from RL.entrenament.entrenamentsComparatius.fase2.entrenament_fase2_curriculum import (
    SEED, N_ACTIONS, HIDDEN_LAYERS, ENV_CONFIG_MA, ENV_CONFIG,
    EVAL_EVERY_STEPS, NUM_ENVS_PPO,
    PPO_LR, PPO_GAMMA, PPO_GAE, PPO_CLIP, PPO_ENT, PPO_VF,
    PPO_EPOCHS, PPO_MINIBATCH, PPO_N_STEPS,
    SB3EvalAgent, evaluar_agent, EVAL_GAMES_RANDOM, EVAL_GAMES_REGLES,
)
from RL.entrenament.entrenamentsComparatius.fase4.entrenament_fase4 import (
    _aplicar_frozen, _make_sessio_env_fn, N_PARTIDES_SESSIO_DEFAULT,
)
from RL.entrenament.entrenamentsComparatius.fase4.pool_oponents import (
    NOMS_VARIANTS, crear_oponent,
)
from joc.entorn.env import TrucEnv

LAMBDA_STD      = 0.5
N_SESSIONS_EVAL = 20


def _init_log(path: Path) -> None:
    header = (
        "step,"
        "wr_random,wr_regles,metric,"
        "wr_conservador,wr_agressiu,wr_truc_bot,wr_envit_bot,wr_faroler,wr_equilibrat,"
        "wr_pool_mean,std_pool,metric_robust,"
        "elapsed\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)


def _append_log(path: Path, step: int,
                wr_random: float, wr_regles: float, metric: float,
                wr_variants: dict[str, float], elapsed: float) -> None:
    vals = list(wr_variants.values())
    pool_mean = np.mean(vals)
    std_pool  = float(np.std(vals))
    metric_robust = pool_mean - LAMBDA_STD * std_pool
    cols = [
        str(step),
        f"{wr_random:.4f}", f"{wr_regles:.4f}", f"{metric:.4f}",
        f"{wr_variants.get('conservador', 0):.4f}",
        f"{wr_variants.get('agressiu',    0):.4f}",
        f"{wr_variants.get('truc_bot',    0):.4f}",
        f"{wr_variants.get('envit_bot',   0):.4f}",
        f"{wr_variants.get('faroler',     0):.4f}",
        f"{wr_variants.get('equilibrat',  0):.4f}",
        f"{pool_mean:.4f}", f"{std_pool:.4f}", f"{metric_robust:.4f}",
        f"{elapsed:.2f}",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")


def _jugar_partida(eval_agent, oponent, rng: random.Random) -> bool:
    cfg = ENV_CONFIG.copy()
    cfg['seed'] = rng.randint(0, 2**31 - 1)
    env = TrucEnv(cfg)
    env.set_agents([eval_agent, oponent])
    state, pid = env.reset()
    while pid is not None:
        a, _ = (eval_agent if pid == 0 else oponent).eval_step(state)
        state, pid = env.step(a)
    return env.game.get_payoffs()[0] > 0


def _evaluar_variants(eval_agent, seed: int = 54321) -> dict[str, float]:
    rng = random.Random(seed)
    resultats = {}
    for nom in NOMS_VARIANTS:
        wins = sum(
            int(_jugar_partida(eval_agent, crear_oponent(nom, seed=rng.randint(0, 2**31-1)), rng))
            for _ in range(N_SESSIONS_EVAL)
        )
        resultats[nom] = 100.0 * wins / N_SESSIONS_EVAL
    return resultats


def _ppo_ablacio_48M(save_dir: Path, timesteps: int, device,
                     pesos_cos: str, num_envs: int, n_partides: int) -> PPO:

    log_path   = save_dir / "training_log.csv"
    _init_log(log_path)

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

    regles_eval     = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric_val = [-1.0]
    best_robust_val = [-1.0]
    best_zip        = save_dir / "best"
    best_robust_zip = save_dir / "best_robust"
    t0              = time.time()
    label           = "F4-ABLACIO-48M"

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last = 0

        def _on_step(self) -> bool:
            t = self.num_timesteps
            if t - self._last >= eval_every:
                self._last = t
                agent = SB3EvalAgent(self.model)

                wr_random, wr_regles, metric = evaluar_agent(
                    agent, ENV_CONFIG, regles_eval,
                    n_random=EVAL_GAMES_RANDOM, n_regles=EVAL_GAMES_REGLES,
                )
                wr_variants = _evaluar_variants(agent)
                vals         = list(wr_variants.values())
                pool_mean    = np.mean(vals)
                std_pool     = float(np.std(vals))
                metric_robust = pool_mean - LAMBDA_STD * std_pool

                elapsed = time.time() - t0
                _append_log(log_path, t, wr_random, wr_regles, metric, wr_variants, elapsed)

                nou_millor = ""
                if metric > best_metric_val[0]:
                    best_metric_val[0] = metric
                    self.model.save(str(best_zip))
                    nou_millor = " → nou millor!"
                if metric_robust > best_robust_val[0]:
                    best_robust_val[0] = metric_robust
                    self.model.save(str(best_robust_zip))

                print(
                    f"[{label} {t:>9,}] "
                    f"metric={metric:.1f}% robust={metric_robust:.1f}% "
                    f"std={std_pool:.1f}%{nou_millor}"
                )
            return True

    model.learn(total_timesteps=timesteps, callback=_Cb())
    vec_env.close()
    model.save(str(save_dir / "final"))
    if not Path(str(best_zip) + ".zip").exists():
        model.save(str(best_zip))
    if not Path(str(best_robust_zip) + ".zip").exists():
        model.save(str(best_robust_zip))
    print(f"[{label}] Complet. Millor metric: {best_metric_val[0]:.2f}%  "
          f"Millor metric_robust: {best_robust_val[0]:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="F4-ablació 48M (baseline per F5)")
    parser.add_argument("--pesos_cos",  type=str, required=True)
    parser.add_argument("--steps",      type=int, default=48_000_000)
    parser.add_argument("--num_envs",   type=int, default=NUM_ENVS_PPO)
    parser.add_argument("--n_partides", type=int, default=N_PARTIDES_SESSIO_DEFAULT)
    parser.add_argument("--save_dir",   type=str, default=None)
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
        save_dir = Path(__file__).parent / "registres" / f"ppo_ablacio_48M_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[F4-ABLACIO-48M] steps={args.steps:,}  num_envs={args.num_envs}  "
          f"n_partides={args.n_partides}  save_dir={save_dir}")

    t_start = time.time()
    _ppo_ablacio_48M(save_dir, args.steps, device,
                     pesos_cos=args.pesos_cos,
                     num_envs=args.num_envs,
                     n_partides=args.n_partides)
    total = time.time() - t_start
    print(f"\nTemps total: {total:.0f}s ({total/3600:.2f}h)")


if __name__ == "__main__":
    main()
