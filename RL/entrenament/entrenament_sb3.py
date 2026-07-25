"""Entrenament base amb Stable-Baselines3 (MaskablePPO) sobre l'entorn de Truc.

Self-play d'una mà (TrucGymEnvMa): l'oponent juga dins de step() i l'agent
nomes veu els seus propis torns. Fa servir CosMultiInput com a extractor de
features i, opcionalment, en carrega pesos preentrenats (preentrenar_cos.py).
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

if '__file__' in globals():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from joc.entorn_ma.gym_env_ma import TrucGymEnvMa
from RL.models.model_propi.agent_regles import AgentRegles
from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3

N_ACTIONS = 19


def mask_fn(env: TrucGymEnvMa) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[env._legal_actions] = True
    return mask


def crear_entorn(env_config: dict, seed: int):
    def _init():
        opponent = AgentRegles(seed=seed)
        env = TrucGymEnvMa(env_config, opponent=opponent)
        return ActionMasker(env, mask_fn)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenament base MaskablePPO (SB3) per al Truc")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_jugadors", type=int, default=2)
    parser.add_argument("--cartes_jugador", type=int, default=3)
    parser.add_argument("--senyes", action="store_true")
    parser.add_argument("--pesos_cos", type=str, default=None,
                         help="Ruta a un .pth de CosMultiInput preentrenat (preentrenar_cos.py)")
    parser.add_argument("--congelar_cos", action="store_true")
    args = parser.parse_args()

    env_config = {
        "num_jugadors": args.num_jugadors,
        "cartes_jugador": args.cartes_jugador,
        "senyes": args.senyes,
    }

    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_env_cls([crear_entorn(env_config, seed=i) for i in range(args.num_envs)])

    run_dir = Path(__file__).resolve().parent / "registres_sb3" / datetime.now().strftime("%d_%m_%y_a_les_%H%M")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )

    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    if args.pesos_cos:
        model.policy.features_extractor.carregar_pesos_preentrenats(args.pesos_cos)
        if args.congelar_cos:
            model.policy.features_extractor.congelar_cos()

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=str(run_dir / "models"),
        name_prefix="ppo_truc",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
    model.save(str(run_dir / "models" / "final"))


if __name__ == "__main__":
    main()
