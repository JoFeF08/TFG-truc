"""Etapa 4 del currículum: self-play amb lliga (MaskablePPO/SB3).

Reutilitza el mateix TrucGymEnvMa + OpponentPool de les etapes anteriors
(entrenament_sb3.py): el "self-play" és simplement un OpponentPool ponderat
majoritàriament cap a 'pool' (checkpoints congelats d'una versió anterior
del propi aprenent), amb una mica de 'random'/'regles' per mantenir
diversitat i no oblidar com respondre a jugades poc habituals.

El directori de checkpoints (`--pool_dir`, per defecte `<run_dir>/pool`)
fa doble funció: és on `CheckpointCallback` desa els checkpoints periòdics
I d'on `OpponentPool` els torna a llegir — la lliga es retroalimenta sola,
sense cap pas manual entremig.
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

from joc.entorn.obs_builder import obs_shapes
from joc.entorn_ma.gym_env_ma import TrucGymEnvMa
from RL.models.model_propi.opponent_pool import OpponentPool
from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.sb3.multi_head_policy import MultiHeadMaskableACPolicy

N_ACTIONS = 19

# Predomini de checkpoints propis (self-play), amb una mica de diversitat
# perquè l'aprenent no oblidi com respondre a jugades poc "òptimes".
SELFPLAY_PESOS = {"random": 0.1, "regles": 0.2, "pool": 0.7}


def mask_fn(env: TrucGymEnvMa) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[env._legal_actions] = True
    return mask


def crear_entorn(env_config: dict, pool_dir: str, seed: int):
    def _init():
        pool = OpponentPool(pool_dir=pool_dir, pesos=SELFPLAY_PESOS, seed=seed)
        env = TrucGymEnvMa(env_config, opponent_pool=pool)
        return ActionMasker(env, mask_fn)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play amb lliga (MaskablePPO/SB3) per al Truc")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_jugadors", type=int, default=2)
    parser.add_argument("--cartes_jugador", type=int, default=3)
    parser.add_argument("--senyes", action="store_true")
    parser.add_argument("--pool_dir", type=str, default=None,
                         help="Directori de la lliga (per defecte <run_dir>/pool). Es pot preomplir amb checkpoints previs.")
    parser.add_argument("--multi_head", action="store_true")
    parser.add_argument("--init_model", type=str, default=None,
                         help="Checkpoint .zip d'una etapa anterior del currículum per continuar-hi l'entrenament")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000,
                         help="Cada quants timesteps (aprox.) es desa un nou checkpoint a la lliga")
    args = parser.parse_args()

    env_config = {
        "num_jugadors": args.num_jugadors,
        "cartes_jugador": args.cartes_jugador,
        "senyes": args.senyes,
        "permetre_apostes": True,
        "permetre_truc": True,
    }

    run_dir = Path(__file__).resolve().parent / "registres_selfplay" / datetime.now().strftime("%d_%m_%y_a_les_%H%M")
    pool_dir = Path(args.pool_dir) if args.pool_dir else (run_dir / "pool")
    pool_dir.mkdir(parents=True, exist_ok=True)

    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_env_cls([crear_entorn(env_config, str(pool_dir), seed=i) for i in range(args.num_envs)])

    if args.init_model:
        model = MaskablePPO.load(args.init_model, env=env)
    else:
        (n_channels, _, _), context_size = obs_shapes(args.num_jugadors, args.senyes)
        policy_kwargs = dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(features_dim=256, in_channels=n_channels, context_size=context_size),
            net_arch=[256, 256],
        )
        policy_cls = MultiHeadMaskableACPolicy if args.multi_head else "MlpPolicy"
        model = MaskablePPO(policy_cls, env, policy_kwargs=policy_kwargs, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(pool_dir),
        name_prefix="selfplay",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
    model.save(str(run_dir / "final"))


if __name__ == "__main__":
    main()
