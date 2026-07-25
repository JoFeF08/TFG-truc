"""Entrenament amb Stable-Baselines3 (MaskablePPO) sobre l'entorn de Truc,
per a les etapes 1-3 del currículum (oponent fix/script; l'etapa 4,
self-play amb lliga, viu a entrenament_selfplay.py).

Self-play d'una mà (TrucGymEnvMa): l'oponent juga dins de step() i l'agent
nomes veu els seus propis torns. Fa servir CosMultiInput com a extractor de
features i, opcionalment, en carrega pesos preentrenats (preentrenar_cos.py).

--stage controla quines accions estan desbloquejades:
  cartes -> nomes jugar cartes (sense truc ni envit)
  envit  -> cartes + envit (sense truc)
  truc   -> joc complet (per defecte)
--opponent controla l'oponent mostrejat per l'OpponentPool a cada mà:
  random -> accio aleatoria
  regles -> AgentRegles amb parametres variats
  pool   -> tambe inclou checkpoints congelats de --pool_dir (lliga)
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

STAGE_FLAGS = {
    "cartes": dict(permetre_apostes=False, permetre_truc=False),
    "envit": dict(permetre_apostes=True, permetre_truc=False),
    "truc": dict(permetre_apostes=True, permetre_truc=True),
}

OPPONENT_PESOS = {
    "random": {"random": 1.0, "regles": 0.0, "pool": 0.0},
    "regles": {"random": 0.1, "regles": 0.9, "pool": 0.0},
    "pool": {"random": 0.1, "regles": 0.3, "pool": 0.6},
}


def mask_fn(env: TrucGymEnvMa) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[env._legal_actions] = True
    return mask


def crear_entorn(env_config: dict, opponent_kwargs: dict, seed: int):
    def _init():
        pool = OpponentPool(**opponent_kwargs, seed=seed)
        env = TrucGymEnvMa(env_config, opponent_pool=pool)
        return ActionMasker(env, mask_fn)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenament amb currículum (MaskablePPO/SB3) per al Truc")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_jugadors", type=int, default=2)
    parser.add_argument("--cartes_jugador", type=int, default=3)
    parser.add_argument("--senyes", action="store_true")
    parser.add_argument("--stage", choices=list(STAGE_FLAGS), default="truc")
    parser.add_argument("--opponent", choices=list(OPPONENT_PESOS), default="regles")
    parser.add_argument("--pool_dir", type=str, default=None,
                         help="Directori de checkpoints .zip per a --opponent pool")
    parser.add_argument("--multi_head", action="store_true",
                         help="Fa servir la política amb caps separats (cartes/truc/envit)")
    parser.add_argument("--pesos_cos", type=str, default=None,
                         help="Ruta a un .pth de CosMultiInput preentrenat (preentrenar_cos.py)")
    parser.add_argument("--congelar_cos", action="store_true")
    args = parser.parse_args()

    env_config = {
        "num_jugadors": args.num_jugadors,
        "cartes_jugador": args.cartes_jugador,
        "senyes": args.senyes,
        **STAGE_FLAGS[args.stage],
    }
    opponent_kwargs = {"pool_dir": args.pool_dir, "pesos": OPPONENT_PESOS[args.opponent]}

    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_env_cls([crear_entorn(env_config, opponent_kwargs, seed=i) for i in range(args.num_envs)])

    run_dir = Path(__file__).resolve().parent / "registres_sb3" / datetime.now().strftime("%d_%m_%y_a_les_%H%M")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    (n_channels, _, _), context_size = obs_shapes(args.num_jugadors, args.senyes)
    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256, in_channels=n_channels, context_size=context_size),
        net_arch=[256, 256],
    )
    policy_cls = MultiHeadMaskableACPolicy if args.multi_head else "MlpPolicy"

    model = MaskablePPO(
        policy_cls, env,
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
