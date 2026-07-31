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
  pool   -> majoritariament checkpoints propis (lliga), amb una mica de regles/random
  mixt   -> meitat AgentRegles (variants), meitat checkpoints propis (self-play)
Als tres darrers, els checkpoints periodics d'aquest mateix run es desen a
--pool_dir, que es el mateix directori d'on l'OpponentPool els torna a
llegir: la banda de self-play es retroalimenta sola durant l'entrenament,
comencant nomes amb AgentRegles/random fins que hi ha el primer checkpoint.
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
    "random": {"random": 1.0, "regles": 0.0, "pool": 0.0, "probabilistic": 0.0},
    "regles": {"random": 0.1, "regles": 0.9, "pool": 0.0, "probabilistic": 0.0},
    "pool": {"random": 0.1, "regles": 0.3, "pool": 0.6, "probabilistic": 0.0},
    "mixt": {"random": 0.0, "regles": 0.5, "pool": 0.5, "probabilistic": 0.0},
    # Incorpora AgentProbabilistic (determinització/minimax, quasi-òptim
    # per al subjoc de cartes). Cost real mesurat amb el seient aleatori
    # (l'oponent a vegades ha de liderar "a cegues", el cas car): a pes=1.0
    # surt a ~256ms/mà. Pes generós perquè el temps no és el factor limitant
    # aquí (es deixa córrer tota la nit).
    "fort": {"random": 0.0, "regles": 0.40, "probabilistic": 0.20, "pool": 0.40},
}


def mask_fn(env: TrucGymEnvMa) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[env._legal_actions] = True
    return mask


def crear_entorn(env_config: dict, opponent_kwargs: dict, seed: int, reward_scale: float = 1.0):
    def _init():
        pool = OpponentPool(**opponent_kwargs, seed=seed)
        env = TrucGymEnvMa(env_config, opponent_pool=pool, reward_scale=reward_scale)
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
                         help="Directori de checkpoints .zip per a --opponent pool/mixt "
                              "(per defecte <run_dir>/pool; s'hi pot preomplir amb checkpoints previs)")
    parser.add_argument("--multi_head", action="store_true",
                         help="Fa servir la política amb caps separats (cartes/truc/envit)")
    parser.add_argument("--pesos_cos", type=str, default=None,
                         help="Ruta a un .pth de CosMultiInput preentrenat (preentrenar_cos.py)")
    parser.add_argument("--congelar_cos", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                         help="Escala la recompensa terminal. Útil a --stage cartes: sense apostes "
                              "el marge sempre és ±1/24≈0.042, un senyal molt feble (proveu 10-20).")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                         help="Coeficient d'entropia de PPO (per defecte SB3=0.0, sense bonus "
                              "d'exploració). Amb reward feble (--stage cartes) proveu 0.01.")
    parser.add_argument("--target_kl", type=float, default=None,
                         help="Si es dona, PPO atura cada època d'actualització quan el KL la "
                              "supera (evita divergència/inestabilitat). Recomanat ~0.03 si "
                              "--reward_scale és alt (kl/clip_fraction sostingudament alts).")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Checkpoint .zip (o directori amb checkpoints *_steps.zip) per "
                              "continuar un entrenament interromput. --total_timesteps es tracta "
                              "com a objectiu ABSOLUT (inclou els passos ja fets).")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "registres_sb3" / datetime.now().strftime("%d_%m_%y_a_les_%H%M")
    pool_dir = Path(args.pool_dir) if args.pool_dir else (run_dir / "pool")
    pool_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
        "num_jugadors": args.num_jugadors,
        "cartes_jugador": args.cartes_jugador,
        "senyes": args.senyes,
        **STAGE_FLAGS[args.stage],
    }
    opponent_kwargs = {"pool_dir": str(pool_dir), "pesos": OPPONENT_PESOS[args.opponent]}

    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_env_cls([
        crear_entorn(env_config, opponent_kwargs, seed=i, reward_scale=args.reward_scale)
        for i in range(args.num_envs)
    ])

    (n_channels, _, _), context_size = obs_shapes(args.num_jugadors, args.senyes)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            checkpoints = sorted(
                resume_path.glob("*_steps.zip"),
                key=lambda p: int(p.stem.rsplit('_', 2)[-2]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"Cap checkpoint '*_steps.zip' a {resume_path}")
            resume_path = checkpoints[-1]
        print(f"Represa des de: {resume_path}")
        model = MaskablePPO.load(str(resume_path), env=env, ent_coef=args.ent_coef, target_kl=args.target_kl)
        print(f"Passos ja fets: {model.num_timesteps:,} / objectiu {args.total_timesteps:,}")
    else:
        policy_kwargs = dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(features_dim=256, in_channels=n_channels, context_size=context_size),
            net_arch=[256, 256],
        )
        policy_cls = MultiHeadMaskableACPolicy if args.multi_head else "MlpPolicy"

        model = MaskablePPO(
            policy_cls, env,
            policy_kwargs=policy_kwargs,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            verbose=1,
        )

        if args.pesos_cos:
            model.policy.features_extractor.carregar_pesos_preentrenats(args.pesos_cos)
            if args.congelar_cos:
                model.policy.features_extractor.congelar_cos()

    # save_path = pool_dir: els checkpoints periodics tambe alimenten
    # l'OpponentPool (banda de self-play de --opponent pool/mixt).
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=str(pool_dir),
        name_prefix="ppo_truc",
    )

    # SB3 tracta `total_timesteps` de .learn() com a passos ADDICIONALS quan
    # reset_num_timesteps=False (internament fa total_timesteps += model.num_timesteps,
    # vegeu BaseAlgorithm._setup_learn) -- NO com a objectiu absolut. Cal
    # restar els passos ja fets perque --total_timesteps segueixi volent dir
    # "objectiu total" tant si es comença de zero com si es reprèn.
    if args.resume_from:
        passos_restants = max(0, args.total_timesteps - model.num_timesteps)
        if passos_restants == 0:
            print(f"Ja s'ha arribat a l'objectiu ({model.num_timesteps:,} >= {args.total_timesteps:,}). Res a fer.")
            return
    else:
        passos_restants = args.total_timesteps

    model.learn(
        total_timesteps=passos_restants,
        callback=checkpoint_cb,
        reset_num_timesteps=(args.resume_from is None),
    )
    model.save(str(run_dir / "final"))


if __name__ == "__main__":
    main()
