"""
Script d'entrenament Fase 5 — Self-Play Mixt + Metriques d'Explotabilitat
Parteix de F4-ablacio (PPO + COS frozen) i entrena contra un pool mixt:
6 variants d'AgentRegles (sempre presents) + snapshots del PPO actual
(finestra rodant de MAX_SNAPSHOTS). Cada SNAPSHOT_EVERY steps es desa
un snapshot i s'incorpora al pool.

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
from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
from RL.models.model_propi.agent_regles import AgentRegles

from RL.entrenament.entrenamentsComparatius.fase2.entrenament_fase2_curriculum import (
    SEED, N_ACTIONS, HIDDEN_LAYERS, ENV_CONFIG_MA, ENV_CONFIG,
    EVAL_EVERY_STEPS, NUM_ENVS_PPO,
    PPO_LR, PPO_GAMMA, PPO_GAE, PPO_CLIP, PPO_ENT, PPO_VF,
    PPO_EPOCHS, PPO_MINIBATCH, PPO_N_STEPS,
    SB3EvalAgent, evaluar_agent, EVAL_GAMES_RANDOM, EVAL_GAMES_REGLES,
)
from RL.entrenament.entrenamentsComparatius.fase4.entrenament_fase4 import _aplicar_frozen
from RL.entrenament.entrenamentsComparatius.fase4.pool_oponents import (
    POOL_OPONENTS, NOMS_VARIANTS, crear_oponent,
)
from RL.entrenament.entrenamentsComparatius.fase5.pool_selfplay import SelfPlayPool

from joc.entorn.env import TrucEnv
from joc.entorn_ma.gym_env_sessio import TrucGymEnvSessio


SNAPSHOT_EVERY  = 1_000_000
MAX_SNAPSHOTS   = 6
LAMBDA_STD      = 0.5
N_RECENT_EVAL   = 3
N_SESSIONS_EVAL = 20
N_PARTIDES_SESSIO_DEFAULT = 5

F4_MODEL_DEFAULT = str(
    Path(__file__).parents[4]
    / "TFG_Doc/notebooks/4_memoria/resultats/ppo_ablacio_pool/best.zip"
)


def _carregar_model_inicial(ruta: str, vec_env, policy_kwargs: dict, device) -> PPO:
    """Carrega F4-ablacio com a punt de partida. Monkey-patch optimizer per COS frozen."""
    _orig = PPO.set_parameters

    def _sense_optimizer(self, load_path_or_dict, exact_match=True, device="auto"):  # exact_match ignorat intencionalment
        if isinstance(load_path_or_dict, dict):
            load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                 if "optimizer" not in k}
        return _orig(self, load_path_or_dict, exact_match=False, device=device)

    PPO.set_parameters = _sense_optimizer
    try:
        model = PPO.load(
            ruta,
            env=vec_env,
            custom_objects={
                "features_extractor_class": CosMultiInputSB3,
                "policy_kwargs": policy_kwargs,
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_MINIBATCH,
                "learning_rate": PPO_LR,
            },
            device=device,
        )
    finally:
        PPO.set_parameters = _orig
    return model


def _make_sessio_env_fn(selfplay_pool: SelfPlayPool, n_partides: int,
                        learner_pid: int, seed: int):
    def _init():
        cfg = ENV_CONFIG_MA.copy()
        cfg['seed'] = seed
        return TrucGymEnvSessio(
            cfg,
            opponent_pool_fn=selfplay_pool.sample,
            n_partides=n_partides,
            learner_pid=learner_pid,
            seed=seed,
        )
    return _init


def _jugar_partida_sencera(eval_agent, oponent, rng: random.Random) -> bool:
    """Juga 1 partida sencera (TrucEnv). Retorna True si eval_agent guanya."""
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
    return env.game.get_payoffs()[0] > 0


def evaluar_per_variant(eval_agent, n_sessions: int = N_SESSIONS_EVAL,
                        seed: int = 54321) -> dict[str, float]:
    """WR per cada variant del pool (partides senceres). Clau per metric_robust."""
    rng = random.Random(seed)
    resultats = {}
    for nom in NOMS_VARIANTS:
        wins = 0
        for _ in range(n_sessions):
            oponent = crear_oponent(nom, seed=rng.randint(0, 2**31 - 1))
            wins += int(_jugar_partida_sencera(eval_agent, oponent, rng))
        resultats[nom] = 100.0 * wins / n_sessions
    return resultats


def evaluar_selfplay(eval_agent, recents: list, n_sessions: int = 10,
                     seed: int = 99999) -> float:
    """WR del model actual contra snapshots recents. Ideal a Nash: ~0.5 (50%)."""
    if not recents:
        return float('nan')
    rng = random.Random(seed)
    wrs = []
    for _nom, snap_agent in recents:
        wins = sum(int(_jugar_partida_sencera(eval_agent, snap_agent, rng))
                   for _ in range(n_sessions))
        wrs.append(100.0 * wins / n_sessions)
    return sum(wrs) / len(wrs)


def init_log(path: Path) -> None:
    header = (
        "step,"
        "wr_random,wr_regles,metric,"
        "wr_conservador,wr_agressiu,wr_truc_bot,wr_envit_bot,wr_faroler,wr_equilibrat,"
        "wr_pool_mean,std_pool,metric_robust,"
        "wr_vs_self,exploit_selfplay,"
        "n_snapshots,elapsed\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)


def append_log(path: Path, step: int,
               wr_random: float, wr_regles: float, metric: float,
               wr_variants: dict[str, float],
               metric_robust: float, std_pool: float,
               wr_vs_self: float, exploit_selfplay: float,
               n_snapshots: int, elapsed: float) -> None:
    wr_pool_mean = sum(wr_variants.values()) / len(wr_variants)
    cols = [
        str(step),
        f"{wr_random:.4f}", f"{wr_regles:.4f}", f"{metric:.4f}",
        f"{wr_variants.get('conservador', 0):.4f}",
        f"{wr_variants.get('agressiu', 0):.4f}",
        f"{wr_variants.get('truc_bot', 0):.4f}",
        f"{wr_variants.get('envit_bot', 0):.4f}",
        f"{wr_variants.get('faroler', 0):.4f}",
        f"{wr_variants.get('equilibrat', 0):.4f}",
        f"{wr_pool_mean:.4f}",
        f"{std_pool:.4f}",
        f"{metric_robust:.4f}",
        f"{wr_vs_self:.4f}" if not (isinstance(wr_vs_self, float) and np.isnan(wr_vs_self)) else "nan",
        f"{exploit_selfplay:.4f}" if not (isinstance(exploit_selfplay, float) and np.isnan(exploit_selfplay)) else "nan",
        str(n_snapshots),
        f"{elapsed:.2f}",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")


def _ppo_selfplay(save_dir: Path, timesteps: int, device,
                  pesos_cos: str, model_inicial: str,
                  num_envs: int, n_partides: int) -> PPO:

    log_path       = save_dir / "training_log.csv"
    snapshot_dir   = save_dir / "snapshots"
    selfplay_pool  = SelfPlayPool(snapshot_dir, max_snapshots=MAX_SNAPSHOTS)

    init_log(log_path)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    eval_every = max(EVAL_EVERY_STEPS, num_envs * 10)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)

    env_fns = [
        _make_sessio_env_fn(selfplay_pool, n_partides, i % 2, SEED + i)
        for i in range(num_envs)
    ]
    vec_env = SB3SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
        activation_fn=nn.ReLU,
    )

    if Path(model_inicial).exists():
        print(f"[F5] Carregant model inicial: {model_inicial}")
        model = _carregar_model_inicial(model_inicial, vec_env, policy_kwargs, device)
        _aplicar_frozen(model, pesos_cos, lr=PPO_LR)
    else:
        print(f"[F5] Model inicial no trobat ({model_inicial}), partint de zero.")
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
    best_robust     = [-1.0]
    best_zip        = save_dir / "best"         # criteri: metric classica (com totes les fases)
    best_robust_zip = save_dir / "best_robust"  # criteri: metric_robust (nova)
    t0              = time.time()
    steps_fets      = [0]
    label           = "F5-SELFPLAY"

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0
            self._last_snap = 0

        def _on_step(self) -> bool:
            t = self.num_timesteps

            if t - self._last_snap >= SNAPSHOT_EVERY:
                self._last_snap = t
                selfplay_pool.add_snapshot(self.model, t)

            if t - self._last_eval >= eval_every:
                self._last_eval = t
                steps_fets[0] = t
                agent = SB3EvalAgent(self.model)

                wr_random, wr_regles, metric = evaluar_agent(
                    agent, ENV_CONFIG, regles_eval,
                    n_random=EVAL_GAMES_RANDOM, n_regles=EVAL_GAMES_REGLES,
                )

                wr_variants = evaluar_per_variant(agent)
                vals = list(wr_variants.values())
                wr_pool_mean = sum(vals) / len(vals)
                std_pool     = float(np.std(vals))
                metric_robust = wr_pool_mean - LAMBDA_STD * std_pool

                recents      = selfplay_pool.get_recent(N_RECENT_EVAL)
                wr_vs_self   = evaluar_selfplay(agent, recents)
                exploit_sp   = abs(wr_vs_self - 50.0) if not np.isnan(wr_vs_self) else float('nan')

                elapsed = time.time() - t0
                append_log(log_path, t,
                           wr_random, wr_regles, metric,
                           wr_variants, metric_robust, std_pool,
                           wr_vs_self, exploit_sp,
                           selfplay_pool.n_snapshots, elapsed)

                if metric > best_metric_val[0]:
                    best_metric_val[0] = metric
                    self.model.save(str(best_zip))
                    nou_millor = " -> nou millor!"
                else:
                    nou_millor = ""

                if metric_robust > best_robust[0]:
                    best_robust[0] = metric_robust
                    self.model.save(str(best_robust_zip))

                sp_str = f"{wr_vs_self:.1f}%" if not np.isnan(wr_vs_self) else "nan"
                print(
                    f"[{label} {t:>9,}] "
                    f"metric={metric:.1f}% robust={metric_robust:.1f}% "
                    f"std={std_pool:.1f} exploit_sp={exploit_sp if not np.isnan(exploit_sp) else 'nan':.2f} "
                    f"snaps={selfplay_pool.n_snapshots}{nou_millor}"
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
          f"Millor metric_robust: {best_robust[0]:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fase 5: Self-Play Mixt")
    parser.add_argument("--pesos_cos",     type=str, required=True,
                        help="Ruta al best_pesos_cos_truc.pth")
    parser.add_argument("--model_inicial", type=str, default=F4_MODEL_DEFAULT,
                        help="Ruta al best.zip de F4-ablacio (punt de partida)")
    parser.add_argument("--steps",         type=int, default=12_000_000)
    parser.add_argument("--num_envs",      type=int, default=NUM_ENVS_PPO)
    parser.add_argument("--n_partides",    type=int, default=N_PARTIDES_SESSIO_DEFAULT)
    parser.add_argument("--save_dir",      type=str, default=None)
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
        save_dir = Path(__file__).parent / "registres" / f"ppo_selfplay_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[F5] steps={args.steps:,}  num_envs={args.num_envs}  "
          f"n_partides={args.n_partides}  save_dir={save_dir}")

    t_start = time.time()
    _ppo_selfplay(
        save_dir, args.steps, device,
        pesos_cos=args.pesos_cos,
        model_inicial=args.model_inicial,
        num_envs=args.num_envs,
        n_partides=args.n_partides,
    )
    total = time.time() - t_start
    print(f"\nTemps total: {total:.0f}s ({total/3600:.2f}h)")


if __name__ == "__main__":
    main()
