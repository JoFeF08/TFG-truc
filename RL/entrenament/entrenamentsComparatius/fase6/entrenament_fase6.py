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
from torch.optim import Adam
from tqdm import tqdm

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
from RL.entrenament.entrenamentsComparatius.fase5.entrenament_fase5 import (
    _carregar_model_inicial, evaluar_per_variant, evaluar_selfplay, _jugar_partida_sencera
)

from RL.models.nfsp.average_policy import AveragePolicyNet, SLAgent
from RL.models.nfsp.reservoir_buffer import ReservoirBuffer
from RL.entrenament.entrenamentsComparatius.fase6.pool_nfsp import NFSPPool
from joc.entorn_ma.gym_env_sessio import TrucGymEnvSessio
from RL.tools.calibracio import mesurar_calibracio, CALIB_ENVIT_MIN


# Reutilitzats de F5
SNAPSHOT_EVERY     = 1_000_000
MAX_SNAPSHOTS      = 9
LAMBDA_STD         = 0.5
N_RECENT_EVAL      = 3
EVAL_EVERY         = max(EVAL_EVERY_STEPS, NUM_ENVS_PPO * 10)
N_SESSIONS_EVAL    = 20
N_PARTIDES_SESSIO_DEFAULT = 5

# Nous F6
RESERVOIR_CAP      = 200_000     
SL_TRAIN_EVERY     = 50_000      
SL_N_MINIBATCHES   = 256
SL_BATCH_SIZE      = 256
SL_LR              = 5e-4
ETA_DEFAULT        = 0.5
N_SESSIONS_SL_EVAL = 60          # 30 partides per posició (resolució ~1,7 pp)
NASH_MIN_STEPS     = 10_000_000  # llindar mínim per arxivar best_nash (evita soroll inicial)
ENVIT_BOT_MIN      = 50.0        # floor de seguretat WR vs envit_bot (gate real = calib_envit)
NASH_PATIENCE      = 3           # evals vàlids sense millora de best_nash -> early stop
CALIB_EVAL_INITS   = 3000        # inicialitzacions per mesurar la calibració (1a jugada)

F5_MODEL_DEFAULT = str(
    Path(__file__).parents[4]
    / "TFG_Doc/notebooks/5_selfplay/resultats/ppo_selfplay_pool_9snaps/best_robust.zip"
)

def _make_sessio_env_fn(nfsp_pool: NFSPPool, n_partides: int, learner_pid: int, seed: int):
    def _init():
        cfg = ENV_CONFIG_MA.copy()
        cfg['seed'] = seed
        return TrucGymEnvSessio(
            cfg,
            opponent_pool_fn=nfsp_pool.sample,
            n_partides=n_partides,
            learner_pid=learner_pid,
            seed=seed,
        )
    return _init

def entrenar_sl_un_cicle(net, optim, buf, device, n_batches=256, batch_size=256) -> float:
    net.train()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(n_batches):
        obs, act = buf.sample(batch_size)
        obs, act = obs.to(device), act.to(device)
        loss = loss_fn(net(obs), act)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    net.eval()
    return float(np.mean(losses))

def evaluar_vs_sl(eval_agent, sl_agent, n_sessions=N_SESSIONS_SL_EVAL, seed=24680) -> float:
    """WR del PPO actual contra SLAgent (estocàstic). Nash ideal: 50%.

    Alterna la posició de l'eval_agent (meitat mà, meitat post) per eliminar el biaix
    posicional. El Truc no admet empats en partida sencera, així que perdre com a post
    equival a (not eval_agent guanya com a primer agent)."""
    rng = random.Random(seed)
    wins = 0
    for i in range(n_sessions):
        if i % 2 == 0:
            wins += int(_jugar_partida_sencera(eval_agent, sl_agent, rng))
        else:
            wins += int(not _jugar_partida_sencera(sl_agent, eval_agent, rng))
    return 100.0 * wins / n_sessions

def init_log_f6(path: Path) -> None:
    header = (
        "step,"
        "wr_random,wr_regles,metric,"
        "wr_conservador,wr_agressiu,wr_truc_bot,wr_envit_bot,wr_faroler,wr_equilibrat,"
        "wr_pool_mean,std_pool,metric_robust,"
        "wr_vs_self,exploit_selfplay,"
        "sl_loss,wr_vs_sl,exploit_vs_sl,eta_actual,"
        "n_snapshots,elapsed,"
        "calib_envit,calib_truc,nash_valid,evals_sense_millora\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)

def append_log_f6(path: Path, step: int,
                  wr_random: float, wr_regles: float, metric: float,
                  wr_variants: dict[str, float],
                  metric_robust: float, std_pool: float,
                  wr_vs_self: float, exploit_selfplay: float,
                  sl_loss: float, wr_vs_sl: float, exploit_vs_sl: float, eta_actual: float,
                  n_snapshots: int, elapsed: float,
                  calib_envit: float = float('nan'), calib_truc: float = float('nan'),
                  nash_valid: bool = False, evals_sense_millora: int = 0) -> None:
    wr_pool_mean = sum(wr_variants.values()) / len(wr_variants)
    
    def _f(v): return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "nan"
    
    cols = [
        str(step),
        _f(wr_random), _f(wr_regles), _f(metric),
        _f(wr_variants.get('conservador', 0)),
        _f(wr_variants.get('agressiu', 0)),
        _f(wr_variants.get('truc_bot', 0)),
        _f(wr_variants.get('envit_bot', 0)),
        _f(wr_variants.get('faroler', 0)),
        _f(wr_variants.get('equilibrat', 0)),
        _f(wr_pool_mean),
        _f(std_pool),
        _f(metric_robust),
        _f(wr_vs_self),
        _f(exploit_selfplay),
        _f(sl_loss),
        _f(wr_vs_sl),
        _f(exploit_vs_sl),
        _f(eta_actual),
        str(n_snapshots),
        f"{elapsed:.2f}",
        _f(calib_envit),
        _f(calib_truc),
        str(int(nash_valid)),
        str(int(evals_sense_millora)),
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

def _ppo_nfsp(save_dir: Path, timesteps: int, device,
              pesos_cos: str, model_inicial: str,
              num_envs: int, n_partides: int, eta_target: float,
              eta_rampup: int, reservoir_cap: int, sl_lr: float, sl_every: int,
              nash_min_steps: int = NASH_MIN_STEPS,
              sl_eval_sessions: int = N_SESSIONS_SL_EVAL,
              nash_patience: int = NASH_PATIENCE,
              ent_coef_override: float | None = None,
              hidden_layers: list[int] | None = None) -> PPO:

    log_path       = save_dir / "training_log.csv"
    snapshot_dir   = save_dir / "snapshots"
    sl_ckpt_dir    = save_dir / "sl_checkpoints"
    sl_ckpt_dir.mkdir(parents=True, exist_ok=True)

    init_log_f6(log_path)

    # INIT SL components
    reservoir = ReservoirBuffer(capacity=reservoir_cap, seed=SEED)

    # Model CPU per al pool (compartit via COW als subprocessos fork)
    sl_net_cpu = AveragePolicyNet(pesos_cos, hidden=hidden_layers)
    sl_agent = SLAgent(sl_net_cpu, 'cpu', deterministic=False, seed=SEED)

    nfsp_pool = NFSPPool(snapshot_dir, sl_agent, eta=0.0 if eta_rampup > 0 else eta_target, max_snapshots=MAX_SNAPSHOTS)

    n_steps    = min(PPO_N_STEPS * NUM_ENVS_PPO // num_envs, 2048)
    batch_size = min(PPO_MINIBATCH, num_envs * n_steps)

    env_fns = [
        _make_sessio_env_fn(nfsp_pool, n_partides, i % 2, SEED + i)
        for i in range(num_envs)
    ]
    # Evita deadlocks de thread pool heretats pel fork
    torch.set_num_threads(1)
    # start_method='fork': subprocessos hereten sl_net_cpu via COW sense re-importar
    vec_env = SB3SubprocVecEnv(env_fns, start_method='fork')

    # Model CUDA per entrenament (DESPRÉS del fork per evitar corrupció de context CUDA)
    sl_net = AveragePolicyNet(pesos_cos, hidden=hidden_layers).to(device)
    sl_opt = Adam(sl_net.head.parameters(), lr=sl_lr)

    # Model per a avaluació
    sl_agent_eval = SLAgent(sl_net, device, deterministic=False, seed=SEED+1)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] {torch.cuda.get_device_name(0)} (SL training)")

    # Arquitectura configurable
    if hidden_layers is None:
        hidden_layers = [256, 256]
    
    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=hidden_layers, vf=hidden_layers),
        activation_fn=nn.ReLU,
    )

    # Detecta si s'ha passat --model_inicial "" per forçar entrenament de zero
    use_model_inicial = model_inicial and model_inicial.strip() and Path(model_inicial).exists()
    
    if use_model_inicial:
        model = _carregar_model_inicial(model_inicial, vec_env, policy_kwargs, torch.device("cpu"))
        _aplicar_frozen(model, pesos_cos, lr=PPO_LR)
    else:
        if model_inicial and model_inicial.strip():
            print(f"[F6] Model inicial no trobat ({model_inicial}), partint de zero.")
        else:
            print(f"[F6] No s'ha especificat model inicial (--model_inicial ''), partint de zero.")
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=PPO_LR, n_steps=n_steps, batch_size=batch_size,
            n_epochs=PPO_EPOCHS, gamma=PPO_GAMMA, gae_lambda=PPO_GAE,
            clip_range=PPO_CLIP, ent_coef=PPO_ENT, vf_coef=PPO_VF,
            policy_kwargs=policy_kwargs, verbose=0, device=torch.device("cpu"),
        )
        _aplicar_frozen(model, pesos_cos, lr=PPO_LR)

    if ent_coef_override is not None:
        model.ent_coef = ent_coef_override
        print(f"[F6] ent_coef override = {ent_coef_override}")

    regles_eval     = AgentRegles(num_actions=N_ACTIONS, seed=789)
    best_metric_val = [-1.0]
    best_robust     = [-1.0]
    best_exploit_sl = [100.0]
    best_calib_combined = [-1.0]  # Mètrica combinada: (calib_envit + calib_truc) / 2
    best_calib_envit = [-1.0]     # Per logging
    best_calib_truc  = [-1.0]     # Per logging

    best_zip        = save_dir / "best"
    best_robust_zip = save_dir / "best_robust"
    best_nash_zip   = save_dir / "best_nash"
    best_calib_zip  = save_dir / "best_calib"
    
    t0              = time.time()
    steps_fets      = [0]
    label           = "F6-NFSP"

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_eval = 0
            self._last_snap = 0
            self._last_sl_train = 0
            self._last_sl_loss = float('nan')
            self._last_sl_calib = 0
            self._last_sl_calib_envit = float('nan')
            self._peak_sl_calib_envit = 0.0
            self._sl_paused = False
            self._last_progress = 0
            self._pbar = None
            self._evals_sense_millora = 0
            self._best_calib = float('-inf')

        def _on_training_start(self) -> None:
            self._pbar = tqdm(total=timesteps, unit="step", dynamic_ncols=True)

        def _on_training_end(self) -> None:
            if self._pbar:
                self._pbar.close()

        def _on_step(self) -> bool:
            t = self.num_timesteps
            if self._pbar is not None:
                self._pbar.n = t
                sl_str = f"{self._last_sl_loss:.4f}" if not np.isnan(self._last_sl_loss) else "nan"
                self._pbar.set_postfix({"res": len(reservoir), "sl": sl_str, "eta": f"{nfsp_pool.eta:.2f}"}, refresh=True)

            if t - self._last_progress >= 100_000:
                self._last_progress = t
                sl_str = f"{self._last_sl_loss:.4f}" if not np.isnan(self._last_sl_loss) else "nan"
                print(f"[F6 {t:>9,}] reservoir={len(reservoir):,}  eta={nfsp_pool.eta:.2f}  sl_loss={sl_str}", flush=True)

            obs_batch = self.locals.get('new_obs')
            if obs_batch is None:
                obs_tensor = self.locals.get('obs_tensor')
                if obs_tensor is not None:
                    obs_batch = obs_tensor.cpu().numpy()
            action_batch = self.locals['actions']
            
            if isinstance(obs_batch, dict):
                from RL.tools.obs_utils import flatten_obs_batch
                try:
                    obs_flat = flatten_obs_batch(obs_batch)
                    reservoir.add_batch(obs_flat, action_batch)
                except ImportError:
                    # Alternativa simple:
                    from RL.tools.obs_utils import flatten_obs
                    for i in range(len(action_batch)):
                        o = {k: v[i] for k, v in obs_batch.items()}
                        reservoir.add(flatten_obs(o), action_batch[i])
            else:
                reservoir.add_batch(obs_batch, action_batch)

            if eta_rampup > 0:
                nfsp_pool.set_eta(min(eta_target, eta_target * t / eta_rampup))
            
            # Força eta=0 si el buffer és molt petit per evitar aprendre escombraries
            if len(reservoir) < SL_BATCH_SIZE * 10:
                eta_efectiva = 0.0
            else:
                eta_efectiva = nfsp_pool.eta

            if t - self._last_snap >= SNAPSHOT_EVERY:
                self._last_snap = t
                nfsp_pool.add_snapshot(self.model, t)
                
            # SL
            if t - self._last_sl_train >= sl_every and len(reservoir) >= SL_BATCH_SIZE:
                self._last_sl_loss = entrenar_sl_un_cicle(sl_net, sl_opt, reservoir, device,
                                                          n_batches=SL_N_MINIBATCHES, batch_size=SL_BATCH_SIZE)
                self._last_sl_train = t
                # Sincronitza pesos cap al model CPU del pool
                sl_net_cpu.load_state_dict(
                    {k: v.detach().cpu() for k, v in sl_net.state_dict().items()}
                )
                if (t // 1_000_000) > (self._last_sl_train // 1_000_000) or t % 1_000_000 < sl_every:
                    torch.save(sl_net.state_dict(), sl_ckpt_dir / f"sl_{t//1_000_000}M.pt")

            # Monitorització de calibració dins SL (detecta col·lapses mid-training)
            if t - self._last_sl_calib >= 1_000_000 and len(reservoir) >= SL_BATCH_SIZE * 20:
                self._last_sl_calib = t
                agent_sl = SLAgent(sl_net, device, deterministic=False, seed=SEED+2)
                calib_sl = mesurar_calibracio(agent_sl, ENV_CONFIG, n_inits=1000)
                calib_envit_sl = calib_sl["calib_envit"]
                
                if not np.isnan(calib_envit_sl):
                    delta = calib_envit_sl - self._last_sl_calib_envit if not np.isnan(self._last_sl_calib_envit) else 0
                    delta_str = f"({delta:+.1f}pp)" if delta != 0 else ""
                    
                    # Update peak
                    if calib_envit_sl > self._peak_sl_calib_envit:
                        self._peak_sl_calib_envit = calib_envit_sl
                    
                    # Dinàmic SL pause/resume basad en calib_envit
                    should_pause = calib_envit_sl < (self._peak_sl_calib_envit - 20.0) and calib_envit_sl < 25.0
                    should_resume = calib_envit_sl > (self._peak_sl_calib_envit - 10.0) and self._sl_paused
                    
                    if should_pause and not self._sl_paused:
                        nfsp_pool.set_eta(0.0)
                        self._sl_paused = True
                        print(f"[SL-PAUSE {t:,}] calib_envit {self._peak_sl_calib_envit:.1f}pp -> {calib_envit_sl:.1f}pp | SL disabled", flush=True)
                    elif should_resume and self._sl_paused:
                        nfsp_pool.set_eta(eta_target)
                        self._sl_paused = False
                        print(f"[SL-RESUME {t:,}] calib_envit={calib_envit_sl:.1f}pp | SL re-enabled", flush=True)
                    
                    status = "[PAUSED]" if self._sl_paused else ""
                    print(f"[SL {t:>9,}] calib_envit={calib_envit_sl:.1f}pp {delta_str} {status}", flush=True)
                    
                    if calib_envit_sl < CALIB_ENVIT_MIN:
                        print(f"[SL-WARN {t:,}] calib_envit < {CALIB_ENVIT_MIN}pp", flush=True)
                    self._last_sl_calib_envit = calib_envit_sl

            # Avaluació
            if t - self._last_eval >= EVAL_EVERY:
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

                recents      = nfsp_pool.get_recent(N_RECENT_EVAL)
                wr_vs_self   = evaluar_selfplay(agent, recents)
                exploit_sp   = abs(wr_vs_self - 50.0) if not np.isnan(wr_vs_self) else float('nan')
                
                # Eval SL
                wr_vs_sl     = evaluar_vs_sl(agent, sl_agent_eval, sl_eval_sessions)
                exploit_vs_sl = abs(wr_vs_sl - 50.0)

                # Calibració estratègica (detecta el col·lapse d'envit)
                calib = mesurar_calibracio(agent, ENV_CONFIG, n_inits=CALIB_EVAL_INITS)
                calib_envit = calib["calib_envit"]
                calib_truc  = calib["calib_truc"]

                # Gates per a best_nash (madur + reservoir ple + no degenerat)
                reservoir_ple = len(reservoir) >= reservoir.capacity
                prou_madur    = t >= nash_min_steps
                no_degenerat  = ((not np.isnan(calib_envit)) and calib_envit >= CALIB_ENVIT_MIN
                                 and wr_variants.get('envit_bot', 0) >= ENVIT_BOT_MIN)
                nash_valid    = reservoir_ple and prou_madur and no_degenerat

                nou_millor = []
                if metric > best_metric_val[0]:
                    best_metric_val[0] = metric
                    self.model.save(str(best_zip))
                    nou_millor.append("M")
                if metric_robust > best_robust[0]:
                    best_robust[0] = metric_robust
                    self.model.save(str(best_robust_zip))
                    nou_millor.append("MR")

                # best_calib: millor calibració encontrada (independent de nash_valid)
                calib_combined = (calib_envit + calib_truc) / 2
                if calib_combined > self._best_calib:
                    self._best_calib = calib_combined
                    self.model.save(str(best_calib_zip))
                    nou_millor.append("C")

                # best_nash: només entre evals vàlids (no degenerats); early stopping equilibrat
                # Exigeix millora SIMULTÀNIA tant de exploit_vs_sl com de calib_combined
                if nash_valid:
                    millora_exploit = exploit_vs_sl < best_exploit_sl[0]
                    millora_calib = calib_combined > best_calib_combined[0]
                    
                    if millora_exploit and millora_calib:
                        best_exploit_sl[0] = exploit_vs_sl
                        best_calib_combined[0] = calib_combined
                        best_calib_envit[0] = calib_envit
                        best_calib_truc[0] = calib_truc
                        self.model.save(str(best_nash_zip))
                        nou_millor.append("N")
                        self._evals_sense_millora = 0
                    else:
                        self._evals_sense_millora += 1

                elapsed = time.time() - t0
                append_log_f6(log_path, t,
                              wr_random, wr_regles, metric,
                              wr_variants, metric_robust, std_pool,
                              wr_vs_self, exploit_sp,
                              self._last_sl_loss, wr_vs_sl, exploit_vs_sl, eta_efectiva,
                              nfsp_pool.n_snapshots, elapsed,
                              calib_envit, calib_truc, nash_valid, self._evals_sense_millora)

                flags = " | ".join(nou_millor) if nou_millor else ""

                exploit_str = f"{exploit_sp:.2f}" if not np.isnan(exploit_sp) else "nan"
                calib_str = f"{calib_envit:.0f}" if not np.isnan(calib_envit) else "nan"
                print(
                    f"[{label} {t:>9,}] "
                    f"m={metric:.1f}% mr={metric_robust:.1f}% "
                    f"std={std_pool:.1f} exp_sp={exploit_str} exp_sl={exploit_vs_sl:.1f} "
                    f"cal_env={calib_str}pp v={int(nash_valid)} ns={self._evals_sense_millora} "
                    f"eta={eta_efectiva:.2f} {flags}"
                )

                if nash_valid and self._evals_sense_millora >= nash_patience:
                    print(f"[{label}] EARLY STOP @ {t:,}: best_nash estable "
                          f"({nash_patience} evals vàlids sense millora dual), "
                          f"exp_sl_min={best_exploit_sl[0]:.2f}%, "
                          f"calib_combined_max={(best_calib_envit[0] + best_calib_truc[0])/2:.1f}pp "
                          f"(envit={best_calib_envit[0]:.1f}pp, truc={best_calib_truc[0]:.1f}pp)")
                    return False
            return True

    model.learn(total_timesteps=timesteps, callback=_Cb())
    vec_env.close()
    model.save(str(save_dir / "final"))
    torch.save(sl_net.state_dict(), str(save_dir / "sl_final.pt"))
    
    if not Path(str(best_zip) + ".zip").exists():
        model.save(str(best_zip))
    if not Path(str(best_robust_zip) + ".zip").exists():
        model.save(str(best_robust_zip))
    if not Path(str(best_nash_zip) + ".zip").exists():
        model.save(str(best_nash_zip))
    if not Path(str(best_calib_zip) + ".zip").exists():
        model.save(str(best_calib_zip))
        
    print(f"[{label}] Complet. Millor metric: {best_metric_val[0]:.2f}%  "
          f"Millor metric_robust: {best_robust[0]:.2f}%  "
          f"Millor Nash exploit: {best_exploit_sl[0]:.2f}% | calib_combined: {best_calib_combined[0]:.1f}pp "
          f"(envit={best_calib_envit[0]:.1f}pp, truc={best_calib_truc[0]:.1f}pp)")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fase 6: NFSP Complet")
    parser.add_argument("--pesos_cos",     type=str, required=True,
                        help="Ruta al best_pesos_cos_truc.pth")
    parser.add_argument("--model_inicial", type=str, default=F5_MODEL_DEFAULT,
                        help="Ruta al best_robust.zip de F5 (punt de partida)")
    parser.add_argument("--steps",         type=int, default=12_000_000)
    parser.add_argument("--num_envs",      type=int, default=NUM_ENVS_PPO)
    parser.add_argument("--n_partides",    type=int, default=N_PARTIDES_SESSIO_DEFAULT)
    parser.add_argument("--save_dir",      type=str, default=None)
    parser.add_argument("--eta",           type=float, default=ETA_DEFAULT)
    parser.add_argument("--eta_rampup",    type=int, default=0)
    parser.add_argument("--reservoir_cap", type=int, default=RESERVOIR_CAP)
    parser.add_argument("--sl_lr",         type=float, default=SL_LR)
    parser.add_argument("--sl_every",      type=int, default=SL_TRAIN_EVERY)
    parser.add_argument("--nash_min_steps", type=int, default=NASH_MIN_STEPS,
                        help="No arxivar best_nash abans d'aquest pas (evita soroll inicial)")
    parser.add_argument("--sl_eval_sessions", type=int, default=N_SESSIONS_SL_EVAL,
                        help="Partides per avaluar exploit_vs_sl (alterna posició)")
    parser.add_argument("--nash_patience", type=int, default=NASH_PATIENCE,
                        help="Evals vàlids sense millora de best_nash abans d'aturar (early stopping)")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[256, 256],
                        help="Arquitectura policy head: [h1 h2 ...]. Default [256 256]")
    parser.add_argument("--ent_coef", type=float, default=None,
                        help="Override del coeficient d'entropia del PPO (protegeix accions rares com apostar_envit)")
    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[CPU] GPU no disponible, SL en CPU.")

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        ts = datetime.now().strftime("%d%m_%H%Mh")
        save_dir = Path(__file__).parent / "registres" / f"ppo_nfsp_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[F6] steps={args.steps:,}  num_envs={args.num_envs}  "
          f"eta={args.eta} rampup={args.eta_rampup}  hidden={args.hidden_layers}  save_dir={save_dir}")

    t_start = time.time()
    _ppo_nfsp(
        save_dir, args.steps, device,
        pesos_cos=args.pesos_cos,
        model_inicial=args.model_inicial,
        num_envs=args.num_envs,
        n_partides=args.n_partides,
        eta_target=args.eta,
        eta_rampup=args.eta_rampup,
        reservoir_cap=args.reservoir_cap,
        sl_lr=args.sl_lr,
        sl_every=args.sl_every,
        nash_min_steps=args.nash_min_steps,
        sl_eval_sessions=args.sl_eval_sessions,
        nash_patience=args.nash_patience,
        ent_coef_override=args.ent_coef,
        hidden_layers=args.hidden_layers,
    )
    total = time.time() - t_start
    print(f"\nTemps total: {total:.0f}s ({total/3600:.2f}h)")


if __name__ == "__main__":
    main()
