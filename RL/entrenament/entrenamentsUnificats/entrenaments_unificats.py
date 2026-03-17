import sys
import os
import argparse
import csv
import shutil
import logging
import types
from collections import deque
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy

import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import random
import torch
import numpy as np
from tqdm import tqdm

# Silenciar logs
logging.getLogger().setLevel(logging.ERROR)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)

from rlcard.agents import DQNAgent, NFSPAgent, RandomAgent
from rlcard.utils import set_seed
from joc.entorn import TrucEnv, reorganize_amb_rewards
from RL.models.xarxa_unificada import XarxaUnificada

# Silenciar loggers de RLCard
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)

SEED = 42

# Config de l'entorn
ENV_CONFIG = {
    'num_jugadors': 2,
    'cartes_jugador': 3,
    'puntuacio_final': 24,
    'seed': SEED,
    'verbose': False,
}

MLP_LAYERS = [512, 512]
MLP_LAYERS_NFSP_SL = [1024, 512]

# DQN
DQN_LR          = 1e-4
DQN_BATCH       = 4096
DQN_MEMORY      = 1_000_000 
DQN_UPDATE_TGT  = 2000
DQN_EPS_MIN     = 0.05
OPP_UPDATE_TAU  = 0.05
DQN_TRAIN_EVERY = 64

# NFSP
NFSP_RL_LR      = 1e-4
NFSP_SL_LR      = 1e-4
NFSP_BATCH      = 4096
NFSP_RESERVOIR  = 1_000_000
NFSP_Q_REPLAY   = 1_000_000
NFSP_Q_UPDATE   = 1000
NFSP_ETA        = 0.25
NFSP_TRAIN_EVERY = 64
USE_BATCH_NORM = True

FINETUNE_LR_COS = 1e-5
FINETUNE_LR_MLP = 5e-5
DECAY_START = 0.75
DECAY_END = 0.95
DECAY_MIN_FACTOR = 0.2
POOL_SIZE = 20
EVALUATE_NUM = 200
DENSE_SAVE = 5000
MILESTONE_SAVE = 25000

class AgentCongelat:
    """Wrapper per usar eval_step"""
    def __init__(self, agent):
        self.agent = agent
        self.use_raw = False

    def step(self, state):
        action, _ = self.agent.eval_step(state)
        return action

    def eval_step(self, state):
        return self.agent.eval_step(state)

def wrap_env_aplanat(env):
    """Aplanem l'OBS per els agents RLCard"""
    original_extract_state = env._extract_state

    def _extract_state_patched(self, state):
        extracted = original_extract_state(state)
        obs = extracted['obs']
        if isinstance(obs, dict):
            flat = np.concatenate([
                obs['obs_cartes'].flatten(),
                obs['obs_context'],
            ], axis=0).astype(np.float32)
            extracted['obs'] = flat
        return extracted

    env._extract_state = types.MethodType(_extract_state_patched, env)
    return env

def avaluar(env, n):
    rewards = [env.run(is_training=False)[1][0] for _ in range(n)]
    mean_r = np.mean(rewards)
    win_rate = np.sum(np.array(rewards) > 0) / n
    return mean_r, win_rate

def get_decay_steps(episodes):
    return [episodes // 4, episodes // 2, 3 * episodes // 4]

def apply_lr_decay(optimizer, ep, episodes):
    """Decay progressiu del LR entre DECAY_START i DECAY_END"""
    decay_start_ep = int(episodes * DECAY_START)
    decay_end_ep = int(episodes * DECAY_END)

    if ep >= decay_start_ep:
        progress = min(1.0, (ep - decay_start_ep) / (decay_end_ep - decay_start_ep))
        lr_factor = DECAY_MIN_FACTOR + (1.0 - DECAY_MIN_FACTOR) * (1.0 - progress)
        for pg in optimizer.param_groups:
            pg['lr'] = pg.get('initial_lr', pg['lr'] / lr_factor) * lr_factor

def carregar_pesos(agent, path, device, verbose=True):
    """Carrega els pesos d'un agent des d'un fitxer .pt"""
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(agent, NFSPAgent):
        q_sd = sd['q'] if 'q' in sd else sd
        sl_sd = sd['sl'] if 'sl' in sd else sd
        
        agent._rl_agent.q_estimator.qnet.load_state_dict(q_sd)
        agent.policy_network.load_state_dict(sl_sd)
    else:
        # Per a DQN sol ser el state_dict de l'MLP directament
        q_sd = sd['q_net'] if (isinstance(sd, dict) and 'q_net' in sd) else sd
        agent.q_estimator.qnet.load_state_dict(q_sd)
    if verbose:
        print(f"Pesos de l'MLP carregats correctament des de {os.path.basename(path)}")


# Injectem XarxaUnificada a RLCARD

#------ DQN ------
def inject_xarxa_dqn(agent, xarxa, mode, lr):
    
    agent.q_estimator.qnet = xarxa
    agent.target_estimator.qnet = deepcopy(xarxa)

    if mode == "finetune":
        params = xarxa.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=FINETUNE_LR_MLP)
        agent.q_estimator.optimizer = torch.optim.Adam(params)
    else:
        params = filter(lambda p: p.requires_grad, xarxa.parameters())
        agent.q_estimator.optimizer = torch.optim.Adam(params, lr=lr)

def init_dqn(env, device, mode, ruta=None, layers=None, use_bn=None):
    layers = layers or MLP_LAYERS
    use_bn = use_bn if use_bn is not None else USE_BATCH_NORM
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=layers,
        learning_rate=DQN_LR,
        batch_size=DQN_BATCH,
        replay_memory_size=DQN_MEMORY,
        replay_memory_init_size=DQN_BATCH,
        update_target_estimator_every=DQN_UPDATE_TGT,
        discount_factor=0.995,
        epsilon_decay_steps=200_000,
        epsilon_end=DQN_EPS_MIN,
        train_every=32,
        device=device,
    )
    
    x = XarxaUnificada(env.num_actions, layers, mode, ruta, device, "q", use_bn=use_bn)
    inject_xarxa_dqn(agent, x, mode, DQN_LR)
    
    return agent



#------ NFSP ------
def inject_xarxes_nfsp(agent, q_net, sl_net, mode, rl_lr, sl_lr):
    
    # Part RL
    agent._rl_agent.q_estimator.qnet = q_net
    agent._rl_agent.target_estimator.qnet = deepcopy(q_net)
    
    if mode == "finetune":
        p_q = q_net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=FINETUNE_LR_MLP)
        agent._rl_agent.q_estimator.optimizer = torch.optim.Adam(p_q)
    else:
        p_q = filter(lambda p: p.requires_grad, q_net.parameters())
        agent._rl_agent.q_estimator.optimizer = torch.optim.Adam(p_q, lr=rl_lr)

    # Part SL
    agent.policy_network = sl_net
    if mode == "finetune":
        p_sl = sl_net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=FINETUNE_LR_MLP)
        agent.policy_network_optimizer = torch.optim.Adam(p_sl, weight_decay=1e-5)
    else:
        p_sl = filter(lambda p: p.requires_grad, sl_net.parameters())
        agent.policy_network_optimizer = torch.optim.Adam(p_sl, lr=sl_lr, weight_decay=1e-5)

def init_nfsp(env, device, mode, ruta=None, layers_q=None, layers_sl=None, use_bn=None):
    layers_q = layers_q or MLP_LAYERS
    layers_sl = layers_sl or MLP_LAYERS_NFSP_SL
    use_bn = use_bn if use_bn is not None else USE_BATCH_NORM
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=layers_sl,
        q_mlp_layers=layers_q,
        rl_learning_rate=NFSP_RL_LR,
        sl_learning_rate=NFSP_SL_LR,
        batch_size=NFSP_BATCH,
        reservoir_buffer_capacity=NFSP_RESERVOIR,
        q_replay_memory_size=NFSP_Q_REPLAY,
        q_update_target_estimator_every=NFSP_Q_UPDATE,
        q_discount_factor=0.995,
        anticipatory_param=NFSP_ETA,
        q_epsilon_decay_steps=200_000,
        q_epsilon_end=DQN_EPS_MIN,
        q_replay_memory_init_size=NFSP_BATCH,
        q_batch_size=NFSP_BATCH,
        q_train_every=32,
        device=device,
    )
    
    q = XarxaUnificada(env.num_actions, layers_q, mode, ruta, device, "q", use_bn=use_bn)
    sl = XarxaUnificada(env.num_actions, layers_sl, mode, ruta, device, "policy", use_bn=use_bn)
    inject_xarxes_nfsp(agent, q, sl, mode, NFSP_RL_LR, NFSP_SL_LR)
    
    original_feed = agent.feed
    agent._sl_counter = 0
    def fast_feed(ts):
        agent._sl_counter += 1
        if agent._sl_counter % NFSP_TRAIN_EVERY == 0:
            original_feed(ts)
        else:
            if hasattr(agent, 'reservoir'):
                agent.reservoir.add(ts[0]['obs'], ts[1])
            agent._rl_agent.feed(ts)
    
    agent.feed = fast_feed
    return agent



class SelectorAdversari:
    """Gestiona qui toca en cada partida de l'entrenament."""
    def __init__(self, agent_principal, pool_base, device, model_pool, ruta_millor):
        self.agent = agent_principal
        self.pool_base = pool_base
        self.device = device
        self.model_pool = model_pool
        self.ruta_millor = ruta_millor
        self.adv_aleatori = RandomAgent(agent_principal.num_actions)

    def triar(self, chance, agent_self_play=None):
        if chance < 0.05:
            return self.adv_aleatori # 5% Random
        
        elif chance < 0.25 and self.ruta_millor.exists():
            carregar_pesos(self.pool_base, self.ruta_millor, self.device, verbose=False)
            return AgentCongelat(self.pool_base) # 20% contra el millor que tenim
        
        elif chance < 0.80:
            return agent_self_play if agent_self_play else AgentCongelat(self.pool_base) # 55% pràctica normal
        
        else:
            ruta_passat = random.choice(self.model_pool)
            carregar_pesos(self.pool_base, ruta_passat, self.device, verbose=False)
            return AgentCongelat(self.pool_base) # 20% contra pool

class SupervisorEntreno:
    """Controla la marxa de l'entreno: logs, avaluació i fitxers."""
    def __init__(self, model_dir, log_dir, tag, num_episodis):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.tag = tag
        self.num_episodis = num_episodis
        self.millor_r = -999
        self.rewards_cua = []
        self.freq_aval = max(num_episodis // 40, 2500)
        
        self.fitxer_log = log_dir / f"{tag}.csv"
        capcalera = ["ep", "reward", "vic%", "r_train", "vic_train%", "best", "is_best", "lr"]
        with open(self.fitxer_log, "w", newline="") as f:
            csv.writer(f).writerow(capcalera)

    def guardar_r(self, r):
        self.rewards_cua.append(r)

    def avaluar_i_guardar(self, ep, agent, env_aval, optimitzador):
        if ep % self.freq_aval != 0:
            return False

        r, tx_victories = avaluar(env_aval, EVALUATE_NUM)
        vic = round(100 * tx_victories, 1)
        
        r_train = np.mean(self.rewards_cua) if self.rewards_cua else 0
        vic_train = round(100 * np.sum(np.array(self.rewards_cua) > 0) / len(self.rewards_cua), 1) if self.rewards_cua else 0
        self.rewards_cua = []

        lr = optimitzador.param_groups[-1]['lr']
        es_millor = 0
        marca = ""

        if r > self.millor_r:
            self.millor_r = r
            if hasattr(agent, '_rl_agent'): # NFSP
                estat = {'q': agent._rl_agent.q_estimator.qnet.state_dict(), 
                         'sl': agent.policy_network.state_dict()}
            else: # DQN
                estat = agent.q_estimator.qnet.state_dict()
            
            torch.save(estat, self.model_dir / "best.pt")
            marca = " *"
            es_millor = 1

        tqdm.write(f"EP {ep} | R {r:.3f} | V {vic}% | RT {r_train:.3f} | VT {vic_train}% | LR {lr:.2e}{marca}")
        
        with open(self.fitxer_log, "a", newline="") as f:
            csv.writer(f).writerow([ep, r, vic, r_train, vic_train, self.millor_r, es_millor, lr])
        
        return es_millor

    def guardar_checkpoint(self, ep, agent, model_pool):
        if ep % DENSE_SAVE == 0:
            ruta = self.model_dir / f"ep{ep}_dense.pt"
            if hasattr(agent, '_rl_agent'):
                estat = {'q': agent._rl_agent.q_estimator.qnet.state_dict(), 'sl': agent.policy_network.state_dict()}
            else:
                estat = agent.q_estimator.qnet.state_dict()
            
            torch.save(estat, ruta)
            
            if len(model_pool) >= POOL_SIZE:
                antic = model_pool.popleft()
                if antic.exists() and "milestone" not in antic.name and "init" not in antic.name:
                    try: antic.unlink()
                    except: pass
            model_pool.append(ruta)

        if ep % MILESTONE_SAVE == 0:
            ruta = self.model_dir / f"ep{ep}_milestone.pt"
            if hasattr(agent, '_rl_agent'):
                estat = {'q': agent._rl_agent.q_estimator.qnet.state_dict(), 'sl': agent.policy_network.state_dict()}
            else:
                estat = agent.q_estimator.qnet.state_dict()
            torch.save(estat, ruta)

    def finalitzar(self, agent):
        final = self.model_dir / "final.pt"
        best = self.model_dir / "best.pt"
        if best.exists(): 
            shutil.copy2(best, final)
        else:
            if hasattr(agent, '_rl_agent'):
                torch.save({'q': agent._rl_agent.q_estimator.qnet.state_dict(), 'sl': agent.policy_network.state_dict()}, final)
            else:
                torch.save(agent.q_estimator.qnet.state_dict(), final)

        for p in self.model_dir.glob("*.pt"):
            if not (p.name in ["final.pt", "best.pt"] or "milestone" in p.name):
                try: p.unlink()
                except: pass
        print(f"Neteja feta a {self.model_dir}")


# Entrenament
def run_dqn(mode, episodes, run_dir, model_dir, log_dir, device, eval_model_path=None):
    env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))

    agent = init_dqn(env, device, mode)
    opp_polyak = init_dqn(env, device, mode)
    opp_polyak.q_estimator.qnet.load_state_dict(agent.q_estimator.qnet.state_dict())
    
    eps_start = 0.5 if mode == "finetune" else 1.0
    agent.epsilons = np.linspace(eps_start, DQN_EPS_MIN, episodes * 100)
    
    init_path = model_dir / "init.pt"
    torch.save(agent.q_estimator.qnet.state_dict(), init_path)
    model_pool = deque([init_path], maxlen=POOL_SIZE)

    selector = SelectorAdversari(agent, init_dqn(env, device, mode), device, model_pool, model_dir / "best.pt")
    supervisor = SupervisorEntreno(model_dir, log_dir, f"dqn_{mode}", episodes)

    agent_aval = RandomAgent(env.num_actions)
    if eval_model_path and os.path.exists(eval_model_path):
        try:
            opp_aval = init_dqn(env, device, mode="frozen")
            carregar_pesos(opp_aval, eval_model_path, device)
            agent_aval = AgentCongelat(opp_aval)
        except: pass
    
    eval_env.set_agents([agent, agent_aval])

    if mode == "finetune":
        for p in agent.q_estimator.qnet.cos.parameters(): p.requires_grad = False

    print(f"\nEntrenant DQN ({mode.upper()}) | {episodes} eps")
    for ep in tqdm(range(1, episodes + 1), desc="Train"):
        
        adversari = selector.triar(random.random(), AgentCongelat(opp_polyak))
        env.set_agents([agent, adversari])

        # Si estem fent finetune, descongelem el cos quan portem un 15%
        if mode == "finetune" and ep == int(episodes * 0.15):
            for p in agent.q_estimator.qnet.cos.parameters(): p.requires_grad = True
            params = agent.q_estimator.qnet.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=DQN_LR)
            agent.q_estimator.optimizer = torch.optim.Adam(params)

        traj, payoffs = env.run(is_training=True)
        supervisor.guardar_r(payoffs[0])
        
        # Passem la trajectòria a l'agent
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                for ts in reorganize_amb_rewards(traj, payoffs)[0]: agent.feed(ts)

        # Actualització Polyak
        with torch.no_grad():
            for p, tp in zip(agent.q_estimator.qnet.parameters(), opp_polyak.q_estimator.qnet.parameters()):
                tp.data.lerp_(p.data, OPP_UPDATE_TAU)

        # Baixem el Learning Rate segons el progrés
        if ep == 1:
            for pg in agent.q_estimator.optimizer.param_groups: pg['initial_lr'] = pg['lr']
        apply_lr_decay(agent.q_estimator.optimizer, ep, episodes)

        # Mirem com va tot
        supervisor.avaluar_i_guardar(ep, agent, eval_env, agent.q_estimator.optimizer)
        supervisor.guardar_checkpoint(ep, agent, model_pool)

    supervisor.finalitzar(agent)


def run_nfsp(mode, episodes, run_dir, model_dir, log_dir, device, eval_model_path=None):
    env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))

    p0 = init_nfsp(env, device, mode)
    p1 = init_nfsp(env, device, mode)
    
    eps_start = 0.5 if mode == "finetune" else 1.0
    for ag in [p0, p1]:
        ag._rl_agent.epsilons = np.linspace(eps_start, DQN_EPS_MIN, episodes * 20)

    init_path = model_dir / "init_nfsp.pt"
    torch.save({'q': p0._rl_agent.q_estimator.qnet.state_dict(), 'sl': p0.policy_network.state_dict()}, init_path)
    model_pool = deque([init_path], maxlen=POOL_SIZE)

    selector = SelectorAdversari(p0, init_nfsp(env, device, mode), device, model_pool, model_dir / "best.pt")
    supervisor = SupervisorEntreno(model_dir, log_dir, f"nfsp_{mode}", episodes)

    agent_aval = RandomAgent(env.num_actions)
    if eval_model_path and os.path.exists(eval_model_path):
        try:
            opp_aval = init_dqn(env, device, mode="frozen")
            carregar_pesos(opp_aval, eval_model_path, device)
            agent_aval = AgentCongelat(opp_aval)
        except: pass
    eval_env.set_agents([p0, agent_aval])

    if mode == "finetune":
        for ag in [p0, p1]:
            for p in ag._rl_agent.q_estimator.qnet.cos.parameters(): p.requires_grad = False
            for p in ag.policy_network.cos.parameters(): p.requires_grad = False

    print(f"\nEntrenant NFSP ({mode.upper()}) | {episodes} eps")
    for ep in tqdm(range(1, episodes + 1), desc="Train"):
        
        # Triem adversari
        adversari = selector.triar(random.random(), p1)
        env.set_agents([p0, adversari])

        # Descongelem el cos si portem un 15% (finetune)
        if mode == "finetune" and ep == int(episodes * 0.15):
            for ag in [p0, p1]:
                for p in ag._rl_agent.q_estimator.qnet.cos.parameters(): p.requires_grad = True
                for p in ag.policy_network.cos.parameters(): p.requires_grad = True
                ag._rl_agent.q_estimator.optimizer = torch.optim.Adam(ag._rl_agent.q_estimator.qnet.get_param_groups(FINETUNE_LR_COS, FINETUNE_LR_MLP))
                ag.policy_network_optimizer = torch.optim.Adam(ag.policy_network.get_param_groups(FINETUNE_LR_COS, FINETUNE_LR_MLP))

        traj, payoffs = env.run(is_training=True)
        supervisor.guardar_r(payoffs[0])
        traj = reorganize_amb_rewards(traj, payoffs)

        # Simulem l'aprenentatge
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                for ts in traj[0]: p0.feed(ts)
                if adversari == p1:
                    for ts in traj[1]: p1.feed(ts)

        # Iniciem el LR decay
        if ep == 1:
            for ag in [p0, p1]:
                for pg in ag._rl_agent.q_estimator.optimizer.param_groups: pg['initial_lr'] = pg['lr']
                for pg in ag.policy_network_optimizer.param_groups: pg['initial_lr'] = pg['lr']

        for ag in [p0, p1]:
            apply_lr_decay(ag._rl_agent.q_estimator.optimizer, ep, episodes)
            apply_lr_decay(ag.policy_network_optimizer, ep, episodes)

        # Cada cert temps, fixem l'estratègia (NFSP)
        if ep % supervisor.freq_aval == 0:
            p0.sample_episode_policy(); p1.sample_episode_policy()
        
        supervisor.avaluar_i_guardar(ep, p0, eval_env, p0._rl_agent.q_estimator.optimizer)
        supervisor.guardar_checkpoint(ep, p0, model_pool)

    supervisor.finalitzar(p0)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "nfsp"], required=True)
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"], required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval_model", type=str, default=None, help="Ruta al model .pt per a l'avaluació")
    args = parser.parse_args()

    set_seed(SEED)
    # Configuració GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"[ACCELERACIÓ] Usant GPU: {gpu_name}")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("[AVÍS] GPU no disponible, usant CPU.")
    
    tag = f"{args.agent}_{args.mode}_{datetime.now().strftime('%d%m_%H%M')}"
    
    
    # Guardar
    run_dir = Path(__file__).parent / "registres" / tag
    model_dir, log_dir = run_dir / "models", run_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Guardant a: {run_dir}")

    if args.agent == "dqn":
        run_dqn(args.mode, args.episodes, run_dir, model_dir, log_dir, device, args.eval_model)
    else:
        run_nfsp(args.mode, args.episodes, run_dir, model_dir, log_dir, device, args.eval_model)

if __name__ == "__main__":
    main()
