import sys
import os
import argparse
import csv
import shutil
import logging
import types
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import random
import torch
import numpy as np
from tqdm import tqdm

from rlcard.agents import DQNAgent, NFSPAgent, RandomAgent
from rlcard.utils import set_seed, reorganize

from joc.entorn import TrucEnv
from RL.models.xarxa_unificada import XarxaUnificada

# Silenciar loggers de RLCard
logging.basicConfig(level=logging.ERROR, force=True)
logging.getLogger().setLevel(logging.ERROR)

SEED = 42

# Config de l'entorn
ENV_CONFIG = {
    'num_jugadors': 2,
    'cartes_jugador': 3,
    'puntuacio_final': 12,
    'seed': SEED,
    'verbose': False,
}

MLP_LAYERS = [256, 256]

# DQN
DQN_LR          = 2e-4
DQN_BATCH       = 256
DQN_MEMORY      = 200_000
DQN_UPDATE_TGT  = 1000
DQN_EPS_MIN     = 0.1
OPP_UPDATE_TAU  = 0.05  # Factor per a l'actualització suau de l'oponent

# NFSP
NFSP_RL_LR      = 5e-4
NFSP_SL_LR      = 5e-4
NFSP_BATCH      = 256
NFSP_RESERVOIR  = 100_000
NFSP_Q_REPLAY   = 100_000
NFSP_Q_UPDATE   = 500
NFSP_ETA        = 0.3

FINETUNE_LR_COS = 1e-5
LR_DECAY_FACTOR = 0.5
EVALUATE_NUM = 1000

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
    total = sum(env.run(is_training=False)[1][0] for _ in range(n))
    return total / n

def get_decay_steps(episodes):
    return [episodes // 4, episodes // 2, 3 * episodes // 4]

def carregar_pesos(agent, path, device):
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
    print(f"Pesos de l'MLP carregats correctament des de {os.path.basename(path)}")


# Injectem XarxaUnificada a RLCARD

#------ DQN ------
def inject_xarxa_dqn(agent, xarxa, mode, lr):
    
    agent.q_estimator.qnet = xarxa
    agent.target_estimator.qnet = deepcopy(xarxa)

    if mode == "finetune":
        params = xarxa.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=lr)
        agent.q_estimator.optimizer = torch.optim.Adam(params)
    else:
        params = filter(lambda p: p.requires_grad, xarxa.parameters())
        agent.q_estimator.optimizer = torch.optim.Adam(params, lr=lr)

def init_dqn(env, device, mode, ruta=None):
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=MLP_LAYERS,
        learning_rate=DQN_LR,
        batch_size=DQN_BATCH,
        replay_memory_size=DQN_MEMORY,
        replay_memory_init_size=DQN_BATCH,
        update_target_estimator_every=DQN_UPDATE_TGT,
        epsilon_decay_steps=200_000,
        epsilon_end=DQN_EPS_MIN,
        device=device,
    )
    
    x = XarxaUnificada(env.num_actions, MLP_LAYERS, mode, ruta, device, "q")
    inject_xarxa_dqn(agent, x, mode, DQN_LR)
    
    return agent



#------ NFSP ------
def inject_xarxes_nfsp(agent, q_net, sl_net, mode, rl_lr, sl_lr):
    
    # Part RL
    agent._rl_agent.q_estimator.qnet = q_net
    agent._rl_agent.target_estimator.qnet = deepcopy(q_net)
    
    if mode == "finetune":
        p_q = q_net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=rl_lr)
        agent._rl_agent.q_estimator.optimizer = torch.optim.Adam(p_q)
    else:
        p_q = filter(lambda p: p.requires_grad, q_net.parameters())
        agent._rl_agent.q_estimator.optimizer = torch.optim.Adam(p_q, lr=rl_lr)

    # Part SL
    agent.policy_network = sl_net
    if mode == "finetune":
        p_sl = sl_net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=sl_lr)
        agent.policy_network_optimizer = torch.optim.Adam(p_sl)
    else:
        p_sl = filter(lambda p: p.requires_grad, sl_net.parameters())
        agent.policy_network_optimizer = torch.optim.Adam(p_sl, lr=sl_lr)

def init_nfsp(env, device, mode, ruta=None):
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=MLP_LAYERS,
        q_mlp_layers=MLP_LAYERS,
        rl_learning_rate=NFSP_RL_LR,
        sl_learning_rate=NFSP_SL_LR,
        batch_size=NFSP_BATCH,
        reservoir_buffer_capacity=NFSP_RESERVOIR,
        q_replay_memory_size=NFSP_Q_REPLAY,
        q_update_target_estimator_every=NFSP_Q_UPDATE,
        anticipatory_param=NFSP_ETA,
        device=device,
    )
    
    q = XarxaUnificada(env.num_actions, MLP_LAYERS, mode, ruta, device, "q")
    sl = XarxaUnificada(env.num_actions, MLP_LAYERS, mode, ruta, device, "policy")
    inject_xarxes_nfsp(agent, q, sl, mode, NFSP_RL_LR, NFSP_SL_LR)
    
    return agent



# Entrenaments
def run_dqn(mode, episodes, model_dir, log_dir, device, eval_model_path=None):
    
    env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))

    # Agent principal configuració
    agent = init_dqn(env, device, mode)
    agent.epsilon_decay_steps = episodes
    agent.epsilons = np.linspace(1.0, DQN_EPS_MIN, agent.epsilon_decay_steps)

    # Oponent Polyak
    opp_polyak_base = init_dqn(env, device, mode)
    opp_polyak_base.q_estimator.qnet.load_state_dict(agent.q_estimator.qnet.state_dict())
    
    # Oponent Pool
    opp_pool_base = init_dqn(env, device, mode)
    
    # comença amb l'estat inicial
    init_path = model_dir / "init.pt"
    torch.save(agent.q_estimator.qnet.state_dict(), init_path)
    model_pool = [init_path]

    env.set_agents([agent, AgentCongelat(opp_polyak_base)])
    
    # Configuració de l'avaluació
    if eval_model_path and os.path.exists(eval_model_path):
        try:
            eval_opp_base = init_dqn(env, device, mode="frozen")
            carregar_pesos(eval_opp_base, eval_model_path, device)
            eval_agent = AgentCongelat(eval_opp_base)
            print(f"Avaluació contra model: {eval_model_path}")
        except Exception as e:
            print(f"Error carregant model d'avaluació: {e}. Usant RandomAgent per defecte.")
            eval_agent = RandomAgent(env.num_actions)
    else:
        eval_agent = RandomAgent(env.num_actions)
        print("Avaluació contra RandomAgent")
    
    eval_env.set_agents([agent, eval_agent])

    decay_steps = get_decay_steps(episodes)
    save_every = max(episodes // 10, 1000)

    # Preparar fitxer de logs
    log_file = log_dir / f"dqn_{mode}.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["ep", "reward", "vic%", "best", "is_best", "lr"])

    print(f"\nIniciant DQN ({mode.upper()}) | {episodes} eps")
    best_r = -999
    
    # Bucle d'entrenament
    random_opponent = RandomAgent(env.num_actions)
    for ep in tqdm(range(1, episodes + 1), desc="Train", unit="ep"):
        
        #Selecció d'oponent
        chance = random.random()
        if chance < 0.20:
            env.set_agents([agent, random_opponent]) # 20% contra Random
        elif chance < 0.60:
            env.set_agents([agent, AgentCongelat(opp_polyak_base)]) # 40% contra Polyak (Latest)
        else:
            # 40% contra una versió random del pool
            past_path = random.choice(model_pool)
            opp_pool_base.q_estimator.qnet.load_state_dict(torch.load(past_path, map_location=device, weights_only=True))
            env.set_agents([agent, AgentCongelat(opp_pool_base)])

        # Reward Scheduling
        progress = ep / episodes
        current_beta = 0.5 + (0.5 * progress)
        env.set_reward_beta(current_beta)

        traj, payoffs = env.run(is_training=True)
        traj = reorganize(traj, payoffs)
        
        with redirect_stdout(open(os.devnull, 'w')):
            for ts in traj[0]: agent.feed(ts)

        #learning rate
        if ep in decay_steps:
            for pg in agent.q_estimator.optimizer.param_groups: pg['lr'] *= LR_DECAY_FACTOR
            print(f"\nLR Decay: {agent.q_estimator.optimizer.param_groups[-1]['lr']:.2e}")

        # Avaluació
        eval_freq = max(episodes // 200, 250)
        if ep % eval_freq == 0:
            r = avaluar(eval_env, EVALUATE_NUM)
            vic = round(100 * (r + 1) / 2, 1)
            lr = agent.q_estimator.optimizer.param_groups[-1]['lr']

            if r > best_r:
                best_r = r
                torch.save(agent.q_estimator.qnet.state_dict(), model_dir / f"best.pt")
                indicador = " *"
                is_best = 1
            else:
                indicador = ""
                is_best = 0

            # Actualització suau de l'oponent Polyak
            with torch.no_grad():
                for param, target_param in zip(agent.q_estimator.qnet.parameters(), opp_polyak_base.q_estimator.qnet.parameters()):
                    target_param.data.copy_(OPP_UPDATE_TAU * param.data + (1.0 - OPP_UPDATE_TAU) * target_param.data)

            tqdm.write(f"EP {ep} | R {r:.3f} | V {vic}% | LR {lr:.2e}{indicador}")
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([ep, r, vic, best_r, is_best, lr])

        # Guardar checkpoints
        if ep % save_every == 0:
            path = model_dir / f"ep{ep}.pt"
            torch.save(agent.q_estimator.qnet.state_dict(), path)
            model_pool.append(path)

    # Guardar final
    final = model_dir / "final.pt"
    if (model_dir / "best.pt").exists(): shutil.copy2(model_dir / "best.pt", final)
    else: torch.save(agent.q_estimator.qnet.state_dict(), final)

    # Netejar pesos intermedis
    for p in model_dir.glob("*.pt"):
        if p.name not in ["final.pt", "best.pt"]:
            p.unlink()
    print(f"Pesos intermedis esborrats. Només queden 'final.pt' i 'best.pt'.")


def run_nfsp(mode, episodes, model_dir, log_dir, device, eval_model_path=None):
    
    env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    eval_env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))

    # Agents per al self-play
    p0 = init_nfsp(env, device, mode)
    p1 = init_nfsp(env, device, mode)
    
    env.set_agents([p0, p1])
    
    # Configuració de l'avaluació
    if eval_model_path and os.path.exists(eval_model_path):
        try:
            eval_opp_base = init_dqn(env, device, mode="frozen")
            carregar_pesos(eval_opp_base, eval_model_path, device)
            eval_agent = AgentCongelat(eval_opp_base)
            print(f"Avaluació contra model: {eval_model_path}")
        except Exception as e:
            print(f"Error carregant model d'avaluació: {e}. Usant RandomAgent per defecte.")
            eval_agent = RandomAgent(env.num_actions)
    else:
        eval_agent = RandomAgent(env.num_actions)
        print("Avaluació contra RandomAgent")

    eval_env.set_agents([p0, eval_agent])

    decay_steps = get_decay_steps(episodes)
    save_every = max(episodes // 10, 1000)

    # Inicialitzar CSV
    log_file = log_dir / f"nfsp_{mode}.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["ep", "r0", "vic0", "m0", "m1", "best", "is_best"])

    print(f"\nIniciant NFSP ({mode.upper()}) | {episodes} eps")
    best_r = -999

    # Bucle d'entrenament
    for ep in tqdm(range(1, episodes + 1), desc="Train", unit="ep"):
        # Reward Scheduling
        progress = ep / episodes
        current_beta = 0.5 + (0.5 * progress)
        env.set_reward_beta(current_beta)

        traj, payoffs = env.run(is_training=True)
        traj = reorganize(traj, payoffs)

        with redirect_stdout(open(os.devnull, 'w')):
            for pid, ag in enumerate([p0, p1]):
                for ts in traj[pid]: ag.feed(ts)

        # LR Decay
        if ep in decay_steps:
            for ag in [p0, p1]:
                for pg in ag._rl_agent.q_estimator.optimizer.param_groups: pg['lr'] *= LR_DECAY_FACTOR
                for pg in ag.policy_network_optimizer.param_groups: pg['lr'] *= LR_DECAY_FACTOR
            print(f"\nLR Decay")

        # Avaluació
        eval_freq = max(episodes // 200, 250)
        if ep % eval_freq == 0:
            p0.sample_episode_policy(); p1.sample_episode_policy()
            r = avaluar(eval_env, EVALUATE_NUM)
            vic = round(100 * (r + 1) / 2, 1)
            m0, m1 = p0._mode[0], p1._mode[0]

            if r > best_r:
                best_r = r
                for i, ag in enumerate([p0, p1]):
                    torch.save({'q': ag._rl_agent.q_estimator.qnet.state_dict(), 
                                'sl': ag.policy_network.state_dict()}, model_dir / f"best_p{i}.pt")
                indicador = "*"
                is_best = 1
            else:
                indicador = ""
                is_best = 0

            tqdm.write(f"EP {ep} | R0 {r:.3f} | V0 {vic}% | MODES {m0}-{m1}{indicador}")
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([ep, r, vic, m0, m1, best_r, is_best])

        # Guardar checkpoints
        if ep % save_every == 0:
            for i, ag in enumerate([p0, p1]):
                torch.save({'q': ag._rl_agent.q_estimator.qnet.state_dict(), 
                            'sl': ag.policy_network.state_dict()}, model_dir / f"ep{ep}_p{i}.pt")

    
    # Playoff per triar el millor p0 o p1
    print("\nPlayoff final P0 vs P1...")
    
    # Carreguem els millors
    b0 = init_nfsp(env, device, mode)
    b1 = init_nfsp(env, device, mode)
    
    p0_path = model_dir / "best_p0.pt"
    if not p0_path.exists():
        torch.save({'q': p0._rl_agent.q_estimator.qnet.state_dict(), 'sl': p0.policy_network.state_dict()}, p0_path)
    
    p1_path = model_dir / "best_p1.pt"
    if not p1_path.exists():
        torch.save({'q': p1._rl_agent.q_estimator.qnet.state_dict(), 'sl': p1.policy_network.state_dict()}, p1_path)

    carregar_pesos(b0, p0_path, device)
    carregar_pesos(b1, p1_path, device)

    play_env = wrap_env_aplanat(TrucEnv(ENV_CONFIG))
    wins = [0, 0]
    for _ in range(1000):
        play_env.set_agents([b0, b1] if _ % 2 == 0 else [b1, b0])
        _, p = play_env.run(False)
        if (p[0] > p[1] if _ % 2 == 0 else p[1] > p[0]): wins[0] += 1
        else: wins[1] += 1
    
    winner = 0 if wins[0] >= wins[1] else 1
    print(f"Guanyador: P{winner} ({wins[0]}-{wins[1]})")
    shutil.copy2(model_dir / f"best_p{winner}.pt", model_dir / "final.pt")

    # Netejar pesos intermedis
    for p in model_dir.glob("*.pt"):
        if p.name not in ["final.pt", "best_p0.pt", "best_p1.pt"]:
            p.unlink()
    print(f"Pesos intermedis esborrats. Només queden 'final.pt' i els millors de cada agent.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "nfsp"], required=True)
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"], required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval_model", type=str, default=None, help="Ruta al model .pt per a l'avaluació")
    args = parser.parse_args()

    set_seed(SEED)
    # Configuració GPU i rendiment PyTorch
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
    run_dir = Path(__file__).parent / "registres" / tag
    model_dir, log_dir = run_dir / "models", run_dir / "logs"
    model_dir.mkdir(parents=True); log_dir.mkdir(parents=True)

    print(f"Guardant a: {run_dir}")

    if args.agent == "dqn":
        run_dqn(args.mode, args.episodes, model_dir, log_dir, device, args.eval_model)
    else:
        run_nfsp(args.mode, args.episodes, model_dir, log_dir, device, args.eval_model)

if __name__ == "__main__":
    main()
