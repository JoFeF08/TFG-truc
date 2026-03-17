import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm, trange
import csv
import random
from datetime import datetime

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from RL.entrenament.entrenamentsPropis.parallel_env import SubprocVecEnv
from RL.models.model_propi.ppo.cap_ppo_mlp import PPOMlpNet, SPLIT, OBS_CONTEXT_SIZE
from RL.models.model_propi.ppo.agent_ppo_mlp import PPOMlpAgent
from RL.entrenament.entrenamentsPropis.ppo.buffers_ppo import RolloutBuffer
from rlcard.agents import RandomAgent
from joc.entorn.env import TrucEnv
from RL.entrenament.entrenamentsPropis.ppo_loss import calcular_gae, calcular_perdua_ppo
from joc.entorn.cartes_accions import ACTION_LIST

# Hyperparams Constants
NUM_ENVS = 48
NUM_STEPS = 512
MINIBATCH_SIZE = 1024
UPDATE_EPOCHS = 10
TOTAL_TIMESTEPS = 24_000_000
LR = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

# Fine-tune Constants
FINETUNE_LR_COS = 1e-5
FINETUNE_LR_MLP = 1e-4
UNFREEZE_FRACTION = 0.15

def extract_obs(states_list):
    b_obs = []
    b_masks = []
    
    for state in states_list:
        raw_obs = state['obs']
        flat = np.concatenate([raw_obs['obs_cartes'].flatten(), raw_obs['obs_context']], axis=0)
        b_obs.append(flat)
        
        mask = np.zeros(len(ACTION_LIST), dtype=bool)
        legal = list(state['legal_actions'].keys()) if hasattr(state['legal_actions'], 'keys') else state['legal_actions']
        mask[legal] = True
        b_masks.append(mask)

    return torch.tensor(np.array(b_obs), dtype=torch.float32), torch.tensor(np.array(b_masks), dtype=torch.bool)


def evaluar_contra_random(agent, env_config, device, num_partides=100):
    """Evalua l'agent PPO contra un agent que tria accions aleatòries."""
    eval_env = TrucEnv(env_config)
    eval_env.set_agents([agent, RandomAgent(num_actions=len(ACTION_LIST))])
    
    recompenses = []
    victories = 0
    
    for _ in range(num_partides):
        trajectoria, payoffs = eval_env.run(is_training=False)
        recompensa = payoffs[0]
        recompenses.append(recompensa)
        if recompensa > 0:
            victories += 1
            
    return np.mean(recompenses), (victories / num_partides) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"], default="frozen")
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    
    total_timesteps = args.total_timesteps
    unfreeze_step = int(total_timesteps * UNFREEZE_FRACTION)
    has_unfrozen = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Iniciant Entrenament PPO Base (MLP) - Mode: {args.mode.upper()}")
    
    env_config = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 24,
        'seed': 42
    }
    
    vec_env = SubprocVecEnv(NUM_ENVS, env_config)
    
    n_accions = len(ACTION_LIST)
    net = PPOMlpNet(n_actions=n_accions, hidden_size=256, device=device)
    agent = PPOMlpAgent(net, n_accions, device=device)
    
    # Pool d'Oponents
    pool_net = PPOMlpNet(n_actions=n_accions, hidden_size=256, device=device)
    pool_net.eval()
    opponent_pool = [] # Llista de rutes .pt
    
    # 20% pool
    NUM_POOL_ENVS = int(NUM_ENVS * 0.2)
    POOL_FREQUENCY = 500

    pool_player_ids = {} # env_idx -> id_jugador
    for i in range(NUM_ENVS - NUM_POOL_ENVS, NUM_ENVS):
        pool_player_ids[i] = i % 2
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)
    
    state_shape = SPLIT + OBS_CONTEXT_SIZE 
    buffer = RolloutBuffer(NUM_STEPS, NUM_ENVS, state_shape, action_dim=n_accions, device=device)
    
    # Inicialitzar entorns
    results = vec_env.reset_all()
    current_states = [res[0] for res in results]
    
    global_step = 0
    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    
    timestamp = datetime.now().strftime("%dd_%mm_%H%Mh")
    save_dir = Path(__file__).parent / "registres" / f"ppo_mlp_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training_log.csv"
    
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "pg_loss", "v_loss", "ent_loss", "reward_mean", "eval_wr", "eval_reward"])

    pbar = trange(1, num_updates + 1, desc="Actualitzacions")
    eval_wr, eval_rev = 0.0, 0.0
    
    for update in pbar:
        batch_rewards = []
        
        if update > 10 and update % POOL_FREQUENCY == 0:
            ckpt_path = save_dir / f"checkpoint_update_{update}.pt"
            torch.save(net.state_dict(), ckpt_path)
            opponent_pool.append(ckpt_path)
            
            random_pool_path = random.choice(opponent_pool)
            pool_net.load_state_dict(torch.load(random_pool_path, map_location=device, weights_only=True))
            print(f"\n[Pool] Oponent actualitzat a: {random_pool_path.name}")
            
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            
            obs_tensor, masks_tensor = extract_obs(current_states)
            obs_tensor = obs_tensor.to(device)
            masks_tensor = masks_tensor.to(device)
            
            active_players = [s['raw_obs']['id_jugador'] for s in current_states]
            
            # Decidir qui actua en cada env
            is_learning_step = torch.ones(NUM_ENVS, device=device)
            
            # Totes les accions triades per defecte pel Learner
            net.eval()
            with torch.no_grad():
                logits, value = net(obs_tensor)
                logits = logits.masked_fill(~masks_tensor, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            
            # Sobreescriure accions de la Pool si toca
            if len(opponent_pool) > 0:
                for i in pool_player_ids:
                    if active_players[i] == pool_player_ids[i]:
                        with torch.no_grad():
                            p_logits, _ = pool_net(obs_tensor[i:i+1])
                            p_logits = p_logits.masked_fill(~masks_tensor[i:i+1], -1e9)
                            p_action = torch.distributions.Categorical(logits=p_logits).sample()
                        
                        action[i] = p_action
                        is_learning_step[i] = 0.0
            
            actions_np = action.cpu().numpy()
            
            # Executar step
            next_states_players, rewards_list, dones_list = vec_env.step(actions_np)
            
            step_rewards = []
            for i in range(NUM_ENVS):
                step_rewards.append(rewards_list[i][active_players[i]])
                
            step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones_list, dtype=torch.float32).to(device)
            
            buffer.add(obs_tensor, action, logprob, step_rewards_tensor, value.squeeze(-1), dones_tensor, masks_tensor, is_learning_step)
            
            current_states = [sp[0] for sp in next_states_players]
            batch_rewards.extend(step_rewards)
            
        # PPO Update
        obs_tensor, _ = extract_obs(current_states)
        obs_tensor = obs_tensor.to(device)
        net.eval()
        with torch.no_grad():
            _, last_value = net(obs_tensor)
            last_value = last_value.squeeze(-1)
            
        avantatges, retorns = calcular_gae(
            buffer.rewards, buffer.values, buffer.dones, 
            last_value, torch.zeros_like(last_value), 
            GAMMA, GAE_LAMBDA
        )
        
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks, b_is_learning = buffer.get(avantatges, retorns)
        b_inds = np.arange(NUM_ENVS * NUM_STEPS)
        
        # Unfreeze per Fine-tune
        if args.mode == "finetune" and not has_unfrozen and global_step >= unfreeze_step:
            net.unfreeze_cos()
            params = net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=LR) # Mantenim LR base per l'MLP
            optimizer = optim.Adam(params, eps=1e-5)
            has_unfrozen = True
            print(f"[Fine-tune] COS descongelat al step {global_step}. Optimizer actualitzat.")

        net.train()
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                
                loss, pg_loss, v_loss, ent_loss = calcular_perdua_ppo(
                    agent, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], 
                    b_advantages[mb_inds], b_returns[mb_inds], b_masks[mb_inds],
                    is_learning=b_is_learning[mb_inds],
                    coef_retall=CLIP_COEF, coef_ent=ENT_COEF, coef_v=VF_COEF
                )
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                
        # Logs i Guardat
        mean_reward = np.mean(batch_rewards)
        if update % 50 == 0:
            eval_rev, eval_wr = evaluar_contra_random(agent, env_config, device)

        if update % 10 == 0:
            pbar.set_postfix({
                "Rew": f"{mean_reward:.4f}", 
                "V": f"{v_loss.item():.3f}", 
                "WR%": f"{eval_wr:.1f}"
            })
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([update, global_step, pg_loss.item(), v_loss.item(), ent_loss.item(), mean_reward, eval_wr, eval_rev])
                
        if update % 500 == 0:
            torch.save(net.state_dict(), save_dir / f"ppo_mlp_update_{update}.pt")
            
    # Finalització
    torch.save(net.state_dict(), save_dir / "best.pt")
    vec_env.close()

if __name__ == "__main__":
    main()
