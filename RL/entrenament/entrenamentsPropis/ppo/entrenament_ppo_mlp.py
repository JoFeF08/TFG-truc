import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv

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
from RL.entrenament.entrenamentsPropis.ppo_loss import calcular_gae, calcular_perdua_ppo

# Hyperparams Constants
NUM_ENVS = 16
NUM_STEPS = 128
MINIBATCH_SIZE = 256
UPDATE_EPOCHS = 4
TOTAL_TIMESTEPS = 3_000_000
LR = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

def extract_obs(states_list):
    b_obs = []
    b_masks = []
    
    for state in states_list:
        raw_obs = state['obs']
        flat = np.concatenate([raw_obs['obs_cartes'].flatten(), raw_obs['obs_context']], axis=0)
        b_obs.append(flat)
        
        mask = np.zeros(21, dtype=bool)
        legal = list(state['legal_actions'].keys()) if hasattr(state['legal_actions'], 'keys') else state['legal_actions']
        mask[legal] = True
        b_masks.append(mask)

    return torch.tensor(np.array(b_obs), dtype=torch.float32), torch.tensor(np.array(b_masks), dtype=torch.bool)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Iniciant Entrenament PPO Base (MLP) - Self-Play Unificat")
    
    env_config = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 24,
        'seed': 42
    }
    
    vec_env = SubprocVecEnv(NUM_ENVS, env_config)
    
    net = PPOMlpNet(n_actions=21, hidden_size=256, device=device)
    agent = PPOMlpAgent(net, 21, device=device)
    
    # cos congelat
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)
    
    state_shape = SPLIT + OBS_CONTEXT_SIZE 
    buffer = RolloutBuffer(NUM_STEPS, NUM_ENVS, state_shape, action_dim=21, device=device)
    
    # Inicialitzar entorns
    results = vec_env.reset_all()
    current_states = [res[0] for res in results]
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    
    save_dir = Path(__file__).parent / "registres"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training_log.csv"
    
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "pg_loss", "v_loss", "ent_loss", "reward_mean"])

    for update in range(1, num_updates + 1):
        batch_rewards = []
        
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            
            obs_tensor, masks_tensor = extract_obs(current_states)
            obs_tensor = obs_tensor.to(device)
            masks_tensor = masks_tensor.to(device)
            
            net.eval()
            with torch.no_grad():
                logits, value = net(obs_tensor)
                logits = logits.masked_fill(~masks_tensor, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            
            actions_np = action.cpu().numpy()
            active_players = [s['raw_obs']['id_jugador'] for s in current_states]
            
            # Executar step
            next_states_players, rewards_list, dones_list = vec_env.step(actions_np)
            
            step_rewards = []
            for i in range(NUM_ENVS):
                step_rewards.append(rewards_list[i][active_players[i]])
                
            step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones_list, dtype=torch.float32).to(device)
            
            buffer.add(obs_tensor, action, logprob, step_rewards_tensor, value.squeeze(-1), dones_tensor, masks_tensor)
            
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
        
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks = buffer.get(avantatges, retorns)
        b_inds = np.arange(NUM_ENVS * NUM_STEPS)
        
        net.train()
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds])
                
                loss, pg_loss, v_loss, ent_loss = calcular_perdua_ppo(
                    agent, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], 
                    b_advantages[mb_inds], b_returns[mb_inds], b_masks[mb_inds],
                    CLIP_COEF, ENT_COEF, VF_COEF
                )
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                
        # Logs i Guardat
        mean_reward = np.mean(batch_rewards)
        if update % 10 == 0:
            print(f"Update: {update}/{num_updates} | Step: {global_step} | Reward: {mean_reward:.4f} | VLoss: {v_loss.item():.4f}")
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([update, global_step, pg_loss.item(), v_loss.item(), ent_loss.item(), mean_reward])
                
        if update % 500 == 0:
            torch.save(net.state_dict(), save_dir / f"ppo_mlp_update_{update}.pt")
            
    # Finalització
    torch.save(net.state_dict(), save_dir / "best.pt")
    vec_env.close()

if __name__ == "__main__":
    main()
