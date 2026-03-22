import sys
import os
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm, trange
import csv
from datetime import datetime

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from RL.entrenament.entrenamentsPropis.parallel_env import SubprocVecEnv
from RL.models.model_propi.ppo_gru.cap_ppo_gru import PPOGruNet
from RL.models.model_propi.ppo_gru.agent_ppo_gru import PPOGruAgent
from RL.entrenament.entrenamentsPropis.ppo_gru.buffers_ppo_gru import RolloutBufferGRU
from rlcard.agents import RandomAgent
from joc.entorn.env import TrucEnv
from RL.entrenament.entrenamentsPropis.ppo_loss import calcular_gae, calcular_perdua_ppo_nucleu
from RL.models.model_propi.ppo.cap_ppo_mlp import PPOMlpNet, SPLIT, OBS_CONTEXT_SIZE
from RL.models.model_propi.ppo.agent_ppo_mlp import PPOMlpAgent
from joc.entorn.cartes_accions import ACTION_LIST

# Hyperparams Constants
NUM_ENVS = 48
NUM_STEPS = 256
MINIBATCH_ENVS = 12       # 8→12: gradients més estables en TBPTT
UPDATE_EPOCHS = 5          # 7→5: menys epochs evita overfitting en xarxes recurrents
TOTAL_TIMESTEPS = 20_000_000  # 60M→20M: primer test post-COS
LR = 2e-4                  # 3e-4→2e-4: lleugerament menor per estabilitat GRU
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02            # 0.01→0.02: més exploració en joc de cartes amb moltes accions
VF_COEF = 0.5

# Fine-tune Constants
FINETUNE_LR_COS = 1e-5
UNFREEZE_FRACTION = 0.20   # 0.15→0.20: esperar més per descongelar COS

GOLDEN_PATH = Path(__file__).parent / "golden_ppo_gru.pt"

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


def evaluar_contra_random(agent, env_config, device, num_partides=50):
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


def evaluar_contra_golden(agent, golden_agent, env_config, num_partides=100):
    """Evalua l'agent contra el golden opponent fix (PPO simple)."""
    if golden_agent is None:
        return 0.0, 0.0

    eval_env = TrucEnv(env_config)
    eval_env.set_agents([agent, golden_agent])

    recompenses = []
    victories = 0

    for _ in range(num_partides):
        _, payoffs = eval_env.run(is_training=False)
        recompenses.append(payoffs[0])
        if payoffs[0] > 0:
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
    print(f"[{device.type.upper()}] Iniciant Entrenament PPO Seqüencial (GRU) - Mode: {args.mode.upper()}")
    
    env_config = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 24,
        'seed': 42
    }
    
    vec_env = SubprocVecEnv(NUM_ENVS, env_config)
    
    n_acc = len(ACTION_LIST)
    net = PPOGruNet(n_actions=n_acc, hidden_size=256, device=device)
    agent = PPOGruAgent(net, n_acc, num_envs=NUM_ENVS, device=device)
    
    # Pool d'Oponents
    pool_net = PPOGruNet(n_actions=n_acc, hidden_size=256, device=device)
    pool_net.eval()
    opponent_pool = []
    pool_hidden_states = torch.zeros(1, NUM_ENVS, 256).to(device)

    # entrenamants anteriors
    registres_dir = Path(__file__).parent / "registres"
    if registres_dir.exists():
        for run_dir in registres_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("ppo_gru_"):
                best_path = run_dir / "best.pt"
                if best_path.exists():
                    opponent_pool.append(best_path)
                    print(f"[Pool Init] Carregat: {run_dir.name} -> {best_path}")
    print(f"[Pool Init] Total oponents carregats: {len(opponent_pool)}")

    # golden opponent
    golden_agent = None
    if GOLDEN_PATH.exists():
        golden_net = PPOMlpNet(n_actions=n_acc, hidden_size=256, device=device)
        golden_net.load_state_dict(torch.load(GOLDEN_PATH, map_location=device, weights_only=True))
        golden_agent = PPOMlpAgent(golden_net, n_acc, device=device)
        print(f"[Golden] Carregat: {GOLDEN_PATH.name}")
    else:
        print(f"[Golden] No trobat: {GOLDEN_PATH.name}. Avaluació només contra random.")

    # 20% pool
    NUM_POOL_ENVS = int(NUM_ENVS * 0.2)
    POOL_FREQUENCY = 300  # 500→300: renovar oponents amb més freqüència
    pool_player_ids = {}
    for i in range(NUM_ENVS - NUM_POOL_ENVS, NUM_ENVS):
        pool_player_ids[i] = i % 2
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)
    
    state_shape = SPLIT + OBS_CONTEXT_SIZE 
    buffer = RolloutBufferGRU(NUM_STEPS, NUM_ENVS, state_shape, action_dim=n_acc, device=device)
    
    results = vec_env.reset_all()
    current_states = [res[0] for res in results]
    last_comptador_ma = [s['raw_obs']['comptador_ma'] for s in current_states]
    
    global_step = 0
    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    
    timestamp = datetime.now().strftime("%dd_%mm_%H%Mh")
    save_dir = Path(__file__).parent / "registres" / f"ppo_gru_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training_log.csv"
    
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "pg_loss", "v_loss", "ent_loss", "reward_mean", "eval_wr", "eval_reward", "eval_wr_golden"])

    best_eval_wr = -1.0
    pbar = trange(1, num_updates + 1, desc="Actualitzacions")
    eval_wr, eval_rev, eval_wr_golden = 0.0, 0.0, 0.0
    for update in pbar:
        batch_rewards = []
        init_hidden = agent.hidden_states.clone()
        
        # afegim a la pool
        if update > 10 and update % POOL_FREQUENCY == 0:
            ckpt_path = save_dir / f"checkpoint_update_{update}.pt"
            torch.save(net.state_dict(), ckpt_path)
            opponent_pool.append(ckpt_path)
            
            random_pool_path = random.choice(opponent_pool)
            pool_net.load_state_dict(torch.load(random_pool_path, map_location=device, weights_only=True))
            pool_net = pool_net.to(device)  # Assegurar que està en GPU
            print(f"\n[Pool] Oponent carregat: {random_pool_path.name}")
        
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            
            obs_tensor, masks_tensor = extract_obs(current_states)
            obs_tensor = obs_tensor.to(device)
            masks_tensor = masks_tensor.to(device)
            
            comptador_ma = [s['raw_obs']['comptador_ma'] for s in current_states]
            
            resets_step = []
            indices_reset = []
            for i in range(NUM_ENVS):
                did_reset = (comptador_ma[i] != last_comptador_ma[i])
                resets_step.append(did_reset)
                if did_reset:
                    indices_reset.append(i)
                    
            if len(indices_reset) > 0:
                
                agent.reset_hidden(indices_reset)
                pool_hidden_states[:, indices_reset, :] = 0.0
                
            resets_tensor = torch.tensor(resets_step, dtype=torch.float32).to(device)
            
            action, logprob, value, _ = agent.step(obs_tensor, masks_tensor)
            

            is_learning_step = torch.ones(NUM_ENVS, device=device)
            active_players = [s['raw_obs']['id_jugador'] for s in current_states]

            # Sobreescriure accions de la Pool
            if len(opponent_pool) > 0:
                for i, pid in pool_player_ids.items():
                    if active_players[i] == pid:
                        with torch.no_grad():
                            p_logits, _, p_h = pool_net(obs_tensor[i:i+1], pool_hidden_states[:, i:i+1, :])
                            p_action = torch.distributions.Categorical(logits=p_logits.masked_fill(~masks_tensor[i:i+1], -1e9)).sample()
                        action[i] = p_action
                        pool_hidden_states[:, i:i+1, :] = p_h
                        is_learning_step[i] = 0.0
            
            actions_np = action.cpu().numpy()
            
            next_states_players, rewards_list, dones_list = vec_env.step(actions_np)
            
            step_rewards = []
            for i in range(NUM_ENVS):
                step_rewards.append(rewards_list[i][active_players[i]])
                
            step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones_list, dtype=torch.float32).to(device)
            
            buffer.add(obs_tensor, action, logprob, step_rewards_tensor, value, dones_tensor, masks_tensor, resets_tensor, is_learning_step)
            
            current_states = [sp[0] for sp in next_states_players]
            last_comptador_ma = list(comptador_ma)
            batch_rewards.extend(step_rewards)
            
        # Update GRU + PPO
        obs_tensor, _ = extract_obs(current_states)
        obs_tensor = obs_tensor.to(device)
        
        # Obtenir el next_value sense avançar hidden recurrent
        agent.net.eval()
        with torch.no_grad():
            _, last_value, _ = agent.net(obs_tensor, agent.hidden_states)
            last_value = last_value.squeeze(-1)
            
        avantatges, retorns = calcular_gae(
            buffer.rewards, buffer.values, buffer.dones, 
            last_value, torch.zeros_like(last_value), 
            GAMMA, GAE_LAMBDA
        )
        
        b_obs, b_actions, b_logprobs, b_advs, b_rets, b_masks, b_resets, b_is_learning = buffer.get(avantatges, retorns)
        b_inds = np.arange(NUM_ENVS)
        
        # fine-tune si toca
        if args.mode == "finetune" and not has_unfrozen and global_step >= unfreeze_step:
            net.unfreeze_cos()
            params = net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=LR)
            optimizer = optim.Adam(params, lr=LR, eps=1e-5)
            has_unfrozen = True
            print(f"[Fine-tune] COS descongelat al step {global_step}. Optimizer actualitzat.")

        agent.net.train()
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS, MINIBATCH_ENVS):
                end = start + MINIBATCH_ENVS
                mb_inds = b_inds[start:end]
                
                h_mb = init_hidden[:, mb_inds, :].contiguous()
                
                newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], b_masks[mb_inds], b_resets[mb_inds], h_mb
                )
                
                mb_logprobs = b_logprobs[mb_inds].reshape(-1)
                mb_advs = b_advs[mb_inds].reshape(-1)
                mb_rets = b_rets[mb_inds].reshape(-1)
                newlogprob = newlogprob.reshape(-1)
                entropy = entropy.reshape(-1)
                newvalue = newvalue.reshape(-1)
                
                loss, pg_loss, v_loss, ent_loss = calcular_perdua_ppo_nucleu(
                    newlogprob, mb_logprobs, entropy, newvalue, mb_advs, mb_rets,
                    is_learning=b_is_learning[mb_inds].reshape(-1),
                    coef_retall=CLIP_COEF, coef_ent=ENT_COEF, coef_v=VF_COEF
                )
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                
        # logs
        mean_reward = np.mean(batch_rewards)
        if update % 50 == 0:
            eval_rev, eval_wr = evaluar_contra_random(agent, env_config, device)
            _, eval_wr_golden = evaluar_contra_golden(agent, golden_agent, env_config)

            metric = (0.3 * eval_wr + 0.7 * eval_wr_golden) if golden_agent is not None else eval_wr
            if metric > best_eval_wr:
                best_eval_wr = metric
                torch.save(net.state_dict(), save_dir / "best.pt")
                tqdm.write(f" -> Nou millor: random={eval_wr:.1f}% golden={eval_wr_golden:.1f}%! Model guardat.")

        if update % 10 == 0:
            pbar.set_postfix({
                "Rew": f"{mean_reward:.4f}",
                "V": f"{v_loss.item():.3f}",
                "WR%": f"{eval_wr:.1f}",
                "GWR%": f"{eval_wr_golden:.1f}"
            })
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([update, global_step, pg_loss.item(), v_loss.item(), ent_loss.item(), mean_reward, eval_wr, eval_rev, eval_wr_golden])
                
        if update % 500 == 0:
            torch.save(net.state_dict(), save_dir / f"ppo_gru_update_{update}.pt")
            
    vec_env.close()

    # Netejar checkpoints intermedis, només queda best.pt i el log
    for f in save_dir.glob("checkpoint_update_*.pt"):
        f.unlink()
    for f in save_dir.glob("ppo_gru_update_*.pt"):
        f.unlink()
    print(f"[Cleanup] Checkpoints intermedis eliminats. Només queda best.pt")

if __name__ == "__main__":
    main()
