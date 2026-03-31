import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import csv
from datetime import datetime

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn.parallel_env import SubprocVecEnv
from RL.models.model_propi.model_ppo.ppo_gru_nash.cap_ppo_gru_nash import PPOGruNashNet
from RL.models.model_propi.model_ppo.ppo_gru_nash.agent_ppo_gru_nash import PPOGruNashAgent
from RL.entrenament.entrenamentsPropis.ppo_gru.buffers_ppo_gru import RolloutBufferGRU
from RL.entrenament.entrenamentsPropis.ppo_gru_nash.buffers_sl import ReservoirBuffer
from RL.models.model_propi.model_ppo.ppo.ppo_loss import calcular_gae, calcular_perdua_ppo_nucleu
from rlcard.agents import RandomAgent
from joc.entorn.env import TrucEnv
from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import PPOMlpNet, SPLIT, OBS_CONTEXT_SIZE
from RL.models.model_propi.model_ppo.ppo.agent_ppo_mlp import PPOMlpAgent
from joc.entorn.cartes_accions import ACTION_LIST
from RL.models.model_propi.agent_regles import AgentRegles
from RL.models.model_propi.model_ppo.ppo.ppo_utils import extract_obs, evaluar_contra_random, evaluar_contra_regles

# Hyperparams Constants
NUM_ENVS = 48  # Mantenim original per evitar duplicar càrrega
NUM_STEPS = 256
MINIBATCH_ENVS = 8
UPDATE_EPOCHS = 7  # Reduït de 10 a 7 per menys iteracions
TOTAL_TIMESTEPS = 24_000_000
LR_RL = 3e-4
LR_SL = 1e-3
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
ETA_START = 0.3
ETA_END = 0.1
RESERVOIR_CAPACITY = 2000

# Fine-tune Constants
FINETUNE_LR_COS = 1e-5
UNFREEZE_FRACTION = 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"], default="frozen")
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    total_timesteps = args.total_timesteps
    unfreeze_step = int(total_timesteps * UNFREEZE_FRACTION)
    has_unfrozen = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Iniciant Entrenament PPO+GRU+Nash (FSP) - Mode: {args.mode.upper()}")
    
    env_config = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 24,
        'seed': 42
    }
    
    vec_env = SubprocVecEnv(NUM_ENVS, env_config)

    n_acc = len(ACTION_LIST)

    regles_agent_eval = AgentRegles(num_actions=n_acc, seed=123)
    regles_agent_train = AgentRegles(num_actions=n_acc, seed=456)
    random_agent_train = RandomAgent(num_actions=n_acc)
    print(f"[Regles/Random] Agents inicialitzats.")

    n_envs_random = int(NUM_ENVS * 0.10)
    n_envs_regles = int(NUM_ENVS * 0.20)

    fixed_opponents = {}
    current_idx = 0
    for i in range(current_idx, current_idx + n_envs_random):
        fixed_opponents[i] = {'type': 'random', 'pid': i % 2}
    current_idx += n_envs_random
    for i in range(current_idx, current_idx + n_envs_regles):
        fixed_opponents[i] = {'type': 'regles', 'pid': i % 2}

    net = PPOGruNashNet(n_actions=n_acc, hidden_size=256, sl_hidden=128, device=device)
    agent = PPOGruNashAgent(net, n_acc, num_envs=NUM_ENVS, eta=ETA_START, device=device)

    # Optimizer for PPO (Actor+Critic+GRU)
    opt_rl = optim.Adam([
        {'params': net.gru.parameters()},
        {'params': net.actor.parameters()},
        {'params': net.critic.parameters()}
    ], lr=LR_RL, eps=1e-5)
    
    # Optimizer for SL (Average Policy ONLY)
    opt_sl = optim.Adam(net.sl_policy.parameters(), lr=LR_SL, weight_decay=1e-5)
    
    state_shape = SPLIT + OBS_CONTEXT_SIZE 
    buffer_rl = RolloutBufferGRU(NUM_STEPS, NUM_ENVS, state_shape, action_dim=n_acc, device=device)
    
    # Buffer SL guarda episodis sencers de num_steps
    buffer_sl = ReservoirBuffer(RESERVOIR_CAPACITY, NUM_STEPS, state_shape, action_dim=n_acc, device=device)
    
    results = vec_env.reset_all()
    current_states = [res[0] for res in results]
    last_comptador_ma = [s['raw_obs']['comptador_ma'] for s in current_states]
    
    global_step = 0
    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    
    timestamp = datetime.now().strftime("%dd_%mm_%H%Mh")
    save_dir = Path(__file__).parent / "registres" / f"ppo_nash_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training_log.csv"
    
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "eta", "pg_loss", "v_loss", "sl_loss", "ent_loss", "reward_mean", "eval_wr", "eval_reward", "eval_wr_regles"])

    # Estructura per guardar quines accions eren RL
    # Shape = [num_steps, num_envs]
    use_rl_history = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float32, device=device)

    best_eval_wr = -1.0
    pbar = trange(1, num_updates + 1, desc="Actualitzacions")
    eval_wr, eval_rev, eval_wr_regles = 0.0, 0.0, 0.0
    for update in pbar:
        # Annealing d'ETA
        if update > num_updates * 0.33:
            agent.eta = ETA_END

        batch_rewards = []
        init_hidden = agent.hidden_states.clone()

        # 1. Rollout Sequencial
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
                if did_reset: indices_reset.append(i)
                    
            if len(indices_reset) > 0: agent.reset_hidden(indices_reset)
            resets_tensor = torch.tensor(resets_step, dtype=torch.float32).to(device)
            
            use_rl = torch.rand(NUM_ENVS, device=device) < agent.eta
            use_rl_history[step] = use_rl.float()
            
            is_learning_step = torch.ones(NUM_ENVS, device=device)
            action, logprob, value = agent.step(obs_tensor, masks_tensor, use_rl)
            
            active_players = [s['raw_obs']['id_jugador'] for s in current_states]
            
            # Sobreescriure accions per Lliga
            for i, opp_info in fixed_opponents.items():
                if active_players[i] == opp_info['pid']:
                    if opp_info['type'] == 'random':
                        action_idx, _ = random_agent_train.eval_step(current_states[i])
                        action[i] = action_idx
                        is_learning_step[i] = 0.0
                        use_rl_history[step, i] = 0.0
                    elif opp_info['type'] == 'regles':
                        action_idx, _ = regles_agent_train.eval_step(current_states[i])
                        action[i] = action_idx
                        is_learning_step[i] = 0.0
                        use_rl_history[step, i] = 0.0
            
            actions_np = action.cpu().numpy()
            
            next_states_players, rewards_list, dones_list = vec_env.step(actions_np)
            
            step_rewards = []
            for i in range(NUM_ENVS): step_rewards.append(rewards_list[i][active_players[i]])
            step_rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones_list, dtype=torch.float32).to(device)
            
            buffer_rl.add(obs_tensor, action, logprob, step_rewards_tensor, value, dones_tensor, masks_tensor, resets_tensor, is_learning_step)
            
            current_states = [sp[0] for sp in next_states_players]
            last_comptador_ma = list(comptador_ma)
            batch_rewards.extend(step_rewards)
            
        # Al final, posem aquesta trajectòria al Reservoir
        b_obs, b_actions, b_logprobs, _, _, b_masks, b_resets, _ = buffer_rl.get(buffer_rl.rewards, buffer_rl.rewards) # Fake GAE per obtenir els Tensors rotats
        buffer_sl.add(b_obs, b_actions, b_resets, b_masks, init_hidden)
        
        
        # 2. Update GRU + PPO
        obs_tensor, _ = extract_obs(current_states)
        obs_tensor = obs_tensor.to(device)
        
        agent.net.eval()
        with torch.no_grad():
            _, last_value, _, _ = agent.net(obs_tensor, agent.hidden_states)
            last_value = last_value.squeeze(-1)
            
        avantatges, retorns = calcular_gae(
            buffer_rl.rewards, buffer_rl.values, buffer_rl.dones, 
            last_value, torch.zeros_like(last_value), 
            GAMMA, GAE_LAMBDA
        )
        
        b_obs, b_actions, b_logprobs, b_advs, b_rets, b_masks, b_resets, b_is_learning = buffer_rl.get(avantatges, retorns)
        b_use_rl = use_rl_history.swapaxes(0, 1) # [num_envs, num_steps]
        
        b_inds = np.arange(NUM_ENVS)
        
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
                
                newvalue = newvalue.reshape(-1)
                
                # 2.2 Unfreeze per Fine-tune si toca
                if args.mode == "finetune" and not has_unfrozen and global_step >= unfreeze_step:
                    net.unfreeze_cos()
                    params = net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=LR_RL)
                    opt_rl = optim.Adam(params, lr=LR_RL, eps=1e-5)
                    has_unfrozen = True
                    print(f"[Fine-tune] COS descongelat al step {global_step}. Optimizer RL actualitzat.")

                loss, pg_loss, v_loss, ent_loss = calcular_perdua_ppo_nucleu(
                    newlogprob, mb_logprobs, entropy, newvalue, mb_advs, mb_rets,
                    is_learning=b_is_learning[mb_inds].reshape(-1),
                    coef_retall=CLIP_COEF, coef_ent=ENT_COEF, coef_v=VF_COEF
                )
                
                opt_rl.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt_rl.step()
                
        # 3. Update SL (Nash Fictitious Play)
        # Agafem un batch del reservoir (per ex. tamany MINIBATCH_ENVS) si n'hi ha prou
        if buffer_sl.size > MINIBATCH_ENVS:
            b_obs_sl, b_act_sl, b_resets_sl, b_masks_sl, h_init_sl = buffer_sl.sample(MINIBATCH_ENVS)
            
            # Passada forward només per la xarxa SL i congelant GRU!
            # Hem prohibit que el backward del cap_sl propagui al GRU?
            # Si ho fem detach() explícitament al GRU és més segur. En `get_sl_logits` net() ho computa normal.
            # No obstant, opt_sl NOMÉS té els paràmetres de net.sl_policy!
            # Així que el .backward() no moure pesos del GRU de tota manera, però consumeix memòria / gradients al gru.
            
            agent.net.train()
            logits_sl = agent.get_sl_logits(b_obs_sl, b_resets_sl, h_init_sl)
            
            # Traiem loss: no cal utilitzar la mascara use_rl de l'històric si assumim que SL vol aprendre
            # una mitjana de TOTS els passos (tant els fets per RL com per SL prèviament).
            # Tot i això, per fidelitat màxima a NFSP, s'aprèn l'històric tal qual s'ha guardat.
            
            loss_sl_raw = F.cross_entropy(logits_sl.reshape(-1, len(ACTION_LIST)), b_act_sl.reshape(-1), reduction='none')
            loss_sl = loss_sl_raw.mean()
            
            opt_sl.zero_grad()
            loss_sl.backward()
            opt_sl.step()
        else:
            loss_sl = torch.tensor(0.0)
            
        # 4. Logs
        mean_reward = np.mean(batch_rewards)
        if update % 50 == 0:
            eval_rev, eval_wr = evaluar_contra_random(agent, env_config, device)
            _, eval_wr_regles = evaluar_contra_regles(agent, regles_agent_eval, env_config)

            metric = 0.25 * eval_wr + 0.75 * eval_wr_regles
            if metric > best_eval_wr:
                best_eval_wr = metric
                torch.save(net.state_dict(), save_dir / "best.pt")
                tqdm.write(f" -> Nou millor: random={eval_wr:.1f}% regles={eval_wr_regles:.1f}%! Model guardat.")

        if update % 10 == 0:
            pbar.set_postfix({
                "Rew": f"{mean_reward:.4f}",
                "SL": f"{loss_sl.item():.3f}",
                "WR%": f"{eval_wr:.1f}",
                "RWR%": f"{eval_wr_regles:.1f}"
            })
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([update, global_step, agent.eta, pg_loss.item(), v_loss.item(), loss_sl.item(), ent_loss.item(), mean_reward, eval_wr, eval_rev, eval_wr_regles])
                
        if update % 500 == 0:
            torch.save(net.state_dict(), save_dir / f"ppo_nash_update_{update}.pt")
            
    vec_env.close()

if __name__ == "__main__":
    main()
