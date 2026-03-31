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
from datetime import datetime

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn_ma.parallel_env_ma import SubprocVecEnvMa
from RL.models.model_propi.model_ppo.ppo_gru.cap_ppo_gru import PPOGruNet
from RL.models.model_propi.model_ppo.ppo_gru.agent_ppo_gru import PPOGruAgent
from RL.entrenament.entrenamentsPropis.ppo_gru.buffers_ppo_gru import RolloutBufferGRU
from rlcard.agents import RandomAgent
from RL.models.model_propi.agent_regles import AgentRegles
from RL.models.model_propi.model_ppo.ppo_loss import calcular_gae, calcular_perdua_ppo_nucleu
from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import SPLIT, OBS_CONTEXT_SIZE
from joc.entorn.cartes_accions import ACTION_LIST
from RL.models.model_propi.model_ppo.ppo_utils import extract_obs, evaluar_contra_random, evaluar_contra_regles

# Hyperparams
NUM_ENVS = 48
NUM_STEPS = 256
MINIBATCH_ENVS = 12
UPDATE_EPOCHS = 5
TOTAL_TIMESTEPS = 20_000_000
LR = 2e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5

FINETUNE_LR_COS = 1e-5
UNFREEZE_FRACTION = 0.20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "frozen", "finetune"], default="frozen")
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--load_model", type=str, default=None, help="Ruta al model .pt a carregar")
    parser.add_argument("--save_dir", type=str, default=None, help="Directori on guardar els resultats")
    args = parser.parse_args()

    total_timesteps = args.total_timesteps
    unfreeze_step = int(total_timesteps * UNFREEZE_FRACTION)
    has_unfrozen = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Entrenament PPO-GRU per MANS - Mode: {args.mode.upper()}")

    env_config_ma = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 999,   # mai acaba per score, sempre per ma
        'seed': 42
    }
    env_config_eval = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 24,
        'seed': 42
    }

    vec_env = SubprocVecEnvMa(NUM_ENVS, env_config_ma)

    n_acc = len(ACTION_LIST)
    net = PPOGruNet(n_actions=n_acc, hidden_size=256, device=device)
    
    if args.load_model and os.path.exists(args.load_model):
        net.load_state_dict(torch.load(args.load_model, map_location=device, weights_only=True))
        print(f"[Init] Model carregat correctament des de: {args.load_model}")

    agent = PPOGruAgent(net, n_acc, num_envs=NUM_ENVS, device=device)

    regles_agent_eval = AgentRegles(num_actions=n_acc, seed=123)
    regles_agent_train = AgentRegles(num_actions=n_acc, seed=456)
    random_agent_train = RandomAgent(num_actions=n_acc)
    print(f"[Regles/Random] Agents inicialitzats.")

    n_envs_random = int(NUM_ENVS * 0.05)
    n_envs_regles = int(NUM_ENVS * 0.65)

    fixed_opponents = {}
    current_idx = 0
    for i in range(current_idx, current_idx + n_envs_random):
        fixed_opponents[i] = {'type': 'random', 'pid': i % 2}
    current_idx += n_envs_random
    for i in range(current_idx, current_idx + n_envs_regles):
        fixed_opponents[i] = {'type': 'regles', 'pid': i % 2}

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, eps=1e-5)

    state_shape = SPLIT + OBS_CONTEXT_SIZE
    buffer = RolloutBufferGRU(NUM_STEPS, NUM_ENVS, state_shape, action_dim=n_acc, device=device)

    results = vec_env.reset_all()
    current_states = [res[0] for res in results]
    last_dones = [False] * NUM_ENVS   # done de l'anterior pas (per reset GRU)

    global_step = 0
    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%dd_%mm_%H%Mh")
        save_dir = Path(__file__).parent / "registres" / f"ppo_gru_ma_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "training_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "pg_loss", "v_loss", "ent_loss", "reward_mean", "eval_wr", "eval_reward", "eval_wr_regles"])

    best_eval_wr = -1.0
    pbar = trange(1, num_updates + 1, desc="Actualitzacions")
    eval_wr, eval_rev, eval_wr_regles = 0.0, 0.0, 0.0

    for update in pbar:
        batch_rewards = []
        init_hidden = agent.hidden_states.clone()

        for step in range(NUM_STEPS):
            global_step += NUM_ENVS

            obs_tensor, masks_tensor = extract_obs(current_states)
            obs_tensor = obs_tensor.to(device)
            masks_tensor = masks_tensor.to(device)

            # Quan done=True al pas anterior, els current_states ja son de la nova ma (post-reset).
            # Cal fer reset del GRU hidden state per als entorns que van acabar.
            resets_step = list(last_dones)
            indices_reset = [i for i, d in enumerate(last_dones) if d]
            if indices_reset:
                agent.reset_hidden(indices_reset)

            resets_tensor = torch.tensor(resets_step, dtype=torch.float32).to(device)

            action, logprob, value, _ = agent.step(obs_tensor, masks_tensor)

            is_learning_step = torch.ones(NUM_ENVS, device=device)
            active_players = [s['raw_obs']['id_jugador'] for s in current_states]

            # Sobreescriure accions (Random, Regles, Pool)
            for i, opp_info in fixed_opponents.items():
                if active_players[i] == opp_info['pid']:
                    if opp_info['type'] == 'random':
                        action_idx, _ = random_agent_train.eval_step(current_states[i])
                        action[i] = action_idx
                        is_learning_step[i] = 0.0
                    elif opp_info['type'] == 'regles':
                        action_idx, _ = regles_agent_train.eval_step(current_states[i])
                        action[i] = action_idx
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
            last_dones = list(dones_list)
            batch_rewards.extend(step_rewards)

        # Update GRU + PPO
        obs_tensor, _ = extract_obs(current_states)
        obs_tensor = obs_tensor.to(device)

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

        if args.mode == "finetune" and not has_unfrozen and global_step >= unfreeze_step:
            net.unfreeze_cos()
            params = net.get_param_groups(lr_cos=FINETUNE_LR_COS, lr_mlp=LR)
            optimizer = optim.Adam(params, lr=LR, eps=1e-5)
            has_unfrozen = True
            print(f"[Fine-tune] COS descongelat al step {global_step}.")

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

        mean_reward = np.mean(batch_rewards)
        if update % 50 == 0:
            eval_rev, eval_wr = evaluar_contra_random(agent, env_config_eval, device)
            _, eval_wr_regles = evaluar_contra_regles(agent, regles_agent_eval, env_config_eval)

            metric = 0.25 * eval_wr + 0.75 * eval_wr_regles
            if metric > best_eval_wr:
                best_eval_wr = metric
                torch.save(net.state_dict(), save_dir / "best.pt")
                tqdm.write(f" -> Nou millor: random={eval_wr:.1f}% regles={eval_wr_regles:.1f}%! Model guardat.")

        if update % 10 == 0:
            pbar.set_postfix({
                "Rew": f"{mean_reward:.4f}",
                "V": f"{v_loss.item():.3f}",
                "WR%": f"{eval_wr:.1f}",
                "RWR%": f"{eval_wr_regles:.1f}"
            })
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([update, global_step, pg_loss.item(), v_loss.item(), ent_loss.item(), mean_reward, eval_wr, eval_rev, eval_wr_regles])

        if update % 500 == 0:
            torch.save(net.state_dict(), save_dir / f"ppo_gru_update_{update}.pt")

    vec_env.close()

    for f in save_dir.glob("ppo_gru_update_*.pt"):
        f.unlink()
    print(f"[Cleanup] Checkpoints intermedis eliminats. Nomes queda best.pt")


if __name__ == "__main__":
    main()
