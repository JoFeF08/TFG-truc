import sys
import os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
import logging
from contextlib import redirect_stdout
from tqdm import tqdm

from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import set_seed, reorganize
from entorn import TrucEnv

VERBOSE_TRAINING = False

# Silenciar els loggers
if not VERBOSE_TRAINING:
    logging.basicConfig(level=logging.ERROR, force=True)
    logging.getLogger().setLevel(logging.ERROR) 
else:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO)

# Configuració
SEED            = 42
NUM_EPISODES    = 200_000
EVALUATE_EVERY  = 2_000
EVALUATE_NUM    = 500
SAVE_EVERY      = 20_000     # Cada quants episodis guardem el model

# Hiperparàmetres NFSP
HIDDEN_LAYERS       = [256, 256]
RL_LEARNING_RATE    = 1e-3
SL_LEARNING_RATE    = 1e-3   # Supervised learning (política mitjana)
BATCH_SIZE          = 256
RESERVOIR_SIZE      = 100_000 # Buffer per SL
Q_REPLAY_SIZE       = 100_000 # Buffer per RL
Q_UPDATE_TARGET     = 300
ANTICIPATORY_PARAM  = 0.3    # η

# Learning Rate Scheduling
LR_DECAY_AT     = [NUM_EPISODES // 4, NUM_EPISODES // 2, 3 * NUM_EPISODES // 4]
LR_DECAY_FACTOR = 0.5

# Carpetes de sortida
BASE_DIR  = Path(__file__).resolve().parent
TIMESTAMP = datetime.now().strftime("%d_%m_%y_a_les_%H%M")
RUN_DIR   = BASE_DIR / "registres" / TIMESTAMP
MODEL_DIR = RUN_DIR / "models"
LOG_DIR   = RUN_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def avaluar(env, num_partides: int) -> float:
    total = sum(env.run(is_training=False)[1][0] for _ in range(num_partides))
    return total / num_partides


def crear_agent_nfsp(env, device):
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=HIDDEN_LAYERS,
        q_mlp_layers=HIDDEN_LAYERS,
        rl_learning_rate=RL_LEARNING_RATE,
        sl_learning_rate=SL_LEARNING_RATE,
        batch_size=BATCH_SIZE,
        reservoir_buffer_capacity=RESERVOIR_SIZE,
        q_replay_memory_size=Q_REPLAY_SIZE,
        q_update_target_estimator_every=Q_UPDATE_TARGET,
        anticipatory_param=ANTICIPATORY_PARAM,
        device=device,
    )


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositiu: {device}")

    # Entorns
    env_config = {
        'num_jugadors': 2, 
        'cartes_jugador': 3, 
        'puntuacio_final': 12,
        'seed': SEED,
        'verbose': VERBOSE_TRAINING
    }
    env      = TrucEnv(config=env_config)
    eval_env = TrucEnv(config=env_config)

    # Dos agents NFSP (self-play)
    agent_0 = crear_agent_nfsp(env, device)
    agent_1 = crear_agent_nfsp(env, device)

    env.set_agents([agent_0, agent_1])

    # Avaluació
    random_agent = RandomAgent(num_actions=env.num_actions)
    eval_env.set_agents([agent_0, random_agent])

    # Fitxer de log
    log_path = os.path.join(LOG_DIR, "nfsp_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episodi", "reward_mig_p0", "vic_pct_p0", "mode_p0", "mode_p1", "millor_historica"])

    print(f"Entrenant NFSP ({NUM_EPISODES} episodis)...")
    devnull = open(os.devnull, 'w')
    best_reward = -float('inf')
    best_episodi = 0

    for episodi in tqdm(range(1, NUM_EPISODES + 1), desc="Entrenant NFSP", unit="ep"):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        if not VERBOSE_TRAINING:
            with redirect_stdout(devnull):
                for i, agent in enumerate([agent_0, agent_1]):
                    for ts in trajectories[i]:
                        agent.feed(ts)
        else:
            for i, agent in enumerate([agent_0, agent_1]):
                for ts in trajectories[i]:
                    agent.feed(ts)

        # Learning Rate Scheduling
        if episodi in LR_DECAY_AT:
            for ag in [agent_0, agent_1]:
                for pg in ag._rl_agent.q_estimator.optimizer.param_groups:
                    pg['lr'] *= LR_DECAY_FACTOR
                for pg in ag.policy_network_optimizer.param_groups:
                    pg['lr'] *= LR_DECAY_FACTOR
            rl_lr = agent_0._rl_agent.q_estimator.optimizer.param_groups[0]['lr']
            sl_lr = agent_0.policy_network_optimizer.param_groups[0]['lr']
            tqdm.write(f"LR reduït: RL={rl_lr:.2e}, SL={sl_lr:.2e} (episodi {episodi})")

        # Avaluació periòdica
        if episodi % EVALUATE_EVERY == 0:
            # NFSP té dos modes: 'average_policy' (estil Nash) i 'best_response'
            agent_0.sample_episode_policy()
            mode_p0 = "avg"  if agent_0._mode == "average_policy" else "br"
            agent_1.sample_episode_policy()
            mode_p1 = "avg"  if agent_1._mode == "average_policy" else "br"

            reward_mig  = avaluar(eval_env, EVALUATE_NUM)
            pct_victoria = round(100 * (reward_mig - (-1)) / (1 - (-1)), 1)

            es_millor = reward_mig > best_reward
            if es_millor:
                best_reward = reward_mig
                best_episodi = episodi
                # Guardar millor model de cada agent
                for pid, agent in enumerate([agent_0, agent_1]):
                    best_path = os.path.join(MODEL_DIR, f"nfsp_truc_p{pid}_best.pt")
                    torch.save({
                        'q_net':  agent._rl_agent.q_estimator.qnet.state_dict(),
                        'sl_net': agent.policy_network.state_dict(),
                    }, best_path)

            tqdm.write(
                f"[Episodi {episodi:>6}] reward mig p0: {reward_mig:.4f} | vic_pct: {pct_victoria:.1f}% | "
                f"mode p0={mode_p0}, p1={mode_p1} | "
                f"{'nou millor' if es_millor else f'millor: ep {best_episodi}'}"
            )

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([episodi, reward_mig, pct_victoria, mode_p0, mode_p1, best_reward])

        # Guardar models
        if episodi % SAVE_EVERY == 0:
            for pid, agent in enumerate([agent_0, agent_1]):
                path = os.path.join(MODEL_DIR, f"nfsp_truc_p{pid}_ep{episodi}.pt")
                torch.save({
                    'q_net':  agent._rl_agent.q_estimator.qnet.state_dict(),
                    'sl_net': agent.policy_network.state_dict(),
                }, path)
            tqdm.write(f"  → Models desats (episodi {episodi})")

    # Desar millors models
    import shutil
    for pid in range(2):
        best_candidate = os.path.join(MODEL_DIR, f"nfsp_truc_p{pid}_best.pt")
        if not os.path.exists(best_candidate):
            agent = [agent_0, agent_1][pid]
            torch.save({
                'q_net':  agent._rl_agent.q_estimator.qnet.state_dict(),
                'sl_net': agent.policy_network.state_dict(),
            }, best_candidate)

    # Enfrontament p0 vs p1: el guanyador és el model final
    print("\nEnfrontament final: p0_best vs p1_best")

    NUM_PLAYOFF = 500

    def carregar_nfsp_best(path, env, device):
        agent = crear_agent_nfsp(env, device)
        ck = torch.load(path, map_location=device, weights_only=True)
        agent._rl_agent.q_estimator.qnet.load_state_dict(ck['q_net'])
        agent.policy_network.load_state_dict(ck['sl_net'])
        return agent

    p0_best = carregar_nfsp_best(os.path.join(MODEL_DIR, "nfsp_truc_p0_best.pt"), env, device)
    p1_best = carregar_nfsp_best(os.path.join(MODEL_DIR, "nfsp_truc_p1_best.pt"), env, device)

    # Ronda 1: p0 com J0, p1 com J1
    playoff_env = TrucEnv(config=env_config)
    playoff_env.set_agents([p0_best, p1_best])
    wins_p0_r1, wins_p1_r1 = 0, 0
    for _ in range(NUM_PLAYOFF):
        _, payoffs = playoff_env.run(is_training=False)
        if payoffs[0] > payoffs[1]:
            wins_p0_r1 += 1
        else:
            wins_p1_r1 += 1

    # Ronda 2: p1 com J0, p0 com J1
    playoff_env.set_agents([p1_best, p0_best])
    wins_p1_r2, wins_p0_r2 = 0, 0
    for _ in range(NUM_PLAYOFF):
        _, payoffs = playoff_env.run(is_training=False)
        if payoffs[0] > payoffs[1]:
            wins_p1_r2 += 1
        else:
            wins_p0_r2 += 1

    total_p0 = wins_p0_r1 + wins_p0_r2
    total_p1 = wins_p1_r1 + wins_p1_r2
    total = 2 * NUM_PLAYOFF

    print(f"  p0: {total_p0}/{total} ({100*total_p0/total:.1f}%)")
    print(f"  p1: {total_p1}/{total} ({100*total_p1/total:.1f}%)")

    # Guardar el guanyador com a model final únic
    final_path = os.path.join(MODEL_DIR, "nfsp_truc.pt")
    if total_p0 >= total_p1:
        guanyador = "p0"
        shutil.copy2(os.path.join(MODEL_DIR, "nfsp_truc_p0_best.pt"), final_path)
    else:
        guanyador = "p1"
        shutil.copy2(os.path.join(MODEL_DIR, "nfsp_truc_p1_best.pt"), final_path)

    print(f"\nGuanya {guanyador} → model desat com a nfsp_truc.pt")
    print(f"Entrenament complet (millor ep={best_episodi}, reward={best_reward:.4f})")
    print(f"Logs: {log_path}")

    devnull.close()

if __name__ == "__main__":
    main()