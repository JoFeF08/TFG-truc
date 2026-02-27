import os
import csv
from pathlib import Path
import torch
import logging
from contextlib import redirect_stdout
from tqdm import tqdm

from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import set_seed, reorganize
from entorn import TrucEnv

# ─── Configuració ─────────────────────────────────────────────────────────────
VERBOSE_TRAINING = False

# Silenciar els loggers
if not VERBOSE_TRAINING:
    logging.basicConfig(level=logging.ERROR, force=True)
    logging.getLogger().setLevel(logging.ERROR) 
else:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().setLevel(logging.INFO)

# ─── Configuració ─────────────────────────────────────────────────────────────
SEED            = 42
NUM_EPISODES    = 50_000     # Episodis d'entrenament
EVALUATE_EVERY  = 1_000      # Cada quants episodis avaluem
EVALUATE_NUM    = 200        # Partides d'avaluació
SAVE_EVERY      = 10_000      # Cada quants episodis guardem el model

# Hiperparàmetres NFSP
HIDDEN_LAYERS       = [256, 256]
RL_LEARNING_RATE    = 1e-3
SL_LEARNING_RATE    = 1e-3   # Supervised learning (política mitjana)
BATCH_SIZE          = 256
RESERVOIR_SIZE      = 20_000  # Buffer per SL (reservoir sampling)
Q_REPLAY_SIZE       = 20_000  # Buffer per RL (Q-network)
Q_UPDATE_TARGET     = 300
ANTICIPATORY_PARAM  = 0.1    # η: probabilitat d'usar la política best-response

# Carpetes de sortida
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


def avaluar(env, num_partides: int) -> float:
    """Calcula el reward mig del jugador 0 sobre num_partides."""
    total = sum(env.run(is_training=False)[1][0] for _ in range(num_partides))
    return total / num_partides


def crear_agent_nfsp(env, device):
    """Crea un agent NFSP amb la configuració definida."""
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

    # Entorns d'entrenament i avaluació
    env_config = {
        'num_jugadors': 2, 
        'cartes_jugador': 3, 
        'seed': SEED,
        'verbose': VERBOSE_TRAINING
    }
    env      = TrucEnv(config=env_config)
    eval_env = TrucEnv(config=env_config)

    # Dos agents NFSP (self-play)
    agent_0 = crear_agent_nfsp(env, device)
    agent_1 = crear_agent_nfsp(env, device)

    env.set_agents([agent_0, agent_1])

    # Avaluació: agent_0 vs Random
    random_agent = RandomAgent(num_actions=env.num_actions)
    eval_env.set_agents([agent_0, random_agent])

    # Fitxer de log
    log_path = os.path.join(LOG_DIR, "nfsp_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episodi", "reward_mig_p0", "vic_pct_p0", "mode_p0", "mode_p1"])

    print(f"Iniciant entrenament NFSP ({NUM_EPISODES} episodis, self-play)...")
    devnull = open(os.devnull, 'w') 

    for episodi in tqdm(range(1, NUM_EPISODES + 1), desc="Entrenant NFSP", unit="ep"):
        # Un episodi = una partida completa de Truc
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        # Alimentar les transicions a cada agent
        if not VERBOSE_TRAINING:
            with redirect_stdout(devnull):
                for i, agent in enumerate([agent_0, agent_1]):
                    for ts in trajectories[i]:
                        agent.feed(ts)
        else:
            for i, agent in enumerate([agent_0, agent_1]):
                for ts in trajectories[i]:
                    agent.feed(ts)

        # Avaluació periòdica
        if episodi % EVALUATE_EVERY == 0:
            # NFSP té dos modes: 'average_policy' (estil Nash) i 'best_response'
            agent_0.sample_episode_policy()
            mode_p0 = "avg"  if agent_0._mode == "average_policy" else "br"
            agent_1.sample_episode_policy()
            mode_p1 = "avg"  if agent_1._mode == "average_policy" else "br"

            reward_mig  = avaluar(eval_env, EVALUATE_NUM)
            pct_victoria = round(100 * (reward_mig - (-1.3)) / (1.3 - (-1.3)), 1)

            tqdm.write(f"[Episodi {episodi:>6}]  reward mig p0: {reward_mig:.4f}  vic_pct: {pct_victoria:.1f}%  "
                       f"(mode p0={mode_p0}, p1={mode_p1})")

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([episodi, reward_mig, pct_victoria, mode_p0, mode_p1])

        # Guardar models periòdicament
        if episodi % SAVE_EVERY == 0:
            for pid, agent in enumerate([agent_0, agent_1]):
                path = os.path.join(MODEL_DIR, f"nfsp_truc_p{pid}_ep{episodi}.pt")
                torch.save({
                    'q_net':  agent._rl_agent.q_estimator.qnet.state_dict(),
                    'sl_net': agent.policy_network.state_dict(),
                }, path)
            tqdm.write(f"  → Models desats (episodi {episodi})")

    # Desar models finals
    for pid, agent in enumerate([agent_0, agent_1]):
        final_path = os.path.join(MODEL_DIR, f"nfsp_truc_p{pid}.pt")
        torch.save({
            'q_net':  agent._rl_agent.q_estimator.qnet.state_dict(),
            'sl_net': agent.policy_network.state_dict(),
        }, final_path)
        print(f"Model final p{pid}: {final_path}")

    print(f"\nEntrenament finalitzat.")
    print(f"Logs: {log_path}")

    devnull.close()

if __name__ == "__main__":
    main()