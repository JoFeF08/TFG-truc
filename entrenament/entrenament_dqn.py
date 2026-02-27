import os
import csv
from pathlib import Path
import torch
import logging

# ─── Configuració ─────────────────────────────────────────────────────────────
VERBOSE_TRAINING = False

# Silenciar els bombardeigs "INFO - Step X, rl-loss: Y" de RLCard
if not VERBOSE_TRAINING:
    logging.getLogger("rlcard").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
else:
    logging.getLogger("rlcard").setLevel(logging.INFO)

from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import set_seed, reorganize
from entorn import TrucEnv

# ─── Configuració ─────────────────────────────────────────────────────────────
SEED            = 42
NUM_EPISODES    = 50_000     # Episodis d'entrenament
EVALUATE_EVERY  = 1_000      # Cada quants episodis avaluem
EVALUATE_NUM    = 200        # Partides d'avaluació
SAVE_EVERY      = 10_000     # Cada quants episodis guardem el model

# Hiperparàmetres DQN
LAYERS          = [256, 256]
LEARNING_RATE   = 5e-4
BATCH_SIZE      = 256
MEMORY_SIZE     = 20_000
UPDATE_TARGET   = 300        # Cada quants passos actualitzem la xarxa target
EPSILON_MIN     = 0.05

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


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositiu: {device}")

    # Entorn
    env_config = {
        'num_jugadors': 2, 
        'cartes_jugador': 3, 
        'seed': SEED,
        'verbose': VERBOSE_TRAINING
    }
    env      = TrucEnv(config=env_config)
    eval_env = TrucEnv(config=env_config)

    # Agent DQN (jugador 0) i Random com a oponent (jugador 1)
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=LAYERS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        replay_memory_size=MEMORY_SIZE,
        replay_memory_init_size=BATCH_SIZE,
        update_target_estimator_every=UPDATE_TARGET,
        epsilon_decay_steps=NUM_EPISODES,
        epsilon_end=EPSILON_MIN,
        device=device,
    )
    random_agent = RandomAgent(num_actions=env.num_actions)

    env.set_agents([dqn_agent, random_agent])
    eval_env.set_agents([dqn_agent, random_agent])

    # Fitxer de log
    log_path = os.path.join(LOG_DIR, "dqn_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episodi", "reward_mig", "victoires"])

    print(f"Iniciant entrenament DQN ({NUM_EPISODES} episodis)...")

    for episodi in range(1, NUM_EPISODES + 1):
        # Un episodi = una partida completa de Truc
        trajectories, payoffs = env.run(is_training=True)
        # Reorganitzar a 5-tuples (state, action, reward, next_state, done)
        trajectories = reorganize(trajectories, payoffs)

        # Alimentar les transicions a l'agent
        for ts in trajectories[0]:
            dqn_agent.feed(ts)

        # Avaluació periòdica
        if episodi % EVALUATE_EVERY == 0:
            reward_mig = avaluar(eval_env, EVALUATE_NUM)
            # reward > 1.0 vol dir victòria (recompensa mínima de guanyar = 1.0)
            pct_victoria = round(100 * (reward_mig - (-1.3)) / (1.3 - (-1.3)), 1)

            print(f"[Episodi {episodi:>6}]  reward mig: {reward_mig:.4f}  vic%: {pct_victoria:.1f}%")

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([episodi, reward_mig, pct_victoria])

        # Guardar model periòdicament
        if episodi % SAVE_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"dqn_truc_ep{episodi}.pt")
            torch.save(dqn_agent.q_estimator.qnet.state_dict(), path)
            print(f"  → Model desat a {path}")

    # Desar model final
    final_path = os.path.join(MODEL_DIR, "dqn_truc.pt")
    torch.save(dqn_agent.q_estimator.qnet.state_dict(), final_path)
    print(f"\nEntrenament finalitzat. Model final: {final_path}")
    print(f"Logs: {log_path}")


if __name__ == "__main__":
    main()
