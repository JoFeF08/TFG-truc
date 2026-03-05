import sys
import os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import csv
import torch
import logging
from contextlib import redirect_stdout
from tqdm import tqdm

from rlcard.agents import DQNAgent, RandomAgent
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
SAVE_EVERY      = 20_000     # Cada quants episodis guardem checkpoint

# Hiperparàmetres DQN
LAYERS          = [256, 256]
LEARNING_RATE   = 5e-4
BATCH_SIZE      = 256
MEMORY_SIZE     = 100_000
UPDATE_TARGET   = 500        
EPSILON_MIN     = 0.05

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


class AgentCongelat:
    """
    Wrapper que redirigeix step() → eval_step().
    Així l'oponent actua greedy (sense exploració) i no s'entrena.
    """
    def __init__(self, agent):
        self.agent = agent
        self.use_raw = False  

    def step(self, state):
        action, _ = self.agent.eval_step(state)
        return action

    def eval_step(self, state):
        return self.agent.eval_step(state)


def crear_dqn_agent(env, device):
    return DQNAgent(
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


def copiar_pesos(src_agent, dst_agent):
    state_dict = src_agent.q_estimator.qnet.state_dict()
    dst_agent.q_estimator.qnet.load_state_dict(state_dict)


def avaluar(env, num_partides: int) -> float:
    total = sum(env.run(is_training=False)[1][0] for _ in range(num_partides))
    return total / num_partides


from models.adapters.feature_extractor import wrap_env_amb_cos, carregar_model_cos


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
    raw_env      = TrucEnv(config=env_config)
    raw_eval_env = TrucEnv(config=env_config)

    # Carreguem el COS actuant com a Feature Extractor
    cos_model = carregar_model_cos(device)
    
    # Embolillem els entorns perquè els agents rebin l'array de 128
    env = wrap_env_amb_cos(raw_env, cos_model, device)
    eval_env = wrap_env_amb_cos(raw_eval_env, cos_model, device)

    # Agent principal (entrena) i oponent congelat (millor versió històrica)
    dqn_agent = crear_dqn_agent(env, device)
    oponent_base = crear_dqn_agent(env, device)
    copiar_pesos(dqn_agent, oponent_base)
    oponent = AgentCongelat(oponent_base)

    # Avaluació vs Random (baseline consistent)
    random_agent = RandomAgent(num_actions=env.num_actions)

    env.set_agents([dqn_agent, oponent])
    eval_env.set_agents([dqn_agent, random_agent])

    # Fitxer de log
    log_path = os.path.join(LOG_DIR, "dqn_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episodi", "reward_mig", "victoires", "millor_historica", "lr"])

    print(f"Entrenant DQN ({NUM_EPISODES} episodis)...")
    
    devnull = open(os.devnull, 'w')
    best_reward = -float('inf')
    best_episodi = 0

    for episodi in tqdm(range(1, NUM_EPISODES + 1), desc="Entrenant DQN", unit="ep"):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        if not VERBOSE_TRAINING:
            with redirect_stdout(devnull):
                for ts in trajectories[0]:
                    dqn_agent.feed(ts)
        else:
            for ts in trajectories[0]:
                dqn_agent.feed(ts)

        # Learning Rate Scheduling
        if episodi in LR_DECAY_AT:
            for pg in dqn_agent.q_estimator.optimizer.param_groups:
                pg['lr'] *= LR_DECAY_FACTOR
            new_lr = dqn_agent.q_estimator.optimizer.param_groups[0]['lr']
            tqdm.write(f"LR reduït a {new_lr:.2e} (episodi {episodi})")

        # Avaluació periòdica contra Random
        if episodi % EVALUATE_EVERY == 0:
            reward_mig = avaluar(eval_env, EVALUATE_NUM)
            pct_victoria = round(100 * (reward_mig - (-1)) / (1 - (-1)), 1)
            current_lr = dqn_agent.q_estimator.optimizer.param_groups[0]['lr']

            es_millor = reward_mig > best_reward
            if es_millor:
                best_reward = reward_mig
                best_episodi = episodi
                
                # Guardar millor model
                best_path = os.path.join(MODEL_DIR, "dqn_truc_best.pt")
                torch.save(dqn_agent.q_estimator.qnet.state_dict(), best_path)
                copiar_pesos(dqn_agent, oponent_base)

            tqdm.write(
                f"[Episodi {episodi:>6}] reward mig: {reward_mig:.4f} | "
                f"vic: {pct_victoria:.1f}% | lr={current_lr:.2e} | "
                f"{'nou millor' if es_millor else f'millor: ep {best_episodi}'}"
            )

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([episodi, reward_mig, pct_victoria, best_reward, current_lr])

        # Guardar checkpoint periòdicament
        if episodi % SAVE_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"dqn_truc_ep{episodi}.pt")
            torch.save(dqn_agent.q_estimator.qnet.state_dict(), path)
            tqdm.write(f"  → Checkpoint desat a {path}")

    # Desar MILLOR model
    final_path = os.path.join(MODEL_DIR, "dqn_truc.pt")
    best_path = os.path.join(MODEL_DIR, "dqn_truc_best.pt")
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, final_path)
        print(f"\nModel final = millor model (episodi {best_episodi}, reward={best_reward:.4f})")
    else:
        torch.save(dqn_agent.q_estimator.qnet.state_dict(), final_path)
        print(f"\nModel final desat (últim): {final_path}")
    
    print(f"Logs: {log_path}")
    devnull.close()


if __name__ == "__main__":
    main()