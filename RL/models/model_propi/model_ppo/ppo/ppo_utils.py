import gc
import numpy as np
import torch
from rlcard.agents import RandomAgent
from joc.entorn.env import TrucEnv
from joc.entorn.cartes_accions import ACTION_LIST

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


def evaluar_contra_random(agent, env_config, _device=None, num_partides=50):
    eval_env = TrucEnv(env_config)
    eval_opponent = RandomAgent(num_actions=len(ACTION_LIST))
    recompenses = []
    victories = 0
    for i in range(num_partides):
        if i % 2 == 0:
            eval_env.set_agents([agent, eval_opponent])
            agent_id = 0
        else:
            eval_env.set_agents([eval_opponent, agent])
            agent_id = 1
        _, payoffs = eval_env.run(is_training=False)
        recompensa = payoffs[agent_id]
        recompenses.append(recompensa)
        if recompensa > 0:
            victories += 1
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(recompenses), (victories / num_partides) * 100


def evaluar_contra_regles(agent, regles_agent, env_config, num_partides=100):
    eval_env = TrucEnv(env_config)
    recompenses = []
    victories = 0
    for i in range(num_partides):
        if i % 2 == 0:
            eval_env.set_agents([agent, regles_agent])
            agent_id = 0
        else:
            eval_env.set_agents([regles_agent, agent])
            agent_id = 1
        _, payoffs = eval_env.run(is_training=False)
        recompensa = payoffs[agent_id]
        recompenses.append(recompensa)
        if recompensa > 0:
            victories += 1
    del eval_env
    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(recompenses), (victories / num_partides) * 100
