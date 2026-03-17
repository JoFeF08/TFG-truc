"""
Mòdul de càlcul de pèrdues PPO (Proximal Policy Optimization) i GAE (Generalized Advantage Estimation).
"""

import torch

def calcular_gae(r, v_critic, done, ultim_v, ultim_done, gamma=0.995, lam=0.95):
    """
    Calcula l'Avantatge Generalitzat Estimat (GAE - Generalized Advantage Estimation).
    
    L'avantatge mesura quant millor és una acció concreta comparada amb la mitjana en un estat.
    GAE permet un equilibri entre biaix i variància mitjançant el paràmetre lambda.
    """
    avantatges = torch.zeros_like(r)
    ultim_gae_lam = 0
    num_passos = r.shape[0]
    
    for t in reversed(range(num_passos)):
        if t == num_passos - 1:
            next_no_done = 1.0 - ultim_done
            next_v = ultim_v
        else:
            next_no_done = 1.0 - done[t]
            next_v = v_critic[t + 1]
            
        # Delta: Error de pèrdua temporal (TD Error)
        # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = r[t] + gamma * next_v * next_no_done - v_critic[t]
        
        # Càlcul recursiu de GAE
        # A_t = delta_t + gamma * lambda * A_{t+1}
        avantatges[t] = ultim_gae_lam = delta + gamma * lam * next_no_done * ultim_gae_lam
        
    retorns = avantatges + v_critic
    return avantatges, retorns

def calcular_perdua_ppo(agent, obs, accions, log_probs_antics, avantatges, 
                        retorns, mascares, is_learning=None, coef_retall=0.2, coef_ent=0.01, coef_v=0.5):

    _, log_probs_nous, entropia, valors_nous = agent.get_action_and_value(obs, accions, mascares)
    
    return calcular_perdua_ppo_nucleu(
        log_probs_nous, log_probs_antics, entropia, 
        valors_nous, avantatges, retorns, 
        is_learning, coef_retall, coef_ent, coef_v
    )

def calcular_perdua_ppo_nucleu(log_probs_nous, log_probs_antics, entropia, valors_nous, avantatges, 
                                retorns, is_learning=None, coef_retall=0.2, coef_ent=0.01, coef_v=0.5):
    """
    Calcula la funció d'objectiu total de PPO:
    1. Pèrdua de la Política
    2. Pèrdua de Valor
    3. Pèrdua d'Entropia
    """
    
    #nova vs l'antiga
    log_raio = log_probs_nous - log_probs_antics
    raio = log_raio.exp()
    
    mitjana_adv = avantatges.mean()
    desviacio_adv = avantatges.std()
    avantatges_norm = (avantatges - mitjana_adv) / (desviacio_adv + 1e-8)

    # Pèrdua de la Política
    perdua_pg1 = -avantatges_norm * raio
    perdua_pg2 = -avantatges_norm * torch.clamp(raio, 1 - coef_retall, 1 + coef_retall)
    
    if is_learning is not None:
        perdua_pg = (torch.max(perdua_pg1, perdua_pg2) * is_learning).sum() / (is_learning.sum() + 1e-8)
        perdua_v = 0.5 * (((valors_nous - retorns) ** 2) * is_learning).sum() / (is_learning.sum() + 1e-8)
        perdua_ent = (entropia * is_learning).sum() / (is_learning.sum() + 1e-8)
    else:
        perdua_pg = torch.max(perdua_pg1, perdua_pg2).mean()
    
    # Pèrdua de Valor
        perdua_v = 0.5 * ((valors_nous - retorns) ** 2).mean()
    
    # Pèrdua d'Entropia
        perdua_ent = entropia.mean()
    
    # Pèrdua Total
    perdua_total = perdua_pg - coef_ent * perdua_ent + coef_v * perdua_v
    
    return perdua_total, perdua_pg, perdua_v, perdua_ent

