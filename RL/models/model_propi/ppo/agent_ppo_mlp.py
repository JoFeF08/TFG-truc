import torch
import numpy as np
from torch.distributions import Categorical

class PPOMlpAgent:

    def __init__(self, net, num_actions, device='cpu'):
        self.net = net
        self.num_actions = num_actions
        self.device = device
        
    def step(self, obs_flat, legal_actions_list):
        """
        Escull una acció aleatòria basada en la distribució donada la política (Actor).
        """
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(obs_flat)
            
            # màscara
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, legal_actions_list] = True
            logits[~mask] = -1e9 # tapem les falses
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()

    def get_action_and_value(self, obs_flat, actions=None, masks=None):
        logits, value = self.net(obs_flat)
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e9)
            
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
            
        return actions, dist.log_prob(actions), dist.entropy(), value.squeeze(-1)

    # Funcio requerida quan RLCard
    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate([obs['obs_cartes'].flatten(), obs['obs_context']], axis=0).astype(np.float32)
        else:
            obs_flat = obs
            
        legal_actions = list(state['legal_actions'].keys())
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(obs_flat)
            
            # mascara
            mask = torch.zeros_like(logits, dtype=torch.bool)
            if mask.dim() == 1:
                mask[legal_actions] = True
            else:
                mask[0, legal_actions] = True
                
            logits[~mask] = -1e9
            
            # millor
            action = torch.argmax(logits, dim=-1).item()
            
        info = {}
        return action, info

    @property
    def use_raw(self):
        return False
