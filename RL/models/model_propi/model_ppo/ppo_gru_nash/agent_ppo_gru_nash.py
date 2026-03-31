import torch
import numpy as np
from torch.distributions import Categorical

class PPOGruNashAgent:
    def __init__(self, net, num_actions, num_envs=1, eta=0.1, device='cpu'):
        self.net = net
        self.num_actions = num_actions
        self.device = device
        self.num_envs = num_envs
        self.eta = eta
        
        self.hidden_states = torch.zeros(1, num_envs, self.net.gru.hidden_size, device=device)
        self.eval_hidden = torch.zeros(1, 1, self.net.gru.hidden_size, device=device)
        
    def reset_hidden(self, env_indices):
        self.hidden_states[:, env_indices, :] = 0.0
        
    def step(self, obs_flat, masks, use_rl=True):
        self.net.eval()
        with torch.no_grad():
            logits_rl, value, logits_sl, self.hidden_states = self.net(obs_flat, self.hidden_states)
            
            if not isinstance(use_rl, torch.Tensor):
                use_rl = torch.tensor(use_rl, device=self.device, dtype=torch.bool)
                
            logits_rl[~masks] = -1e9
            logits_sl[~masks] = -1e9
            
            logits_final = torch.where(use_rl.unsqueeze(-1), logits_rl, logits_sl)
            
            dist = Categorical(logits=logits_final)
            action = dist.sample()
            
            # Necessitem logprob de l'actor RL independentment de si ha escollit SL
            # Tot i que en PPO no ho podrem usar directament si la política de mostreig era SL!
            # Controlarem això al bucle d'entrenament, posant use_rl=True normalment.
            dist_rl = Categorical(logits=logits_rl)
            logprob_rl = dist_rl.log_prob(action)
            
            return action, logprob_rl, value.squeeze(-1)

    def get_action_and_value(self, obs_seq, actions, masks, resets, init_hidden):
        B, S, F = obs_seq.shape
        hidden = init_hidden
        logits_list = []
        values_list = []
        
        for t in range(S):
            m_reset = resets[:, t].unsqueeze(0).unsqueeze(-1)
            hidden = hidden * (1.0 - m_reset)
            
            logit_rl, val, _, hidden = self.net(obs_seq[:, t], hidden)
            logits_list.append(logit_rl)
            values_list.append(val)
            
        logits_rl = torch.stack(logits_list, dim=1) 
        value = torch.stack(values_list, dim=1)  
        
        if masks is not None:
            logits_rl = logits_rl.masked_fill(~masks, -1e9)
            
        dist = Categorical(logits=logits_rl)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return logprobs, entropy, value.squeeze(-1)

    def get_sl_logits(self, obs_seq, resets, init_hidden):
        B, S, F = obs_seq.shape
        hidden = init_hidden
        logits_sl_list = []
        
        for t in range(S):
            m_reset = resets[:, t].unsqueeze(0).unsqueeze(-1)
            hidden = hidden * (1.0 - m_reset)
            
            _, _, logit_sl, hidden = self.net(obs_seq[:, t], hidden)
            logits_sl_list.append(logit_sl)
            
        return torch.stack(logits_sl_list, dim=1)

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate([obs['obs_cartes'].flatten(), obs['obs_context']], axis=0).astype(np.float32)
        else:
            obs_flat = obs
            
        obs_tensor = torch.tensor(obs_flat, device=self.device).unsqueeze(0)
        
        legal_actions = list(state['legal_actions'].keys())
        self.net.eval()
        with torch.no_grad():
            _, _, logits_sl, self.eval_hidden = self.net(obs_tensor, self.eval_hidden)
            
            mask = torch.zeros_like(logits_sl, dtype=torch.bool)
            mask[0, legal_actions] = True
            logits_sl[~mask] = -1e9
            
            action = torch.argmax(logits_sl, dim=-1).item()
            
        return action, {}
        
    @property
    def use_raw(self):
        return False
