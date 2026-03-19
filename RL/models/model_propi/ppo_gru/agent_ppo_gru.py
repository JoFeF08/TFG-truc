import torch
import numpy as np
from torch.distributions import Categorical

class PPOGruAgent:
    """
    L'Agent PPO equipat amd GRU que manté i controla conscientment els hidden states dels entorns actius.
    """
    def __init__(self, net, num_actions, num_envs=1, device='cpu'):
        self.net = net
        self.num_actions = num_actions
        self.device = device
        self.num_envs = num_envs
        
        # [1, Num_Envs, Hidden_Size]
        self.hidden_states = torch.zeros(1, num_envs, self.net.gru.hidden_size, device=device)
        self.eval_hidden = torch.zeros(1, 1, self.net.gru.hidden_size, device=device)
        
    def reset_hidden(self, env_indices):
        self.hidden_states[:, env_indices, :] = 0.0
        
    def step(self, obs_flat, masks):
        self.net.eval()
        with torch.no_grad():
            logits, value, self.hidden_states = self.net(obs_flat, self.hidden_states)
            logits[~masks] = -1e9
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            return action, dist.log_prob(action), value.squeeze(-1), self.hidden_states.clone()

    def get_action_and_value(self, obs_seq, actions, masks, resets, init_hidden):
        """
        Passada batch (TBPTT) completa: Recorre el temps de pas en pas per recalcular correctament
        el graf computacional tot respectant els límits entre episodis (resets).
        """
        B, S, F = obs_seq.shape
        hidden = init_hidden
        logits_list = []
        values_list = []
        
        for t in range(S):
            # Maskejar a zero l'estat ocult si l'entorn s'havia reiniciat just previ a aquest step
            # resets[b, t] indica si en temps t aquell batch `b` acabava de fer reset.
            m_reset = resets[:, t].unsqueeze(0).unsqueeze(-1) # [1, B, 1]
            hidden = hidden * (1.0 - m_reset)
            
            logit, val, hidden = self.net(obs_seq[:, t], hidden)
            logits_list.append(logit)
            values_list.append(val)
            
        logits = torch.stack(logits_list, dim=1)  # [B, S, Num_Action]
        value = torch.stack(values_list, dim=1)   # [B, S]
        
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e9)
            
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return logprobs, entropy, value.squeeze(-1)

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
            logits, _, self.eval_hidden = self.net(obs_tensor, self.eval_hidden)
            
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, legal_actions] = True
            logits[~mask] = -1e9
            
            action = torch.argmax(logits, dim=-1).item()
            
        return action, {}
        
    @property
    def use_raw(self):
        return False
