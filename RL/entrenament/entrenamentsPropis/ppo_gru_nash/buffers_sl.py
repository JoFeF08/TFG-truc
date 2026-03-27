import torch
import random
import numpy as np

class ReservoirBuffer:
    """Buffer que emmagatzema seqüències temporals seqüencials uniformement sense sesg temporal recent per entrenar la Average Policy SL."""
    def __init__(self, capacity, num_steps, state_shape, action_dim=21, device='cpu'):
        self.capacity = capacity
        self.num_steps = num_steps
        self.device = device
        
        self.buffer_obs = torch.zeros((capacity, num_steps, state_shape), dtype=torch.float32).cpu()
        self.buffer_actions = torch.zeros((capacity, num_steps), dtype=torch.long).cpu()
        self.buffer_resets = torch.zeros((capacity, num_steps), dtype=torch.float32).cpu()
        self.buffer_masks = torch.zeros((capacity, num_steps, action_dim), dtype=torch.bool).cpu()
        self.buffer_hiddens = torch.zeros((capacity, 1, 256), dtype=torch.float32).cpu()
        
        self.n_total = 0
        self.size = 0
        
    def add(self, b_obs, b_act, b_resets, b_masks, h_init):
        # b_obs: [num_envs, num_steps, F] etc.
        B = b_obs.shape[0]
        for i in range(B):
            self.n_total += 1
            if self.size < self.capacity:
                idx = self.size
                self.size += 1
            else:
                idx = random.randint(0, self.n_total - 1)
                
            if idx < self.capacity:
                self.buffer_obs[idx] = b_obs[i].cpu()
                self.buffer_actions[idx] = b_act[i].cpu()
                self.buffer_resets[idx] = b_resets[i].cpu()
                self.buffer_masks[idx] = b_masks[i].cpu()
                self.buffer_hiddens[idx] = h_init[:, i, :].cpu()
                
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        b_obs = self.buffer_obs[idxs].to(self.device).clone()
        b_act = self.buffer_actions[idxs].to(self.device).clone()
        b_resets = self.buffer_resets[idxs].to(self.device).clone()
        b_masks = self.buffer_masks[idxs].to(self.device).clone()
        
        h_init = self.buffer_hiddens[idxs] # [batch, 1, 256]
        h_init = h_init.transpose(0, 1).to(self.device).clone() # -> [1, batch, 256]
        
        return b_obs, b_act, b_resets, b_masks, h_init
