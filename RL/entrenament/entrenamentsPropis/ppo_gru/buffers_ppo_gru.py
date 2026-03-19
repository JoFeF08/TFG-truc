import torch

class RolloutBufferGRU:
    """
    Emmagatzema transicions pel PPO Seqüencial (GRU).
    Incorpora la capacitat de rebre resets per dividir les seqüències temporals en mans.
    """
    def __init__(self, num_steps, num_envs, state_shape, action_dim=21, device='cpu'):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        self.obs = torch.zeros((num_steps, num_envs, state_shape), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.resets = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.masks = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.bool).to(device)
        self.is_learning = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        
        self.step = 0
        
    def add(self, obs, action, logprob, reward, value, done, mask, resets, is_learning=1.0):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.masks[self.step] = mask
        self.resets[self.step] = resets
        self.is_learning[self.step] = is_learning
        self.step = (self.step + 1) % self.num_steps
        
    def get(self, advantages, returns):
        b_obs = self.obs.swapaxes(0, 1) # [num_envs, num_steps, F]
        b_actions = self.actions.swapaxes(0, 1)
        b_logprobs = self.logprobs.swapaxes(0, 1)
        b_advantages = advantages.swapaxes(0, 1)
        b_returns = returns.swapaxes(0, 1)
        b_masks = self.masks.swapaxes(0, 1)
        b_resets = self.resets.swapaxes(0, 1)
        b_is_learning = self.is_learning.swapaxes(0, 1)
        
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks, b_resets, b_is_learning
