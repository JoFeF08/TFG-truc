import torch

class RolloutBuffer:
    """
    Emmagatzema transicions pel PPO.
    Guardem dades en BATCH paral·lel: (num_steps, num_envs, data_size).
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
        self.masks = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.bool).to(device)
        self.is_learning = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        
        self.step = 0
        
    def add(self, obs, action, logprob, reward, value, done, mask, is_learning=1.0):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.masks[self.step] = mask
        self.is_learning[self.step] = is_learning
        self.step = (self.step + 1) % self.num_steps
        
    def get(self, advantages, returns):
        """
        Retorna tots els elements del buffer aplanats
        """
        b_obs = self.obs.reshape((-1, self.obs.shape[-1]))
        b_actions = self.actions.reshape((-1,))
        b_logprobs = self.logprobs.reshape((-1,))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_masks = self.masks.reshape((-1, self.masks.shape[-1]))
        b_is_learning = self.is_learning.reshape((-1,))
        
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks, b_is_learning
