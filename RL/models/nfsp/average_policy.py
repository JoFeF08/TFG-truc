import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.tools.obs_utils import flatten_obs

class AveragePolicyNet(nn.Module):
    """COS frozen (instància pròpia, mateixos pesos que PPO) + head MLP -> 24 logits.
    Arquitectura configurable: hidden=(256,256) per defecte (compatible F5)."""
    def __init__(self, pesos_cos: str, hidden=(256, 256), n_actions: int = 24, dropout: float = 0.0):
        super().__init__()
        self.cos = CosMultiInputSB3(observation_space=Box(low=-1, high=1, shape=(240,)),
                                    features_dim=256)
        self.cos.carregar_pesos_preentrenats(pesos_cos)
        self.cos.congelar_cos()
        layers, in_dim = [], 256
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, n_actions)]
        self.head = nn.Sequential(*layers)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.head(self.cos(obs_flat))   # logits (B, 24)


class SLAgent:
    """Adaptador amb `eval_step(state) -> (action, dict)`. Sampling estocàstic
    sobre logits emmascarats (legal actions). Mixed policy => no determinista."""
    use_raw = False

    def __init__(self, net: AveragePolicyNet, device, n_actions: int = 24,
                 deterministic: bool = False, seed: int = 0):
        self.net = net.to(device).eval()
        self.device = device
        self.n_actions = n_actions
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def eval_step(self, state):
        obs = flatten_obs(state['obs']).astype(np.float32)
        logits = self.net(torch.from_numpy(obs[None]).to(self.device)).cpu().numpy()[0]
        legal = list(state['legal_actions'].keys())
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[legal] = 0.0
        logits = logits + mask
        if self.deterministic:
            a = int(np.argmax(logits))
        else:
            # Gumbel-max trick o softmax
            logits_shifted = logits - logits.max()
            p = np.exp(logits_shifted)
            p /= p.sum()
            a = int(self._rng.choice(self.n_actions, p=p))
        return (a if a in legal else legal[0]), {}
