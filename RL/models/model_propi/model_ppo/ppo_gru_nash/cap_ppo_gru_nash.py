import torch
import torch.nn as nn
from RL.models.core.feature_extractor import CosMultiInput
from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import SPLIT, OBS_CARTES_SHAPE, LATENT_DIM, COS_WEIGHTS_PATH
import os

class PPOGruNashNet(nn.Module):
    def __init__(self, n_actions, hidden_size=256, sl_hidden=128, ruta_weights=None, device='cpu'):
        super().__init__()
        self.device = device
        self.cos = CosMultiInput()
        
        w = ruta_weights or COS_WEIGHTS_PATH
        if w and os.path.exists(w):
            self.cos.load_state_dict(torch.load(w, map_location=self.device, weights_only=True))
            print(f"[PPOGruNashNet] Pesos carregats al COS: {os.path.basename(os.path.dirname(os.path.dirname(w)))}")
            
        for p in self.cos.parameters():
            p.requires_grad = False
        self.cos.eval()
        self.cos_congelat = True
        
        self.gru = nn.GRU(LATENT_DIM, hidden_size, batch_first=True)
        # Cap RL (PPO) - ActorCritic
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Cap SL (Nash Average Policy)
        self.sl_policy = nn.Sequential(
            nn.Linear(hidden_size, sl_hidden),
            nn.ReLU(),
            nn.Linear(sl_hidden, n_actions)
        )
        
        self.to(device)

    def _prepare_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        elif obs.device != self.device:
            obs = obs.to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        carte_features = obs[:, :SPLIT].view(-1, *OBS_CARTES_SHAPE)
        context = obs[:, SPLIT:]
        return carte_features, context
        
    def get_features(self, obs):
        cartes, context = self._prepare_obs(obs)
        if self.cos_congelat:
            with torch.no_grad():
                self.cos.eval()
                z = self.cos(cartes, context)
        else:
            z = self.cos(cartes, context)
        return z

    def unfreeze_cos(self):
        """Descongela el cos per a fine-tuning"""
        for p in self.cos.parameters():
            p.requires_grad = True
        self.cos_congelat = False
        self.cos.train()
        print("[PPOGruNashNet] COS descongelat per a fine-tuning.")

    def get_param_groups(self, lr_cos=1e-5, lr_mlp=5e-4):
        """Retorna els paràmetres dividits per LR (només els de RL)"""
        return [
            {"params": self.cos.parameters(), "lr": lr_cos},
            {"params": self.gru.parameters(), "lr": lr_mlp},
            {"params": self.actor.parameters(), "lr": lr_mlp},
            {"params": self.critic.parameters(), "lr": lr_mlp}
        ]

    def forward(self, obs, hidden_state):
        is_sequence = len(obs.shape) == 3 
        
        if is_sequence:
            B, S, F = obs.shape
            obs_flat = obs.view(B * S, F)
            z = self.get_features(obs_flat)
            z = z.view(B, S, LATENT_DIM)
        else:
            z = self.get_features(obs)
            z = z.unsqueeze(1) 
            
        gru_out, new_hidden = self.gru(z, hidden_state)
        
        if not is_sequence:
            gru_out = gru_out.squeeze(1)
            
        logits_rl = self.actor(gru_out)
        value = self.critic(gru_out)
        
        # Desconnectem el graf computacional perquè SL aprengui sol del features post-RNN passius
        logits_sl = self.sl_policy(gru_out.detach())
        
        return logits_rl, value, logits_sl, new_hidden
