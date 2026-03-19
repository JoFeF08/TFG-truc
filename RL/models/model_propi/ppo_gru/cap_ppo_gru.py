import torch
import torch.nn as nn
from RL.models.core.feature_extractor import CosMultiInput
from RL.models.model_propi.ppo.cap_ppo_mlp import SPLIT, OBS_CARTES_SHAPE, LATENT_DIM, COS_WEIGHTS_PATH
import os

class PPOGruNet(nn.Module):
    """
    Xarxa Actor-Critic amb memòria GRU.
    """
    def __init__(self, n_actions, hidden_size=256, ruta_weights=None, device='cpu'):
        super().__init__()
        self.device = device
        self.cos = CosMultiInput()
        
        w = ruta_weights or COS_WEIGHTS_PATH
        if w and os.path.exists(w):
            self.cos.load_state_dict(torch.load(w, map_location=self.device, weights_only=True))
            print(f"[PPOGruNet] Pesos carregats al COS: {os.path.basename(os.path.dirname(os.path.dirname(w)))}")
        else:
            print("[PPOGruNet] Avís: Cap pes pre-entrenat trobat, usant random.")
            
        for p in self.cos.parameters():
            p.requires_grad = False
        self.cos.eval()
        self.cos_congelat = True
        
        # Unitat Recurrent
        self.gru = nn.GRU(LATENT_DIM, hidden_size, batch_first=True)
        
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
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
        print("[PPOGruNet] COS descongelat per a fine-tuning.")

    def get_param_groups(self, lr_cos=1e-5, lr_mlp=5e-4):
        """Retorna els paràmetres dividits per LR"""
        return [
            {"params": self.cos.parameters(), "lr": lr_cos},
            {"params": self.gru.parameters(), "lr": lr_mlp},
            {"params": self.actor.parameters(), "lr": lr_mlp},
            {"params": self.critic.parameters(), "lr": lr_mlp}
        ]

    def forward(self, obs, hidden_state):
        """
        Calcula una passada "forward" d'un batch d'observacions flat [B, F].
        """
        z = self.get_features(obs)
        z = z.unsqueeze(1) # [B, 1, LATENT]
        
        gru_out, new_hidden = self.gru(z, hidden_state)
        gru_out = gru_out.squeeze(1) # [B, Hidden]
            
        logits = self.actor(gru_out)
        value = self.critic(gru_out)
        
        return logits, value, new_hidden
