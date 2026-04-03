import torch
import torch.nn as nn
from RL.models.core.feature_extractor import CosMultiInput
import os
from pathlib import Path

LATENT_DIM = 256
OBS_CARTES_SHAPE = (6, 4, 9)
SPLIT = OBS_CARTES_SHAPE[0] * OBS_CARTES_SHAPE[1] * OBS_CARTES_SHAPE[2] # 216
OBS_CONTEXT_SIZE = 23
N_ACCIONS_CARTES  = 3   # índexs 0-2: play_card_0/1/2
N_ACCIONS_APOSTES = 16  # índexs 3-18: apostes, vull/fora, senyes
HEAD_SIZE = 64

COS_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "entrenament" / "entrenamentEstatTruc" / "registres" / "22_03_26_a_les_0118" / "models" / "best_pesos_cos_truc.pth")

class PPOMlpNet(nn.Module):
    """
    Xarxa Actor-Critic base
    """
    def __init__(self, n_actions, hidden_size=256, ruta_weights=None, device='cpu', use_cos=True):
        super().__init__()
        self.device = device
        self.use_cos = use_cos

        if use_cos:
            self.cos = CosMultiInput()

            # Carreguem pesos
            w = ruta_weights or COS_WEIGHTS_PATH
            if w and os.path.exists(w):
                self.cos.load_state_dict(torch.load(w, map_location=self.device, weights_only=True))
                print(f"[PPOMlpNet] Pesos carregats al COS: {os.path.basename(os.path.dirname(os.path.dirname(w)))}")
            else:
                print("[PPOMlpNet] Avís: Cap pes pre-entrenat trobat, usant random.")

            # Congelem cos
            for p in self.cos.parameters():
                p.requires_grad = False
            self.cos.eval()
            self.cos_congelat = True
        else:
            self.cos = None
            self.cos_congelat = False
            obs_dim = SPLIT + OBS_CONTEXT_SIZE  # 239
            print(f"[PPOMlpNet] Mode sense COS: MLP directe ({obs_dim} → {hidden_size} → {hidden_size})")

        if use_cos:
            self.cap_cartes = nn.Sequential(
                nn.Linear(LATENT_DIM, HEAD_SIZE),
                nn.ReLU(),
                nn.Linear(HEAD_SIZE, N_ACCIONS_CARTES)
            )
            self.cap_apostes = nn.Sequential(
                nn.Linear(LATENT_DIM, HEAD_SIZE),
                nn.ReLU(),
                nn.Linear(HEAD_SIZE, N_ACCIONS_APOSTES)
            )
            self.critic = nn.Sequential(
                nn.Linear(LATENT_DIM, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            # Arquitectura equivalent a fase1: actor i critic independents sobre obs
            obs_dim = SPLIT + OBS_CONTEXT_SIZE  # 239
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

        self.to(device)

    def _prepare_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        elif obs.device != self.device:
            obs = obs.to(self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        cartes_f = obs[:, :SPLIT]
        context = obs[:, SPLIT:]
        cartes = cartes_f.view(-1, *OBS_CARTES_SHAPE)
        return cartes, context

    def get_features(self, obs):
        if not self.use_cos:
            raise RuntimeError("get_features no s'ha d'usar en mode no_cos — actor/critic reben obs directament.")

        cartes, context = self._prepare_obs(obs)

        # Modo freeze si toca
        if self.cos_congelat:
            with torch.no_grad():
                self.cos.eval()
                z = self.cos(cartes, context)
        else:
            z = self.cos(cartes, context)
        return z

    def unfreeze_cos(self):
        """Descongela el cos per a fine-tuning"""
        if not self.use_cos:
            return
        for p in self.cos.parameters():
            p.requires_grad = True
        self.cos_congelat = False
        self.cos.train()
        print("[PPOMlpNet] COS descongelat per a fine-tuning.")

    def get_param_groups(self, lr_cos=1e-5, lr_mlp=5e-4):
        """Retorna els paràmetres dividits per LR"""
        if not self.use_cos:
            return [{"params": self.parameters(), "lr": lr_mlp}]
        return [
            {"params": self.cos.parameters(),         "lr": lr_cos},
            {"params": self.cap_cartes.parameters(),  "lr": lr_mlp},
            {"params": self.cap_apostes.parameters(), "lr": lr_mlp},
            {"params": self.critic.parameters(),      "lr": lr_mlp},
        ]

    def forward(self, obs):
        if not self.use_cos:
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            elif obs.device != self.device:
                obs = obs.to(self.device)
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            return self.actor(obs), self.critic(obs)
        z = self.get_features(obs)
        logits_cartes  = self.cap_cartes(z)   # (batch, 3)
        logits_apostes = self.cap_apostes(z)  # (batch, 16)
        logits = torch.cat([logits_cartes, logits_apostes], dim=-1)  # (batch, 19)
        return logits, self.critic(z)
