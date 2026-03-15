import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from RL.models.xarxa_truc import CosMultiInput

# Dimensions fixes
LATENT_DIM = 128
OBS_CARTES_SHAPE = (6, 4, 9)
OBS_CONTEXT_SIZE = 23

COS_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent.parent / "RL" / 
                       "entrenament" / "entrenamentEstatTruc" / "registres" / 
                       "13_03_26_a_les_1909" / "models" / "best_pesos_cos_truc.pth")

def construir_mlp(in_dim, layers, out_dim, final="none", use_bn=True):
    """Construeix un MLP clàssic (ReLU) amb Batch Normalization opcional"""
    dims = [in_dim] + layers
    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        if use_bn:
            net.append(nn.BatchNorm1d(dims[i + 1]))
        net.append(nn.ReLU())
    
    net.append(nn.Linear(dims[-1], out_dim))
    if final == "logsoftmax":
        net.append(nn.LogSoftmax(dim=-1))
    
    return nn.Sequential(*net)

class XarxaUnificada(nn.Module):
    """
    Xarxa end-to-end que combina el Feature Extractor (COS) amb un MLP
    per a ser usada directament amb agents de RLCard (DQN/NFSP).
    """
    def __init__(self, n_actions, mlp_layers, mode, weights=None, device=None, output="q", use_bn=True):
        super().__init__()
        
        self.mode = mode
        self.device = device or torch.device("cpu")

        # Cos
        self.cos = CosMultiInput()
        
        # Pre-calcular el tall del vector d'entrada
        self.split = OBS_CARTES_SHAPE[0] * OBS_CARTES_SHAPE[1] * OBS_CARTES_SHAPE[2]

        # Carregar pesos si no estem en mode 'scratch'
        if mode in ("frozen", "finetune"):
            w = weights or COS_WEIGHTS_PATH
            
            if w and os.path.exists(w):
                self.cos.load_state_dict(torch.load(w, map_location=self.device, weights_only=True))
                
                print(f"[XarxaUnificada] Pesos carregats: {os.path.basename(os.path.dirname(os.path.dirname(w)))}")
            
            else:
                print("[XarxaUnificada] Avís: No s'han trobat pesos, usant valors aleatoris")

        if mode == "frozen":
            for p in self.cos.parameters(): p.requires_grad = False
            self.cos.eval()

        # MLP superior
        if output == "policy":
            activacio = "logsoftmax"
        else:
            activacio = "none"

        self.mlp = construir_mlp(LATENT_DIM, mlp_layers, n_actions, activacio, use_bn=use_bn)

        for p in self.mlp.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        self.to(self.device)

    def forward(self, obs):
        # Assegurar que obs és un tensor a la GPU
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        elif obs.device != self.device:
            obs = obs.to(self.device)

        # Convertir l'array a format cartes + context
        # obs sol ser [Batch, Flat_Dim]
        cartes_f = obs[:, :self.split]
        context = obs[:, self.split:]
        
        cartes = cartes_f.view(-1, *OBS_CARTES_SHAPE)

        #cos
        if self.mode == "frozen":
            with torch.no_grad(): z = self.cos(cartes, context)
        else:
            z = self.cos(cartes, context)

        # Si el batch és 1 i estem en training, passem a eval temporalment per la BatchNorm
        if obs.shape[0] == 1:
            was_training = self.mlp.training
            self.mlp.eval()
            with torch.no_grad():
                out = self.mlp(z)
            if was_training:
                self.mlp.train()
            return out

        return self.mlp(z)

    def get_param_groups(self, lr_cos=1e-5, lr_mlp=5e-4):
        """Grups de paràmetres per usar LR diferenciat en fine-tuning"""
        return [
            {"params": self.cos.parameters(), "lr": lr_cos},
            {"params": self.mlp.parameters(), "lr": lr_mlp}
        ]

    def set_train_mode(self):
        self.train()
        if self.mode == "frozen":
            self.cos.eval()

    def set_eval_mode(self):
        self.eval()
