import os
import torch
import torch.nn as nn
from pathlib import Path
from models.xarxa_truc import CosMultiInput

# Dimensions fixes
LATENT_DIM = 128
OBS_CARTES_SHAPE = (6, 4, 9)
OBS_CONTEXT_SIZE = 17

def get_cos_weights():
    """Cerca darrers pesos entrenats del COS"""
    root = Path(__file__).resolve().parent.parent.parent
    reg_dir = root / "entrenament" / "entrenamentEstatTruc" / "registres"
    
    if not reg_dir.exists():
        return None

    # Ordenem per data
    folders = sorted(reg_dir.iterdir(), key=os.path.getmtime, reverse=True)
    for folder in folders:
        path = folder / "models" / "best_pesos_cos_truc.pth"
        if path.exists(): return str(path)
    
    return None

def construir_mlp(in_dim, layers, out_dim, final="none"):
    """Construeix un MLP clàssic (tanh) per a RLCard"""
    dims = [in_dim] + layers
    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        net.append(nn.Tanh())
    
    net.append(nn.Linear(dims[-1], out_dim))
    if final == "logsoftmax":
        net.append(nn.LogSoftmax(dim=-1))
    
    return nn.Sequential(*net)

class XarxaUnificada(nn.Module):
    """
    Xarxa end-to-end que combina el Feature Extractor (COS) amb un MLP
    per a ser usada directament amb agents de RLCard (DQN/NFSP).
    """
    def __init__(self, n_actions, mlp_layers, mode, weights=None, device=None, output="q"):
        super().__init__()
        
        self.mode = mode
        self.device = device or torch.device("cpu")

        # Cos
        self.cos = CosMultiInput()

        # Carregar pesos si no estem en mode 'scratch'
        if mode in ("frozen", "finetune"):
            w = weights or get_cos_weights()
            
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

        self.mlp = construir_mlp(LATENT_DIM, mlp_layers, n_actions, activacio)

        for p in self.mlp.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        self.to(self.device)

    def forward(self, obs):
        # Convertir l'array a format cartes + context
        split = OBS_CARTES_SHAPE[0] * OBS_CARTES_SHAPE[1] * OBS_CARTES_SHAPE[2]
        cartes_f = obs[:, :split]
        context = obs[:, split:]
        
        cartes = cartes_f.view(-1, *OBS_CARTES_SHAPE)

        #cos
        if self.mode == "frozen":
            with torch.no_grad(): z = self.cos(cartes, context)
        else:
            z = self.cos(cartes, context)

        #model
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
