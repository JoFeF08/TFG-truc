import torch
import numpy as np
import os
from pathlib import Path

def wrap_env_amb_cos(env, pre_trained_cos, device):
    """
    Monkey-patch de TrucEnv perquè el mètode intern `_extract_state` 
    passi pel model `cos` preentrenat, modificant `obs` i modificant `state_shape`.
    """
    original_extract_state = env._extract_state
    
    def _extract_state_patched(self, state):
        # Primer extraiem el diccionari original
        extracted = original_extract_state(state)
        obs_cartes = extracted['obs']['obs_cartes']
        obs_context = extracted['obs']['obs_context']
        
        t_cartes = torch.tensor(obs_cartes, dtype=torch.float32).unsqueeze(0).to(device)
        t_context = torch.tensor(obs_context, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = pre_trained_cos(t_cartes, t_context)
        feat_final = latent.squeeze(0).cpu().numpy()
        
        extracted['obs'] = feat_final
        return extracted

    import types
    env._extract_state = types.MethodType(_extract_state_patched, env)
    env.state_shape = [[128] for _ in range(env.num_jugadors)]
    
    return env


def carregar_model_cos(device):
    from models.xarxa_truc import CosMultiInput
    cos_model = CosMultiInput().to(device)
    cos_model.eval()
    
    directori_registres = Path(__file__).resolve().parent.parent.parent / "entrenament" / "entrenamentEstatTruc" / "registres"
    if directori_registres.exists():
        ultim_registre = sorted(directori_registres.iterdir(), key=os.path.getmtime, reverse=True)
        for reg in ultim_registre:
            ruta_pesos = reg / "models" / "best_pesos_cos_truc.pth"
            if ruta_pesos.exists():
                print(f"[PRE-ENTRENAMENT] Carregant extractor de característiques a l'agent: {ruta_pesos}")
                try:
                    cos_model.load_state_dict(torch.load(ruta_pesos, map_location=device))
                    return cos_model
                except Exception as e:
                    print(f"Error carregant {ruta_pesos}: {e}")
    print("[PRE-ENTRENAMENT] No s'han trobat pesos per al COS. Es farà servir un COS inicialitzat a l'atzar.")
    return cos_model
