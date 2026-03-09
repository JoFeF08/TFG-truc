import torch
import numpy as np
import os
import types
from pathlib import Path

def wrap_env_aplanat(env):
    """
    Monkey-patch de l'entorn TrucEnv perquè el mètode `_extract_state` 
    retorni una observació aplanada (233 dimensions), adequada per a 
    models unificats que ja integren el Feature Extractor (COS).
    """
    original_extract_state = env._extract_state

    def _extract_state_patched(self, state):
        extracted = original_extract_state(state)
        obs = extracted['obs']
        if isinstance(obs, dict):
            # Concatenem cartes (aplanades: 216) i context (17): Total 233
            flat = np.concatenate([
                obs['obs_cartes'].flatten(),
                obs['obs_context'],
            ], axis=0).astype(np.float32)
            extracted['obs'] = flat
        return extracted

    env._extract_state = types.MethodType(_extract_state_patched, env)
    return env
