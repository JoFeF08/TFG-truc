import numpy as np
import types
from typing import Any


def crear_env_aplanat(env_config: dict[str, Any]):
    """Crea un TrucEnv amb l'observació aplanada (233 dimensions)."""
    from joc.entorn.env import TrucEnv
    env = TrucEnv(
        config={
            "num_jugadors": env_config.get("num_jugadors", 2),
            "cartes_jugador": env_config.get("cartes_jugador", 3),
            "senyes": env_config.get("senyes", False),
        }
    )
    return wrap_env_aplanat(env)


def wrap_env_aplanat(env):
    """
    Retorna una observació aplanada (233 dimensions), 
    adequada per a models que integren el COS.
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
