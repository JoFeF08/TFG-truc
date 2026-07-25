"""Utilitats del contracte d'observació del Truc.

Única font de veritat per l'aplanament de l'observació a un vector pla de 240
dimensions (216 cartes + 24 context). Qualsevol lloc que hagi de convertir una
observació de Truc a un vector pla ha de cridar `flatten_obs()`.
"""

import numpy as np


def flatten_obs(obs) -> np.ndarray:
    """Aplana una observació del Truc a un `np.ndarray` de 240 dimensions.
    """
    if isinstance(obs, dict):
        return np.concatenate(
            [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)
