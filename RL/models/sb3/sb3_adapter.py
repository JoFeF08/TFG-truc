import numpy as np
from joc.entorn.cartes_accions import ACTION_LIST
from RL.tools.obs_utils import flatten_obs

N_ACTIONS = len(ACTION_LIST)

class SB3PPOEvalAgent:
    """
    Wrapper per fer un model MaskablePPO de SB3 compatible amb
    l'interfície eval_step() de RLCard (usada per evaluar_agent).
    Aquest adaptador permet avaluar i fer jugar els models SB3 de
    la mateixa manera que es feia amb els agents propis o de RLCard.
    """
    use_raw = False

    def __init__(self, model, n_actions: int = N_ACTIONS):
        self.model = model
        self.num_actions = n_actions

    def eval_step(self, state):
        obs_flat = flatten_obs(state['obs'])
        action, _ = self.model.predict(
            obs_flat[np.newaxis],
            deterministic=True,
        )
        action = int(action[0])
        legal = list(state['legal_actions'].keys())
        if action not in legal:
            action = legal[0]
        return action, {}
