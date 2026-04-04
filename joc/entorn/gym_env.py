"""
TrucGymEnv – Wrapper Gymnasium per a TrucEnv (RLCard).

Exposa la interfície estàndard de Gymnasium per permetre l'ús d'algorismes
com el PPO de Stable-Baselines3 (SB3) amb l'entorn multi-agent de RLCard.

Lògica interna:
  - El wrapper gestiona els torns de l'oponent internament.
  - L'agent aprenent sempre actua com a jugador `learner_pid`.
  - Els rewards intermedis s'acumulen (amplificats ×5) i el payoff final
    s'afegeix quan acaba la partida.
  - Suporta action masking via el mètode `action_masks()`,
    compatible amb MaskablePPO de sb3_contrib.
"""

import sys
import os
import numpy as np

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

import gymnasium
from gymnasium import spaces


# Dimensions de l'observació aplanada: 6×4×9 (cartes) + 23 (context)
OBS_DIM = 6 * 4 * 9 + 23  # = 239


class TrucGymEnv(gymnasium.Env):
    """
    Entorn Gymnasium de Truc per a algoritmes SB3.

    Paràmetres
    ----------
    env_config : dict
        Configuració per a TrucEnv (num_jugadors, cartes_jugador, etc.).
    opponent : agent amb eval_step(state) -> (action, info), opcional
        Agent oponent. Si és None s'usa RandomAgent.
    learner_pid : int
        ID del jugador aprenent (0 o 1).
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict, opponent=None, learner_pid: int = 0):
        super().__init__()

        from joc.entorn.env import TrucEnv
        from joc.entorn.cartes_accions import ACTION_LIST

        self.rlcard_env = TrucEnv(env_config)
        self.learner_pid = learner_pid
        self.n_actions = len(ACTION_LIST)

        # Oponent per defecte: Random
        if opponent is None:
            from rlcard.agents import RandomAgent
            self.opponent = RandomAgent(num_actions=self.n_actions)
        else:
            self.opponent = opponent

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Estat intern
        self._current_state = None
        self._legal_actions: list = list(range(self.n_actions))
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, state) -> np.ndarray:
        """Converteix l'estat RLCard en un vector pla de 239 dimensions."""
        obs = state['obs']
        if isinstance(obs, dict):
            return np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _reward_from_raw(self, state) -> float:
        """Extreu el reward intermedi del jugador aprenent (amplificat ×5)."""
        raw = state.get('raw_obs', {})
        ri = raw.get('reward_intermedis', [0.0, 0.0])
        equip = self.learner_pid % 2
        if isinstance(ri, (list, tuple)) and len(ri) > equip:
            return float(ri[equip]) * 5.0
        return 0.0

    # ------------------------------------------------------------------
    # Action masking (per a MaskablePPO de sb3_contrib)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Retorna un array booleà amb les accions legals actuals."""
        mask = np.zeros(self.n_actions, dtype=bool)
        for a in self._legal_actions:
            mask[a] = True
        return mask

    def set_opponent(self, opponent) -> None:
        """Permet canviar l'agent oponent en calent (per a callbacks)."""
        self.opponent = opponent

    # ------------------------------------------------------------------
    # Interfície Gymnasium
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state, player_id = self.rlcard_env.reset()

        # Gestionar torns de l'oponent si va primer
        state, player_id = self._skip_opponent_turns(state, player_id, reward_acc=0.0)

        if player_id is None:
            # Cas extrem: la partida ha acabat abans del primer torn
            return self._last_obs.copy(), {}

        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, {}

    def step(self, action: int):
        """
        Executa l'acció de l'aprenent i gestiona els torns de l'oponent fins
        que torni a ser el torn de l'aprenent o acabi la partida.

        Retorna: (obs, reward, terminated, truncated, info)
        """
        state, player_id = self.rlcard_env.step(int(action))
        done = (player_id is None)
        reward = 0.0

        if done:
            # Partida acabada en el torn de l'aprenent
            payoffs = self.rlcard_env.game.get_payoffs()
            reward += float(payoffs[self.learner_pid])
            return self._last_obs.copy(), reward, True, False, {}

        # Reward intermedi pel pas actual
        reward += self._reward_from_raw(state)

        # Gestionar torns de l'oponent
        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)

            if player_id is None:
                # Partida acabada en un torn de l'oponent
                payoffs = self.rlcard_env.game.get_payoffs()
                reward += float(payoffs[self.learner_pid])
                return self._last_obs.copy(), reward, True, False, {}

            reward += self._reward_from_raw(state)

        # Actualitzar estat intern
        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, reward, False, False, {}

    # ------------------------------------------------------------------
    # Utilitat interna
    # ------------------------------------------------------------------

    def _skip_opponent_turns(self, state, player_id, reward_acc: float):
        """Avança els torns de l'oponent fins que toqui a l'aprenent."""
        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)
        return state, player_id
