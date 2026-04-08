"""
TrucGymEnv – Wrapper Gymnasium per a TrucEnv (RLCard).

Exposa la interfície estàndard de Gymnasium per permetre l'ús d'algorismes
com el PPO de Stable-Baselines3 (SB3) amb l'entorn multi-agent de RLCard.
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


class TrucGymEnv(gymnasium.Env):
    """
    Entorn Gymnasium de Truc per a algoritmes SB3.
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

        # Calcular OBS_DIM
        dummy_state, _ = self.rlcard_env.reset()
        obs_dim = self._flatten_obs(dummy_state).shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Estat intern
        self._current_state = None
        self._legal_actions: list = list(range(self.n_actions))
        self._last_obs = np.zeros(obs_dim, dtype=np.float32)


    def _flatten_obs(self, state) -> np.ndarray:
        """Converteix l'estat RLCard en un vector pla de 239 dimensions."""
        obs = state['obs']
        if isinstance(obs, dict):
            return np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _reward_from_raw(self, state) -> float:
        """Extreu el reward intermedi"""
        raw = state.get('raw_obs', {})
        ri = raw.get('reward_intermedis', [0.0, 0.0])
        equip = self.learner_pid % 2
        if isinstance(ri, (list, tuple)) and len(ri) > equip:
            return float(ri[equip]) * 5.0
        return 0.0

    def set_opponent(self, opponent) -> None:
        """Permet canviar l'agent oponent en calent (per a callbacks)."""
        self.opponent = opponent


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state, player_id = self.rlcard_env.reset()

        # Gestionar torns
        state, player_id = self._skip_opponent_turns(state, player_id, reward_acc=0.0)

        if player_id is None:
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
            payoffs = self.rlcard_env.game.get_payoffs()
            reward += float(payoffs[self.learner_pid])
            return self._last_obs.copy(), reward, True, False, {}

        # Reward intermedi pel pas actual
        reward += self._reward_from_raw(state)

        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)

            if player_id is None:
                payoffs = self.rlcard_env.game.get_payoffs()
                reward += float(payoffs[self.learner_pid])
                return self._last_obs.copy(), reward, True, False, {}

            reward += self._reward_from_raw(state)

        # Actualitzar estat
        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, reward, False, False, {}


    def _skip_opponent_turns(self, state, player_id, reward_acc: float):
        """Avança els torns de l'oponent fins que toqui a l'aprenent."""
        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)
        return state, player_id
