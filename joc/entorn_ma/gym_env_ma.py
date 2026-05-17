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

from RL.tools.obs_utils import flatten_obs


class TrucGymEnvMa(gymnasium.Env):
    """
    Entorn Gymnasium de Truc per a mans individuals (SB3).
    Cada episodi correspon a una mà completa.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict, opponent=None, learner_pid: int = 0):
        super().__init__()

        from joc.entorn_ma.env_ma import TrucEnvMa
        from joc.entorn.cartes_accions import ACTION_LIST

        self.rlcard_env = TrucEnvMa(env_config)
        self.learner_pid = learner_pid
        self.n_actions = len(ACTION_LIST)

        if opponent is None:
            from rlcard.agents import RandomAgent
            self.opponent = RandomAgent(num_actions=self.n_actions)
        else:
            self.opponent = opponent

        dummy_state, _ = self.rlcard_env.reset()
        obs_dim = self._flatten_obs(dummy_state).shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._current_state = None
        self._legal_actions: list = list(range(self.n_actions))
        self._last_obs = np.zeros(obs_dim, dtype=np.float32)
        self._pending_reward = 0.0

    def _flatten_obs(self, state) -> np.ndarray:
        return flatten_obs(state['obs'])

    def _reward_from_raw(self, state) -> float:
        raw = state.get('raw_obs', {})
        ri = raw.get('reward_intermedis', [0.0, 0.0])
        equip = self.learner_pid % 2
        if isinstance(ri, (list, tuple)) and len(ri) > equip:
            return float(ri[equip]) * 5.0
        return 0.0

    def set_opponent(self, opponent) -> None:
        self.opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state, player_id = self.rlcard_env.reset()
        state, player_id, pending = self._skip_opponent_turns(state, player_id)
        self._pending_reward = pending

        if player_id is None:
            return self._last_obs.copy(), {}

        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, {}

    def step(self, action: int):
        action = int(action)
        if action not in self._legal_actions:
            action = self._legal_actions[0]

        state, player_id = self.rlcard_env.step(action)
        done = (player_id is None)
        reward = self._pending_reward
        self._pending_reward = 0.0

        if done:
            payoffs = self.rlcard_env.game.get_payoffs()
            reward += float(payoffs[self.learner_pid])
            return self._last_obs.copy(), reward, True, False, {}

        reward += self._reward_from_raw(state)

        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)

            if player_id is None:
                payoffs = self.rlcard_env.game.get_payoffs()
                reward += float(payoffs[self.learner_pid])
                return self._last_obs.copy(), reward, True, False, {}

            reward += self._reward_from_raw(state)

        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, reward, False, False, {}

    def _skip_opponent_turns(self, state, player_id):
        """Avança els torns de l'oponent fins que toqui a l'aprenent,
        acumulant els rewards intermedis generats pels passos de l'oponent."""
        reward_acc = 0.0
        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.rlcard_env.step(opp_action)
            if player_id is not None:  # no acumular si la partida ha acabat
                reward_acc += self._reward_from_raw(state)
        return state, player_id, reward_acc
