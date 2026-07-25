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
    Entorn Gymnasium de Truc per a mans individuals (SB3), amb un sol seient
    actiu ("learner_pid"); la resta de seients els juga un oponent.

    L'oponent es pot fixar (`opponent=`, per avaluació/depuració) o mostrejar
    a cada `reset()` d'un `OpponentPool` (`opponent_pool=`) — Random /
    AgentRegles amb paràmetres variats / checkpoints congelats d'una lliga.
    Aquest mateix mecanisme cobreix tant les etapes de currículum (pool
    ponderat cap a Random/AgentRegles) com el self-play amb lliga (pool
    ponderat cap a checkpoints propis anteriors).

    Cada episodi correspon a una mà completa. La recompensa és 0 a cada pas
    intermedi i el marge de punts normalitzat (`reward_intermedis`, per
    equip) quan la mà acaba.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict, opponent=None, opponent_pool=None, learner_pid: int = 0):
        super().__init__()

        from joc.entorn_ma.env_ma import TrucEnvMa
        from joc.entorn.cartes_accions import ACTION_LIST

        self.truc_env = TrucEnvMa(env_config)
        self.learner_pid = learner_pid
        self.n_actions = len(ACTION_LIST)

        self.opponent_pool = opponent_pool
        self._fixed_opponent = opponent
        if self._fixed_opponent is None and self.opponent_pool is None:
            from RL.models.model_propi.random_agent import RandomAgent
            self._fixed_opponent = RandomAgent(num_actions=self.n_actions)

        self.opponent = self._fixed_opponent if self._fixed_opponent is not None else self.opponent_pool.sample()

        dummy_state, _ = self.truc_env.reset()
        obs_dim = self._flatten_obs(dummy_state).shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._current_state = None
        self._legal_actions: list = list(range(self.n_actions))
        self._last_obs = np.zeros(obs_dim, dtype=np.float32)

    def _flatten_obs(self, state) -> np.ndarray:
        return flatten_obs(state['obs'])

    def _reward_equip(self, state) -> float:
        """Marge de punts normalitzat (reward_intermedis) de l'equip de l'aprenent,
        disponible a l'estat terminal (fi de mà)."""
        raw = state.get('raw_obs', {})
        ri = raw.get('reward_intermedis', [0.0, 0.0])
        equip = self.learner_pid % 2
        if isinstance(ri, (list, tuple)) and len(ri) > equip:
            return float(ri[equip])
        return 0.0

    def set_opponent(self, opponent) -> None:
        self._fixed_opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.opponent = self._fixed_opponent if self._fixed_opponent is not None else self.opponent_pool.sample()

        state, player_id = self.truc_env.reset()
        state, player_id = self._skip_opponent_turns(state, player_id)

        # Cas rar (nomes si learner_pid != "mà"): la mà s'acaba abans que
        # l'aprenent arribi a actuar. reset() no pot retornar una recompensa
        # (l'API de Gymnasium no ho permet), així que simplement repartim
        # una altra mà.
        intents = 0
        while player_id is None and intents < 8:
            state, player_id = self.truc_env.reset()
            state, player_id = self._skip_opponent_turns(state, player_id)
            intents += 1

        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, {}

    def step(self, action: int):
        action = int(action)
        if action not in self._legal_actions:
            action = self._legal_actions[0]

        state, player_id = self.truc_env.step(action)

        if player_id is None:
            reward = self._reward_equip(state)
            self._last_obs = self._flatten_obs(state)
            return self._last_obs.copy(), reward, True, False, {}

        state, player_id = self._skip_opponent_turns(state, player_id)

        if player_id is None:
            reward = self._reward_equip(state)
            self._last_obs = self._flatten_obs(state)
            return self._last_obs.copy(), reward, True, False, {}

        self._current_state = state
        self._legal_actions = list(state['legal_actions'].keys())
        obs = self._flatten_obs(state)
        self._last_obs = obs
        return obs, 0.0, False, False, {}

    def _skip_opponent_turns(self, state, player_id):
        """Avança els torns de l'oponent fins que toqui a l'aprenent o s'acabi la mà."""
        while player_id != self.learner_pid and player_id is not None:
            opp_action, _ = self.opponent.eval_step(state)
            state, player_id = self.truc_env.step(opp_action)
        return state, player_id
