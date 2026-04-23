"""Entorn de sessions multi-partida per a Fase 4 (memòria cross-partides).

Un episodi RL = N partides consecutives contra el mateix oponent. L'oponent es
samplea del pool al principi de cada sessió (a `reset()`). Això força l'LSTM a
detectar el patró de l'oponent durant la sessió i adaptar la política.
"""
import random
import numpy as np
import gymnasium
from gymnasium import spaces

from joc.entorn_ma.gym_env_ma import TrucGymEnvMa


class TrucGymEnvSessio(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict, opponent_pool_fn,
                 n_partides: int = 5, learner_pid: int = 0, seed=None):
        """
        opponent_pool_fn(rng) -> (nom, AgentRegles). Es crida a cada reset.
        """
        super().__init__()
        self.env_config = env_config
        self.opponent_pool_fn = opponent_pool_fn
        self.n_partides = int(n_partides)
        self.learner_pid = learner_pid
        self._rng = random.Random(seed)

        inicial_oponent = self._nou_oponent()
        self._inner = TrucGymEnvMa(env_config, opponent=inicial_oponent,
                                   learner_pid=learner_pid)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

        self._partides_fetes = 0
        self._oponent_actual_nom = None

    def _nou_oponent(self):
        nom, agent = self.opponent_pool_fn(self._rng)
        self._oponent_actual_nom = nom
        return agent

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self._inner.set_opponent(self._nou_oponent())
        self._partides_fetes = 0
        obs, info = self._inner.reset()
        info = dict(info)
        info['partida_idx'] = 0
        info['oponent_nom'] = self._oponent_actual_nom
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        info = dict(info)
        if terminated:
            self._partides_fetes += 1
            info['partida_idx'] = self._partides_fetes - 1
            info['partida_acabada'] = True
            info['oponent_nom'] = self._oponent_actual_nom
            if self._partides_fetes < self.n_partides:
                # Continua sessió: nova partida, mateix oponent, LSTM NO es reseteja
                obs, _ = self._inner.reset()
                terminated = False
            # Si partides_fetes == n_partides: terminated=True, fi de sessió
        else:
            info['partida_idx'] = self._partides_fetes
            info['oponent_nom'] = self._oponent_actual_nom
        return obs, reward, terminated, truncated, info
