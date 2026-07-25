"""Entorn PettingZoo (AEC) del Truc.

Font única de veritat per a l'entrenament multi-agent (self-play amb
SuperSuit + MaskablePPO, vegeu RL/entrenament/entrenament_selfplay.py).
`TrucGymEnvMa` n'és un embolcall d'un sol seient, per a les etapes del
currículum amb un oponent fix/script (Random/AgentRegles/pool) en lloc
d'un altre agent après.

Una mà = un episodi (`TrucGameMa`, igual que `TrucEnvMa`). Reward = 0 a
cada pas intermedi, i el marge de punts normalitzat (`reward_intermedis`,
per equip) quan la mà acaba.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from joc.entorn_ma.game_ma import TrucGameMa
from joc.entorn.cartes_accions import ACTION_LIST
from joc.entorn.obs_builder import extract_obs, obs_shapes

N_ACTIONS = len(ACTION_LIST)


def env(**kwargs):
    """Factoria amb els embolcalls estàndard de PettingZoo (comprovacions
    d'accions il·legals/fora de rang i d'ordre de torn)."""
    e = TrucAECEnv(**kwargs)
    e = wrappers.TerminateIllegalWrapper(e, illegal_reward=-1)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class TrucAECEnv(AECEnv):
    metadata = {"render_modes": [], "name": "truc_v1", "is_parallelizable": False}

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        env_config = env_config or {}
        self.num_jugadors = env_config.get('num_jugadors', 2)
        self.cartes_jugador = env_config.get('cartes_jugador', 3)
        self.senyes = env_config.get('senyes', False)
        self.puntuacio_final = env_config.get('puntuacio_final', 999)
        self.permetre_apostes = env_config.get('permetre_apostes', True)
        self.permetre_truc = env_config.get('permetre_truc', True)

        self.possible_agents = [f"player_{i}" for i in range(self.num_jugadors)]
        self.agents = self.possible_agents[:]

        cartes_shape, context_size = obs_shapes(self.num_jugadors, self.senyes)
        self._obs_dim = int(np.prod(cartes_shape)) + context_size

        self.observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
            }) for a in self.possible_agents
        }
        self.action_spaces = {a: spaces.Discrete(N_ACTIONS) for a in self.possible_agents}

        self.game = TrucGameMa(
            num_jugadors=self.num_jugadors,
            cartes_jugador=self.cartes_jugador,
            senyes=self.senyes,
            puntuacio_final=self.puntuacio_final,
            permetre_apostes=self.permetre_apostes,
            permetre_truc=self.permetre_truc,
        )

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self.possible_agents[0]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _pid(self, agent: str) -> int:
        return self.possible_agents.index(agent)

    def _flat_obs(self, state: dict) -> np.ndarray:
        obs_cartes, obs_context = extract_obs(state, self.num_jugadors, self.cartes_jugador, self.senyes)
        return np.concatenate([obs_cartes.flatten(), obs_context], axis=0).astype(np.float32)

    def observe(self, agent: str):
        state = self.game.get_state(self._pid(agent))
        obs = self._flat_obs(state)

        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if agent == self.agent_selection:
            for a in state['accions_legals']:
                mask[a] = 1
        return {"observation": obs, "action_mask": mask}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.game.np_random = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        _state, player_id = self.game.init_game()
        self.agent_selection = self.possible_agents[player_id]

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)

        self._cumulative_rewards[agent] = 0
        state, next_player_id = self.game.step(int(action))

        if next_player_id is None:
            # Fi de la mà: reward terminal = marge de punts normalitzat, per equip.
            reward_intermedis = state.get('reward_intermedis', [0.0, 0.0])
            for a in self.agents:
                equip = self._pid(a) % 2
                self.rewards[a] = float(reward_intermedis[equip]) if len(reward_intermedis) > equip else 0.0
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()
        else:
            self.rewards = {a: 0.0 for a in self.agents}
            self.agent_selection = self.possible_agents[next_player_id]

    def close(self):
        pass
