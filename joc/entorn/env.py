from __future__ import annotations

import sys
import os
import numpy as np
from collections import OrderedDict

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn.game import TrucGame
from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, ACTIONS_SIGNAL, init_joc_cartes
from joc.entorn.obs_builder import extract_obs, obs_shapes


class BaseTrucEnv:
    """Base comuna per als entorns de Truc (TrucEnv, TrucEnvMa): reset/step
    turn-based sobre un `self.game`, seeding i un mode `run()` per jugar
    partides senceres amb agents `eval_step()`/`step()`."""

    def __init__(self, config: dict) -> None:
        self.allow_step_back = self.game.allow_step_back = config.get('allow_step_back', False)
        self.action_recorder: list = []

        self.num_players = self.game.get_num_players()
        self.num_actions = self.game.get_num_actions()

        self.timestep = 0
        self.seed(config.get('seed'))

    def reset(self):
        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    def step(self, action, raw_action: bool = False):
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        self.action_recorder.append((self.get_player_id(), action))
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def step_back(self):
        if not self.allow_step_back:
            raise Exception("Step back is off. Cal allow_step_back=True per fer-lo servir.")

        if not self.game.step_back():
            return False

        player_id = self.get_player_id()
        state = self.get_state(player_id)
        return state, player_id

    def set_agents(self, agents) -> None:
        """Agents que interactuaran amb l'entorn. Cal cridar-ho abans de `run()`."""
        self.agents = agents

    def run(self, is_training: bool = False):
        """Juga una partida completa amb els agents fixats via `set_agents()`."""
        trajectories: list = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        trajectories[player_id].append(state)
        while not self.is_over():
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            trajectories[player_id].append(action)

            state = next_state
            player_id = next_player_id

            if not self.game.is_over():
                trajectories[player_id].append(state)

        for player_id in range(self.num_players):
            trajectories[player_id].append(self.get_state(player_id))

        return trajectories, self.get_payoffs()

    def is_over(self) -> bool:
        return self.game.is_over()

    def get_player_id(self) -> int:
        return self.game.get_player_id()

    def get_state(self, player_id: int):
        return self._extract_state(self.game.get_state(player_id))

    def get_payoffs(self):
        raise NotImplementedError

    def get_action_feature(self, action: int) -> np.ndarray:
        feature = np.zeros(self.num_actions, dtype=np.int8)
        feature[action] = 1
        return feature

    def seed(self, seed: int | None = None) -> int | None:
        self.np_random = np.random.RandomState(seed)
        self.game.np_random = self.np_random
        return seed

    def _extract_state(self, state):
        raise NotImplementedError

    def _decode_action(self, action_id: int):
        raise NotImplementedError

    def _get_legal_actions(self):
        raise NotImplementedError


class TrucEnv(BaseTrucEnv):
    """
    Entorn del joc del Truc.
    Aquesta classe connecta la lògica del joc (TrucGame) amb una interfície
    estàndard d'entorn turn-based (reset/step/run) reutilitzada per TrucEnvMa.
    """
    def __init__(self, config):
        self.name = 'truc'
        self.num_jugadors = config.get('num_jugadors', 2)
        self.cartes_jugador = config.get('cartes_jugador', 3)
        self.puntuacio_final = config.get('puntuacio_final', 24)
        senyes = config.get('senyes', False)
        verbose = config.get('verbose', False)
        player_class = config.get('player_class', None)

        if player_class:
            self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final,
                                 player_class=player_class,
                                 verbose=verbose)
        else:
             self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final,
                                 verbose=verbose)
        
        config.setdefault('allow_step_back', False)
        config.setdefault('seed', None)
        super().__init__(config)
        
        self.cartes = init_joc_cartes()
        self.carta_map = {carta: i for i, carta in enumerate(self.cartes)}
        self.signal_map = {signal: i for i, signal in enumerate(ACTIONS_SIGNAL)}

        # Format multi-entrada (vegeu joc/entorn/obs_builder.py)
        self.OBS_CARTES_SHAPE, self.OBS_CONTEXT_SIZE = obs_shapes(self.num_jugadors, senyes)

        # state_size i state_shape per compatibilitat amb RLCard
        self.state_size = (
            self.OBS_CARTES_SHAPE[0] * self.OBS_CARTES_SHAPE[1] * self.OBS_CARTES_SHAPE[2]
            + self.OBS_CONTEXT_SIZE
        )
        self.state_shape = [[self.state_size] for _ in range(self.num_jugadors)]
        self.action_shape = [[len(ACTION_LIST)] for _ in range(self.num_jugadors)]

    def _extract_state(self, state):
        """
        Extreu l'estat del joc en format multi-entrada per a la xarxa neuronal.

        Retorna un diccionari 'obs' amb:
          - 'obs_cartes'  : np.ndarray (C, 4, 9) float32
          - 'obs_context' : np.ndarray (17,)     float32
        """
        obs_cartes, obs_context = extract_obs(
            state, self.num_jugadors, self.cartes_jugador, self.game.senyes
        )

        # Accions legals
        legal_actions_list = state['accions_legals']
        legal_actions = OrderedDict({a: None for a in legal_actions_list})

        # Estat final per a l'agent
        extracted_state = {
            'obs': {'obs_cartes': obs_cartes, 'obs_context': obs_context},
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': [ACTION_LIST[a] for a in legal_actions_list],
            'action_record': self.action_recorder
        }
        return extracted_state

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_estat_taula(self, player_id):
        """
        Retorna l'estat brut del joc per mostrar la taula
        """
        return self.game.get_state(player_id)

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        return self.game.get_legal_actions()


def reorganize_amb_rewards(trajectories, payoffs, reward_key='reward_intermedis'):
    """
    RLCard per disseny posa reward=0 a totes les transicions intermèdies i el payoff final a l'última.
    Aquesta versió injecta els rewards intermedis llegits del raw_obs.
    """
    num_players = len(trajectories)
    # traj[i] = state, traj[i+1] = action, traj[i+2] = next_state
    hist_trajectoria = [[] for _ in range(num_players)]
    
    for player in range(num_players):
        traj = trajectories[player]
        for i in range(0, len(traj) - 2, 2):
            is_last = (i == len(traj) - 3)
            
            if is_last:
                r = payoffs[player]
                done = True
            else:
                next_state = traj[i + 2]
                raw = next_state.get('raw_obs', {})
                ri = raw.get(reward_key, [0.0, 0.0])
                
                equip = player % 2
                r = ri[equip] if isinstance(ri, list) else 0.0
                done = False
                
            transition = traj[i:i+3].copy()
            transition.insert(2, r)
            transition.append(done)
            hist_trajectoria[player].append(transition)
            
    return hist_trajectoria
