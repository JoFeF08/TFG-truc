"""Wrapper d'avaluació per a models RecurrentPPO (Fase 4)."""
import numpy as np


class SB3LSTMEvalAgent:
    use_raw = False

    def __init__(self, model, num_actions: int = 19):
        self.model = model
        self.num_actions = num_actions
        self._lstm_states = None
        self._episode_start = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        """Reinicialitza l'estat LSTM. Cal cridar a l'inici de cada sessió."""
        self._lstm_states = None
        self._episode_start = np.ones((1,), dtype=bool)

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        else:
            obs_flat = np.asarray(obs, dtype=np.float32)

        legal = list(state['legal_actions'].keys())
        action, self._lstm_states = self.model.predict(
            obs_flat[np.newaxis],
            state=self._lstm_states,
            episode_start=self._episode_start,
            deterministic=True,
        )
        self._episode_start = np.zeros((1,), dtype=bool)

        action = int(action[0])
        if action not in legal:
            action = legal[0]
        return action, {}

    def step(self, state):
        action, _ = self.eval_step(state)
        return action
