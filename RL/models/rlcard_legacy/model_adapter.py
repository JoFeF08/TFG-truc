from typing import Any, Callable


class _RLCardModelAdapter:

    def __init__(self, agent: Any, state_extractor: Callable[[dict[str, Any]], dict[str, Any]]):
        self._agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat: dict[str, Any]) -> int:
        rlcard_state = self._extract(estat)
        action, _ = self._agent.eval_step(rlcard_state)
        return int(action)

    def reset_memoria(self) -> None:
        """Reinicialitza l'estat intern de l'agent (ex: LSTM hidden state).
        No fa res si l'agent no manté estat."""
        if hasattr(self._agent, 'reset'):
            self._agent.reset()
