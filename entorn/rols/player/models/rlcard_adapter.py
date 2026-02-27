from typing import Callable


class RLCardModelAdapter:
    """
    Adaptador que embolcalla un agent RLCard perquè sigui compatible
    amb la interfície TrucModel.
    """

    def __init__(self, agent, state_extractor: Callable[[dict], dict]):
        self.agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat: dict) -> int:
        rlcard_state = self._extract(estat)
        action, _ = self.agent.eval_step(rlcard_state)
        return int(action)
