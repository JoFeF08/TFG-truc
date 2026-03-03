from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class TrucPlayer:
    """
    Jugador que delega la tria d'acció en un model injectat.
    Si el model és None, actua com un agent per defecte i tria una acció
    aleatòria entre les legals.
    """

    def __init__(
        self,
        player_id: int,
        np_random: np.random.RandomState,
        model: TrucModel | None = None,
    ) -> None:
        self.player_id = player_id
        self.np_random = np_random
        self.model = model
        self.hand: list[str] = []
        self.ma_envits: list[int] = []

    def triar_accio(self, estat: dict[str, Any]) -> int:
        if self.model is not None:
            return self.model.triar_accio(estat)

        accions_legals: list[int] = estat["accions_legals"]
        if len(accions_legals) == 1:
            return int(accions_legals[0])
        return int(self.np_random.choice(accions_legals))
