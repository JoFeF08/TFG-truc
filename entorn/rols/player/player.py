from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class TrucPlayer(ABC):
    """
    Classe abstracta base per a qualsevol jugador de Truc.
    Defineix la interfície mínima requerida per l'entorn TrucGame.
    """

    def __init__(self, player_id: int, np_random: np.random.RandomState) -> None:
        self._player_id = player_id
        self.np_random = np_random
        self.hand: list[str] = []
        self.ma_envits: list[int] = []

    @property
    def player_id(self) -> int:
        return self._player_id

    @abstractmethod
    def triar_accio(self, estat: dict[str, Any]) -> int:
        ...
