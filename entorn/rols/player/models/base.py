from typing import Protocol


class TrucModel(Protocol):
    """Contracte: qualsevol model ha de tenir triar_accio(estat) -> int."""

    def triar_accio(self, estat: dict) -> int:
        """Rep l'estat brut (TrucGame.get_state) i retorna l'índex d'acció."""
        ...
