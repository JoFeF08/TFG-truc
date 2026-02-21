from typing import Protocol


class EscollidorAccio(Protocol):
    """
    Interfície per escollir una acció humana.
    Rep accions legals i l'estat raw_obs; retorna el codi d'acció escollit.
    """

    def escollir_accio(self, accions_legals: list, state: dict) -> int:
        ...
