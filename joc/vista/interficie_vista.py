from __future__ import annotations
from typing import Protocol


class Vista(Protocol):
    """
    Contracte que ha de complir qualsevol vista del joc del Truc.
    """

    def demanar_config(self) -> dict:
        """Demana i retorna la configuració del joc."""
        ...

    def mostrar_estat(self, estat: dict) -> None:
        """Mostra l'estat actual del joc (taula, mà, puntuació, info de ronda)."""
        ...

    def escollir_accio(self, accions_legals: list, estat: dict) -> int | None:
        """Llista accions legals; retorna el codi d'acció escollit, o None si s'ha tancat."""
        ...

    def mostrar_accio(self, jugador_id: int, nom_accio: str, es_bot: bool) -> None:
        """Indica quina acció ha executat un jugador.
        Si es_bot=True, la vista pot afegir delay visual."""
        ...

    def mostrar_guanyador_envit(self, equip: int, punts: int, punts_detall: list[int]) -> None:
        """Comunica qui ha guanyat l'envit, quants punts i quins punts tenia cadascú."""
        ...

    def mostrar_guanyador_truc(self, equip: int, punts: int) -> None:
        """Comunica qui ha guanyat el truc (la mà) i quants punts."""
        ...

    def mostrar_fi_partida(self, score: list, payoffs: list) -> None:
        """Mostra el resultat final de la partida."""
        ...

    def demanar_repetir(self) -> bool:
        """Pregunta si es vol jugar una altra partida. Retorna True si sí."""
        ...

    def mostrar_sortint(self) -> None:
        """Indica que es surt de l'aplicació."""
        ...
