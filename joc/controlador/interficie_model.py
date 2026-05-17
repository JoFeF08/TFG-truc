from __future__ import annotations
from typing import Protocol


class Model(Protocol):
    """
    Contracte que el controlador usa per parlar amb el model.
    """

    def iniciar(self, config: dict) -> None:
        """Crea i inicialitza una partida amb la configuració donada."""
        ...

    def get_estat(self, jugador_id: int) -> dict:
        """Retorna l'estat visible per un jugador."""
        ...

    def get_jugador_actual(self) -> int:
        """Retorna l'id del jugador que ha de jugar."""
        ...

    def es_huma(self, jugador_id: int) -> bool:
        """Retorna True si el jugador és humà."""
        ...

    def get_accio_bot(self, jugador_id: int) -> tuple[int, str]:
        """Retorna (acció_id, nom_acció) triada pel bot."""
        ...

    def aplicar_accio(self, accio: int) -> None:
        """Aplica una acció i avança l'estat del joc."""
        ...

    def get_guanyador_envit_recent(self) -> tuple[int, int, list[int]] | None:
        """Retorna (equip, punts, punts_detall) de l'envit que s'acaba de tancar, si n'hi ha."""
        ...

    def get_guanyador_truc_recent(self) -> tuple[int, int] | None:
        """Retorna (equip, punts) del truc (mà) que s'acaba de tancar, si n'hi ha."""
        ...

    def es_final(self) -> bool:
        """Retorna True si la partida ha acabat."""
        ...

    def get_resultat(self) -> dict:
        """Retorna {'score': [...], 'payoffs': [...]}."""
        ...
