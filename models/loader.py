from __future__ import annotations
from typing import Any, Protocol


class TrucModel(Protocol):
    """Contracte: qualsevol model ha de tenir triar_accio(estat) -> int."""

    def triar_accio(self, estat: dict[str, Any]) -> int:
        ...


def crear_model(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel | None:
    """Crea una instància d'un model segons l'especificació donada."""
    tipus = spec.get("tipus", "default")

    if tipus in ("huma", "default"):
        return None

    if tipus == "nfsp":
        from models.adapters.rlcard_model import _crear_nfsp
        return _crear_nfsp(spec, env_config)

    if tipus == "dqn":
        from models.adapters.rlcard_model import _crear_dqn
        return _crear_dqn(spec, env_config)

    return None
