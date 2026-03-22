from __future__ import annotations
from typing import Any, Protocol


class TrucModel(Protocol):
    """Contracte: qualsevol model ha de tenir triar_accio(estat) -> int."""

    def triar_accio(self, estat: dict[str, Any]) -> int:
        ...


def crear_model(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel | None:
    """Crea una instància d'un model segons l'especificació donada."""
    if spec is None:
        spec = {"tipus": "ppo_mlp", "ruta": "best.pt"}
        
    tipus = spec.get("tipus", "ppo_mlp")

    if tipus in ("huma", "default"):
        return None

    if tipus == "nfsp":
        from RL.models.rlcard_legacy.adapters.rlcard_model import _crear_nfsp
        return _crear_nfsp(spec, env_config)

    if tipus == "dqn":
        from RL.models.rlcard_legacy.adapters.rlcard_model import _crear_dqn
        return _crear_dqn(spec, env_config)
        
    if tipus == "numpy_dqn":
        from RL.models.rlcard_legacy.numpy_agent import _crear_numpy_dqn
        return _crear_numpy_dqn(spec, env_config)

    if tipus == "ppo_mlp":
        from RL.models.rlcard_legacy.adapters.rlcard_model import _crear_ppo_mlp
        if not spec.get("ruta"):
            spec["ruta"] = "best.pt"
        return _crear_ppo_mlp(spec, env_config)

    if tipus == "ppo_gru":
        from RL.models.rlcard_legacy.adapters.rlcard_model import _crear_ppo_gru
        if not spec.get("ruta"):
            spec["ruta"] = "best.pt"
        return _crear_ppo_gru(spec, env_config)

    # Si es "default", usem PPO MLP per defecte
    if tipus == "default":
        from RL.models.rlcard_legacy.adapters.rlcard_model import _crear_ppo_mlp
        spec_def = {"tipus": "ppo_mlp", "ruta": "best.pt"}
        return _crear_ppo_mlp(spec_def, env_config)

    return None
