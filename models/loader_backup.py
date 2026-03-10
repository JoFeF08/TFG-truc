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

    import sys
    is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, 'nuitka_version')

    if tipus == "nfsp":
        if is_frozen:
             raise ImportError("NFSP (Torch) no s'ha d'usar en mode frozen. Usa l'adaptador ONNX.")
        from models.adapters.rlcard_model import _crear_nfsp
        return _crear_nfsp(spec, env_config)

    if tipus == "dqn":
        if is_frozen:
             raise ImportError("DQN (Torch) no s'ha d'usar en mode frozen. Usa l'adaptador ONNX.")
        from models.adapters.rlcard_model import _crear_dqn
        return _crear_dqn(spec, env_config)

    ruta = spec.get("ruta")
    if tipus == "onnx" or (ruta and ruta.lower().endswith(".onnx")):
        from models.adapters.onnx_model import ONNXModelAdapter
        from entorn.env import TrucEnv
        from models.adapters.feature_extractor import wrap_env_aplanat

        env = TrucEnv(
            config={
                "num_jugadors": env_config.get("num_jugadors", 2),
                "cartes_jugador": env_config.get("cartes_jugador", 3),
                "senyes": env_config.get("senyes", False),
            }
        )
        env_wrapped = wrap_env_aplanat(env)

        return ONNXModelAdapter(ruta, env_wrapped._extract_state)

    return None
