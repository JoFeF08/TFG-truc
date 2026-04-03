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

    if tipus == "ppo_mlp":
        from RL.models.model_propi.ppo_loaders import _crear_ppo_mlp
        if not spec.get("ruta"):
            spec["ruta"] = "best.pt"
        return _crear_ppo_mlp(spec, env_config)

    if tipus == "ppo_gru":
        from RL.models.model_propi.ppo_loaders import _crear_ppo_gru
        if not spec.get("ruta"):
            spec["ruta"] = "best.pt"
        return _crear_ppo_gru(spec, env_config)

    if tipus == "regles":
        from RL.models.core.obs_adapter import crear_env_aplanat
        from RL.models.rlcard_legacy.model_adapter import _RLCardModelAdapter
        from RL.models.model_propi.agent_regles import AgentRegles
        env_wrapped = crear_env_aplanat(env_config)
        agent = AgentRegles(num_actions=env_wrapped.num_actions, seed=spec.get("seed"))
        return _RLCardModelAdapter(agent, env_wrapped._extract_state)

    return None
