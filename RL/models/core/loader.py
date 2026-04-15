from __future__ import annotations
from typing import Any, Protocol


class TrucModel(Protocol):
    """Contracte: qualsevol model ha de tenir triar_accio(estat) -> int."""

    def triar_accio(self, estat: dict[str, Any]) -> int:
        ...


def _build_env(env_config: dict[str, Any]):
    from joc.entorn.env import TrucEnv
    return TrucEnv(config={
        "num_jugadors": env_config.get("num_jugadors", 2),
        "cartes_jugador": env_config.get("cartes_jugador", 3),
        "senyes": env_config.get("senyes", False),
    })


def crear_model(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel | None:
    """Crea una instància d'un model segons l'especificació donada."""
    if spec is None:
        return None

    tipus = spec.get("tipus", "default")

    if tipus in ("huma", "default"):
        return None

    if tipus == "regles":
        from RL.models.rlcard_legacy.model_adapter import _RLCardModelAdapter
        from RL.models.model_propi.agent_regles import AgentRegles
        env = _build_env(env_config)
        agent = AgentRegles(num_actions=env.num_actions, seed=spec.get("seed"))
        return _RLCardModelAdapter(agent, env._extract_state)

    if tipus == "sb3":
        
        from RL.models.rlcard_legacy.model_adapter import _RLCardModelAdapter
        from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
        
        ruta = spec.get("ruta")
        if not ruta:
            raise ValueError("spec['ruta'] és obligatori per tipus='sb3'")
        algorisme = spec.get("algorisme", "ppo").lower()
        
        if algorisme == "ppo":
            from stable_baselines3 import PPO
            sb3_model = PPO.load(ruta)
        
        elif algorisme == "dqn":
            from stable_baselines3 import DQN
            sb3_model = DQN.load(ruta)
        else:
            raise ValueError(f"algorisme SB3 desconegut: {algorisme!r}")
        env = _build_env(env_config)
        eval_agent = SB3PPOEvalAgent(sb3_model, n_actions=env.num_actions)
        return _RLCardModelAdapter(eval_agent, env._extract_state)

    return None
