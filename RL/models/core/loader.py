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
        from RL.models.model_propi.model_adapter import EvalStepModelAdapter
        from RL.models.model_propi.agent_regles import AgentRegles
        env = _build_env(env_config)
        agent = AgentRegles(num_actions=env.num_actions, seed=spec.get("seed"))
        return EvalStepModelAdapter(agent, env._extract_state)

    if tipus == "sb3":
        from RL.models.model_propi.model_adapter import EvalStepModelAdapter
        from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent

        ruta = spec.get("ruta")
        if not ruta:
            raise ValueError("spec['ruta'] és obligatori per tipus='sb3'")
        algorisme = spec.get("algorisme", "ppo").lower()

        if algorisme in ("ppo", "dqn", "maskable_ppo"):
            from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
            if algorisme == "maskable_ppo":
                from sb3_contrib import MaskablePPO as ModelCls
            else:
                sb3_cls = __import__("stable_baselines3", fromlist=[algorisme.upper()])
                ModelCls = getattr(sb3_cls, algorisme.upper())
            env = _build_env(env_config)

            _orig_set_parameters = ModelCls.set_parameters
            def _sense_optimizer(self, load_path_or_dict, exact_match=True, device="auto"):  # exact_match ignorat intencionalment
                if isinstance(load_path_or_dict, dict):
                    load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                         if "optimizer" not in k}
                return _orig_set_parameters(self, load_path_or_dict, exact_match=False, device=device)
            ModelCls.set_parameters = _sense_optimizer
            try:
                sb3_model = ModelCls.load(ruta,
                                          custom_objects={"features_extractor_class": CosMultiInputSB3},
                                          device="cpu")
            finally:
                ModelCls.set_parameters = _orig_set_parameters
            eval_agent = SB3PPOEvalAgent(sb3_model, n_actions=env.num_actions)

        elif algorisme == "ppo_lstm":
            from sb3_contrib import RecurrentPPO
            from RL.models.sb3.sb3_lstm_eval_agent import SB3LSTMEvalAgent
            sb3_model = RecurrentPPO.load(ruta)
            env = _build_env(env_config)
            eval_agent = SB3LSTMEvalAgent(sb3_model, num_actions=env.num_actions)

        else:
            raise ValueError(f"algorisme SB3 desconegut: {algorisme!r}")

        return EvalStepModelAdapter(eval_agent, env._extract_state)

    return None
