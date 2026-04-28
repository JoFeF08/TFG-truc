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
            env = _build_env(env_config)
            _orig_ppo = PPO.set_parameters
            def _ppo_sense_optimizer(self, load_path_or_dict, exact_match=True, device="auto"):  # exact_match ignorat intencionalment
                if isinstance(load_path_or_dict, dict):
                    load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                         if "optimizer" not in k}
                return _orig_ppo(self, load_path_or_dict, exact_match=False, device=device)
            PPO.set_parameters = _ppo_sense_optimizer
            try:
                from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
                sb3_model = PPO.load(ruta,
                                     custom_objects={"features_extractor_class": CosMultiInputSB3},
                                     device="cpu")
            finally:
                PPO.set_parameters = _orig_ppo
            eval_agent = SB3PPOEvalAgent(sb3_model, n_actions=env.num_actions)

        elif algorisme == "dqn":
            from stable_baselines3 import DQN
            from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
            env = _build_env(env_config)
            # El model es va entrenar amb COS congelat → l'optimitzador guardat té
            # menys grups de paràmetres que el model reconstruït. Per a inferència
            # no cal l'optimitzador; el patchem per saltar-lo.
            _orig = DQN.set_parameters
            def _sense_optimizer(self, load_path_or_dict, exact_match=True, device="auto"):  # exact_match ignorat intencionalment
                if isinstance(load_path_or_dict, dict):
                    load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                         if "optimizer" not in k}
                return _orig(self, load_path_or_dict, exact_match=False, device=device)
            DQN.set_parameters = _sense_optimizer
            try:
                sb3_model = DQN.load(ruta,
                                     custom_objects={"features_extractor_class": CosMultiInputSB3},
                                     device="cpu")
            finally:
                DQN.set_parameters = _orig
            eval_agent = SB3PPOEvalAgent(sb3_model, n_actions=env.num_actions)

        elif algorisme == "ppo_lstm":
            from sb3_contrib import RecurrentPPO
            from RL.models.sb3.sb3_lstm_eval_agent import SB3LSTMEvalAgent
            sb3_model = RecurrentPPO.load(ruta)
            env = _build_env(env_config)
            eval_agent = SB3LSTMEvalAgent(sb3_model, num_actions=env.num_actions)

        else:
            raise ValueError(f"algorisme SB3 desconegut: {algorisme!r}")

        return _RLCardModelAdapter(eval_agent, env._extract_state)

    return None
