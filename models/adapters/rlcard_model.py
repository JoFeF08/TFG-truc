import os
from typing import Any, Callable

from models.loader import TrucModel

_DEFAULT_HIDDEN_LAYERS = [256, 256]


class _RLCardModelAdapter:
    """
    Adaptador que encapsula un agent procedent de la llibreria RLCard
    perquè sigui compatible amb la interfície universal `TrucModel`.
    """

    def __init__(self, agent: Any, state_extractor: Callable[[dict[str, Any]], dict[str, Any]]):
        self._agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat: dict[str, Any]) -> int:
        rlcard_state = self._extract(estat)
        action, _ = self._agent.eval_step(rlcard_state)
        return int(action)


def _crear_env_temp(env_config: dict[str, Any]):
    """
    Crea i retorna un entorn `TrucEnv` temporal només per extreure les dimensions 
    de les capes d'entrada (state_shape) i sortida (num_actions) que l'arquitectura 
    de la xarxa neuronal necessita durant la inicialització.
    """
    from entorn.env import TrucEnv

    return TrucEnv(
        config={
            "num_jugadors": env_config.get("num_jugadors", 2),
            "cartes_jugador": env_config.get("cartes_jugador", 3),
            "senyes": env_config.get("senyes", False),
        }
    )


def _crear_rlcard_model(
    spec: dict[str, Any],
    env_config: dict[str, Any],
    nom_tipus: str,
    crear_agent: Callable,
    carregar_pesos: Callable[[Any, dict[str, Any]], None],
) -> TrucModel:
    """
    Funció base que implementa tota la lògica compartida per a la instanciació i 
    càrrega de pesos de qualsevol model de RLCard (NFSP, DQN, etc
    """
    import torch

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model {nom_tipus} a: {ruta}")

    hidden_layers = spec.get("hidden_layers", _DEFAULT_HIDDEN_LAYERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_temp = _crear_env_temp(env_config)

    agent = crear_agent(env_temp, hidden_layers, device)

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    carregar_pesos(agent, checkpoint)
    print(f"Model {nom_tipus} carregat des de: {ruta}")

    return _RLCardModelAdapter(agent, env_temp._extract_state)


def _crear_nfsp(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Fa servir la factoria comuna per encapsular i generar un model NFSP.
    Mapeja tant els pesos del classificador supervisat ('sl_net') com de l'estimador Q ('q_net').
    """
    from rlcard.agents import NFSPAgent

    return _crear_rlcard_model(
        spec,
        env_config,
        "NFSP",
        crear_agent=lambda env, hl, dev: NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=hl,
            q_mlp_layers=hl,
            device=dev,
        ),
        carregar_pesos=lambda ag, ck: (
            ag._rl_agent.q_estimator.qnet.load_state_dict(ck["q_net"]),
            ag.policy_network.load_state_dict(ck["sl_net"]),
        ),
    )


def _crear_dqn(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Fa servir la factoria comuna per encapsular i generar un model Deep Q-Network (DQN).
    Tan sols mapeja la xarxa Q ('q_net').
    """
    from rlcard.agents import DQNAgent

    return _crear_rlcard_model(
        spec,
        env_config,
        "DQN",
        crear_agent=lambda env, hl, dev: DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=hl,
            device=dev,
        ),
        carregar_pesos=lambda ag, ck: ag.q_estimator.qnet.load_state_dict(ck),
    )
