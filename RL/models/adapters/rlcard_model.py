import os
from typing import Any, Callable

from RL.models.loader import TrucModel

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
    Crea i retorna un entorn `TrucEnv` temporal aplanat (233 dims)
    per extreure les dimensions de l'arquitectura.
    """
    from joc.entorn.env import TrucEnv
    from RL.models.adapters.feature_extractor import wrap_env_aplanat

    env = TrucEnv(
        config={
            "num_jugadors": env_config.get("num_jugadors", 2),
            "cartes_jugador": env_config.get("cartes_jugador", 3),
            "senyes": env_config.get("senyes", False),
        }
    )
    return wrap_env_aplanat(env)


def _crear_nfsp(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Crea un model NFSP unificat (integra el COS i el MLP).
    """
    import torch
    import copy
    from rlcard.agents.nfsp_agent import NFSPAgent
    from RL.models.xarxa_unificada import XarxaUnificada

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model NFSP a: {ruta}")

    hidden_layers = spec.get("hidden_layers", _DEFAULT_HIDDEN_LAYERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Entorn aplanat (233 dimensions)
    env_wrapped = _crear_env_temp(env_config)

    # Agent NFSP base
    agent = NFSPAgent(
        num_actions=env_wrapped.num_actions,
        state_shape=env_wrapped.state_shape[0],
        hidden_layers_sizes=hidden_layers,
        q_mlp_layers=hidden_layers,
        device=device,
    )

    # Xarxes unificades (mode scratch perquè carregarem pesos del checkpoint)
    q_net = XarxaUnificada(env_wrapped.num_actions, hidden_layers, "scratch", device=device, output="q")
    sl_net = XarxaUnificada(env_wrapped.num_actions, hidden_layers, "scratch", device=device, output="policy")

    # Carregar pesos
    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    
    # Suport per a diferents estils de guardat (unificats o antics)
    q_sd = checkpoint.get("q", checkpoint.get("q_net", checkpoint))
    sl_sd = checkpoint.get("sl", checkpoint.get("sl_net", checkpoint))

    q_net.load_state_dict(q_sd if isinstance(q_sd, dict) else checkpoint)
    sl_net.load_state_dict(sl_sd if isinstance(sl_sd, dict) else checkpoint)

    # Injectar xarxes a l'agent
    agent._rl_agent.q_estimator.qnet = q_net
    agent._rl_agent.target_estimator.qnet = copy.deepcopy(q_net)
    agent.policy_network = sl_net

    print(f"Model NFSP Unificat carregat des de: {ruta}")
    return _RLCardModelAdapter(agent, env_wrapped._extract_state)


def _crear_dqn(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Crea un model DQN unificat (integra el COS i el MLP).
    """
    import torch
    import copy
    from rlcard.agents.dqn_agent import DQNAgent
    from RL.models.xarxa_unificada import XarxaUnificada

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model DQN a: {ruta}")

    hidden_layers = spec.get("hidden_layers", _DEFAULT_HIDDEN_LAYERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Entorn aplanat (233 dimensions)
    env_wrapped = _crear_env_temp(env_config)

    # Agent DQN base
    agent = DQNAgent(
        num_actions=env_wrapped.num_actions,
        state_shape=env_wrapped.state_shape[0],
        mlp_layers=hidden_layers,
        device=device,
    )

    # Xarxa unificada (mode scratch)
    xarxa = XarxaUnificada(
        n_actions=env_wrapped.num_actions,
        mlp_layers=hidden_layers,
        mode="scratch",
        device=device,
        output="q"
    )

    # Carregar pesos
    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    q_sd = checkpoint.get("q_net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    xarxa.load_state_dict(q_sd)

    # Injectar estimadors a l'agent
    agent.q_estimator.qnet = xarxa
    agent.target_estimator.qnet = copy.deepcopy(xarxa)

    print(f"Model DQN Unificat carregat des de: {ruta}")
    return _RLCardModelAdapter(agent, env_wrapped._extract_state)
