import os
from typing import Any, Callable

from RL.models.rlcard_legacy.loader import TrucModel

_DEFAULT_HIDDEN_LAYERS = [256, 256]


class _RLCardModelAdapter:

    def __init__(self, agent: Any, state_extractor: Callable[[dict[str, Any]], dict[str, Any]]):
        self._agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat: dict[str, Any]) -> int:
        rlcard_state = self._extract(estat)
        action, _ = self._agent.eval_step(rlcard_state)
        return int(action)


def _crear_env_temp(env_config: dict[str, Any]):

    from joc.entorn.env import TrucEnv
    from RL.models.rlcard_legacy.adapters.obs_adapter import wrap_env_aplanat

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
    from RL.models.core.base_networks import XarxaUnificada

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model NFSP a: {ruta}")

    hidden_layers = spec.get("hidden_layers", _DEFAULT_HIDDEN_LAYERS)
    hidden_layers_q = spec.get("hidden_layers_q", hidden_layers)
    hidden_layers_sl = spec.get("hidden_layers_sl", hidden_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_bn = spec.get("use_bn", True)

    # Entorn aplanat
    env_wrapped = _crear_env_temp(env_config)

    # Agent NFSP base
    agent = NFSPAgent(
        num_actions=env_wrapped.num_actions,
        state_shape=env_wrapped.state_shape[0],
        hidden_layers_sizes=hidden_layers_sl,
        q_mlp_layers=hidden_layers_q,
        device=device,
    )

    # Xarxes unificades
    q_net = XarxaUnificada(env_wrapped.num_actions, hidden_layers_q, "scratch", device=device, output="q", use_bn=use_bn)
    sl_net = XarxaUnificada(env_wrapped.num_actions, hidden_layers_sl, "scratch", device=device, output="policy", use_bn=use_bn)

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    
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
    from RL.models.core.base_networks import XarxaUnificada

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model DQN a: {ruta}")

    hidden_layers = spec.get("hidden_layers", _DEFAULT_HIDDEN_LAYERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_bn = spec.get("use_bn", True)

    # Entorn aplanat
    env_wrapped = _crear_env_temp(env_config)

    # Agent DQN base
    agent = DQNAgent(
        num_actions=env_wrapped.num_actions,
        state_shape=env_wrapped.state_shape[0],
        mlp_layers=hidden_layers,
        device=device,
    )

    # Xarxa unificada
    xarxa = XarxaUnificada(
        n_actions=env_wrapped.num_actions,
        mlp_layers=hidden_layers,
        mode="scratch",
        device=device,
        output="q",
        use_bn=use_bn
    )

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    q_sd = checkpoint.get("q_net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    xarxa.load_state_dict(q_sd)

    # Injectar estimadors a l'agent
    agent.q_estimator.qnet = xarxa
    agent.target_estimator.qnet = copy.deepcopy(xarxa)

    print(f"Model DQN Unificat carregat des de: {ruta}")
    return _RLCardModelAdapter(agent, env_wrapped._extract_state)


def _crear_ppo_mlp(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Crea un model PPO MLP unificat.
    """
    import torch
    from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import PPOMlpNet
    from RL.models.model_propi.model_ppo.ppo.agent_ppo_mlp import PPOMlpAgent

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        # Intentem buscar-lo a l'arrel si la ruta és relativa i no existeix
        if not os.path.isabs(ruta):
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            ruta_absoluta = os.path.join(root_path, ruta)
            if os.path.exists(ruta_absoluta):
                ruta = ruta_absoluta
        
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No s'ha trobat el model PPO a: {ruta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Entorn aplanat (necessari per a l'extractor de l'adapter)
    env_wrapped = _crear_env_temp(env_config)

   
    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    
    # Intentem inferir la mida des de l'última capa de l'actor
    if 'actor.4.weight' in checkpoint:
        n_accions_model = checkpoint['actor.4.weight'].shape[0]
    else:
        # Fallback al valor de l'entorn si no podem inferir-ho
        n_accions_model = env_wrapped.num_actions

    net = PPOMlpNet(n_actions=n_accions_model, device=device)
    net.load_state_dict(checkpoint)
    agent = PPOMlpAgent(net, num_actions=n_accions_model, device=device)

    print(f"Model PPO MLP carregat des de: {ruta} (Mida: {n_accions_model} accions)")
    return _RLCardModelAdapter(agent, env_wrapped._extract_state)


def _crear_ppo_gru(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    """
    Crea un model PPO GRU unificat.
    """
    import torch
    from RL.models.model_propi.model_ppo.ppo_gru.cap_ppo_gru import PPOGruNet
    from RL.models.model_propi.model_ppo.ppo_gru.agent_ppo_gru import PPOGruAgent

    ruta = spec.get("ruta", "best.pt")
    if not os.path.exists(ruta):
        if not os.path.isabs(ruta):
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            ruta_absoluta = os.path.join(root_path, ruta)
            if os.path.exists(ruta_absoluta):
                ruta = ruta_absoluta
        
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No s'ha trobat el model PPO GRU a: {ruta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Entorn aplanat
    env_wrapped = _crear_env_temp(env_config)

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)
    
    if 'actor.weight' in checkpoint:
        n_accions_model = checkpoint['actor.weight'].shape[0]
    else:
        n_accions_model = env_wrapped.num_actions

    # Hidden size per defecte 256
    hidden_size = spec.get("hidden_size", 256)
    
    net = PPOGruNet(n_actions=n_accions_model, hidden_size=hidden_size, device=device)
    net.load_state_dict(checkpoint)
    agent = PPOGruAgent(net, num_actions=n_accions_model, device=device)

    print(f"Model PPO GRU carregat des de: {ruta} (Mida: {n_accions_model} accions, Hidden: {hidden_size})")
    return _RLCardModelAdapter(agent, env_wrapped._extract_state)
