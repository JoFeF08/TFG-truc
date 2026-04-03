import os
from typing import Any

from RL.models.core.loader import TrucModel
from RL.models.core.obs_adapter import crear_env_aplanat


class _PPOAdapter:

    def __init__(self, agent: Any, state_extractor):
        self._agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat: dict[str, Any]) -> int:
        state = self._extract(estat)
        action, _ = self._agent.eval_step(state)
        return int(action)


def _crear_ppo_mlp(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    import torch
    from RL.models.model_propi.model_ppo.ppo.cap_ppo_mlp import PPOMlpNet
    from RL.models.model_propi.model_ppo.ppo.agent_ppo_mlp import PPOMlpAgent

    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        if not os.path.isabs(ruta):
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            ruta_absoluta = os.path.join(root_path, ruta)
            if os.path.exists(ruta_absoluta):
                ruta = ruta_absoluta

        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No s'ha trobat el model PPO a: {ruta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_wrapped = crear_env_aplanat(env_config)

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)

    if 'actor.4.weight' in checkpoint:
        n_accions_model = checkpoint['actor.4.weight'].shape[0]
    else:
        n_accions_model = env_wrapped.num_actions

    net = PPOMlpNet(device=device)
    net.load_state_dict(checkpoint)
    agent = PPOMlpAgent(net, num_actions=n_accions_model, device=device)

    print(f"Model PPO MLP carregat des de: {ruta} (Mida: {n_accions_model} accions)")
    return _PPOAdapter(agent, env_wrapped._extract_state)


def _crear_ppo_gru(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    import torch
    from RL.models.model_propi.model_ppo.ppo_gru.cap_ppo_gru import PPOGruNet
    from RL.models.model_propi.model_ppo.ppo_gru.agent_ppo_gru import PPOGruAgent

    ruta = spec.get("ruta", "best.pt")
    if not os.path.exists(ruta):
        if not os.path.isabs(ruta):
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            ruta_absoluta = os.path.join(root_path, ruta)
            if os.path.exists(ruta_absoluta):
                ruta = ruta_absoluta

        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No s'ha trobat el model PPO GRU a: {ruta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_wrapped = crear_env_aplanat(env_config)

    checkpoint = torch.load(ruta, map_location=device, weights_only=True)

    if 'actor.weight' in checkpoint:
        n_accions_model = checkpoint['actor.weight'].shape[0]
    else:
        n_accions_model = env_wrapped.num_actions

    hidden_size = spec.get("hidden_size", 256)

    net = PPOGruNet(n_actions=n_accions_model, hidden_size=hidden_size, device=device)
    net.load_state_dict(checkpoint)
    agent = PPOGruAgent(net, num_actions=n_accions_model, device=device)

    print(f"Model PPO GRU carregat des de: {ruta} (Mida: {n_accions_model} accions, Hidden: {hidden_size})")
    return _PPOAdapter(agent, env_wrapped._extract_state)
