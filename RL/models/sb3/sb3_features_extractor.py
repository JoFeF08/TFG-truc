import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from RL.models.core.feature_extractor import CosMultiInput


class CosMultiInputSB3(BaseFeaturesExtractor):
    """
    Adaptador perquè SB3 pugui utilitzar CosMultiInput amb una observació
    Box(239,) aplanada (216 de obs_cartes + 23 de obs_context).

    Ús:
        policy_kwargs = dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256],
        )
        model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)

        # Opcionalment, després de construir el model:
        model.policy.features_extractor.carregar_pesos_preentrenats(ruta)
        model.policy.features_extractor.congelar_cos()
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cos = CosMultiInput()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cartes = observations[:, :216].view(-1, 6, 4, 9)
        context = observations[:, 216:]
        return self.cos(cartes, context)

    def carregar_pesos_preentrenats(self, ruta: str) -> None:
        """Carrega els pesos del cos des d'un arxiu .pth produït per preentrenar_cos.py."""
        state = torch.load(ruta, map_location="cpu")
        self.cos.load_state_dict(state)

    def congelar_cos(self) -> None:
        """Congela tots els paràmetres del cos (requires_grad=False)."""
        for p in self.cos.parameters():
            p.requires_grad = False
