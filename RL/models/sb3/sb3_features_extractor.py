import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from RL.models.core.feature_extractor import CosMultiInput, DEFAULT_IN_CHANNELS, DEFAULT_CONTEXT_SIZE


class CosMultiInputSB3(BaseFeaturesExtractor):
    """
    Adaptador perquè SB3 pugui utilitzar CosMultiInput amb una observació
    Box(N,) aplanada (obs_cartes.flatten() + obs_context, vegeu
    joc.entorn.obs_builder). `in_channels`/`context_size` s'han de fer
    coincidir amb la configuració (`num_jugadors`/`senyes`) de l'entorn.

    Ús:
        policy_kwargs = dict(
            features_extractor_class=CosMultiInputSB3,
            features_extractor_kwargs=dict(in_channels=3, context_size=17),
            net_arch=[256, 256],
        )
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)

        # Opcionalment, després de construir el model:
        model.policy.features_extractor.carregar_pesos_preentrenats(ruta)
        model.policy.features_extractor.congelar_cos()
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        in_channels: int = DEFAULT_IN_CHANNELS,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ):
        super().__init__(observation_space, features_dim)
        self.in_channels = in_channels
        self.context_size = context_size
        self.n_cartes_flat = in_channels * 4 * 9
        self.cos = CosMultiInput(in_channels=in_channels, context_size=context_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cartes = observations[:, :self.n_cartes_flat].view(-1, self.in_channels, 4, 9)
        context = observations[:, self.n_cartes_flat:]
        return self.cos(cartes, context)

    def carregar_pesos_preentrenats(self, ruta: str) -> None:
        """Carrega els pesos del cos des d'un arxiu .pth produït per preentrenar_cos.py."""
        state = torch.load(ruta, map_location="cpu")
        self.cos.load_state_dict(state)

    def congelar_cos(self) -> None:
        """Congela tots els paràmetres del cos (requires_grad=False)."""
        for p in self.cos.parameters():
            p.requires_grad = False
