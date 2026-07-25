"""Política MaskablePPO amb `action_net` multi-cap.

En lloc d'una única capa lineal que prediu els logits de les 19 accions
alhora, es reparteix en 4 caps independents (cartes / truc / envit /
residual passar+senyes) sobre el mateix `latent` compartit (produït pel
`CosMultiInput` via `features_extractor`). Cada cap només veu i actualitza
els seus propis pesos, evitant que l'aprenentatge d'envidar (que depèn
sobretot de la força de la mà) interfereixi amb el de jugar cartes o el
bluff del truc. El masking d'accions legals (`ActionMasker`/`MaskablePPO`)
no canvia: segueix aplicant-se sobre els 19 logits resultants.
"""
from functools import partial

import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from joc.entorn.cartes_accions import ACTION_SPACE, ACTIONS_SIGNAL

_CARTES = ['play_card_0', 'play_card_1', 'play_card_2']
_TRUC = ['apostar_truc', 'vull_truc', 'fora_truc']
_ENVIT = ['apostar_envit', 'vull_envit', 'fora_envit']
_RESIDUAL = ['passar'] + list(ACTIONS_SIGNAL)


class MultiHeadActionNet(nn.Module):
    """4 sub-xarxes lineals (cartes/truc/envit/residual) concatenades en
    l'ordre d'ACTION_LIST perquè siguin compatibles amb el masking existent."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.n_actions = len(ACTION_SPACE)
        self._cartes_idx = [ACTION_SPACE[a] for a in _CARTES]
        self._truc_idx = [ACTION_SPACE[a] for a in _TRUC]
        self._envit_idx = [ACTION_SPACE[a] for a in _ENVIT]
        self._residual_idx = [ACTION_SPACE[a] for a in _RESIDUAL]

        assert sorted(self._cartes_idx + self._truc_idx + self._envit_idx + self._residual_idx) == list(range(self.n_actions))

        self.head_cartes = nn.Linear(latent_dim, len(self._cartes_idx))
        self.head_truc = nn.Linear(latent_dim, len(self._truc_idx))
        self.head_envit = nn.Linear(latent_dim, len(self._envit_idx))
        self.head_residual = nn.Linear(latent_dim, len(self._residual_idx))

        for head in (self.head_cartes, self.head_truc, self.head_envit, self.head_residual):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(latent.shape[0], self.n_actions, device=latent.device, dtype=latent.dtype)
        out[:, self._cartes_idx] = self.head_cartes(latent)
        out[:, self._truc_idx] = self.head_truc(latent)
        out[:, self._envit_idx] = self.head_envit(latent)
        out[:, self._residual_idx] = self.head_residual(latent)
        return out


class MultiHeadMaskableACPolicy(MaskableActorCriticPolicy):
    """`MaskableActorCriticPolicy` amb `action_net` substituït per `MultiHeadActionNet`."""

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        self.action_net = MultiHeadActionNet(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            # El multi-cap ja s'inicialitza ell mateix (gain=0.01, com feia
            # ortho_init per a action_net); aquí només tronc + valor.
            module_gains = {
                self.features_extractor: 2 ** 0.5,
                self.mlp_extractor: 2 ** 0.5,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = 2 ** 0.5
                module_gains[self.vf_features_extractor] = 2 ** 0.5

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
