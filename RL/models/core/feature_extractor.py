import torch
import torch.nn as nn

# Mides per defecte: 2 jugadors, sense senyes (vegeu joc/entorn/obs_builder.py)
DEFAULT_IN_CHANNELS = 3   # 1 (mà pròpia) + num_jugadors (2)
DEFAULT_CONTEXT_SIZE = 17


def _calcular_mida_flatten(in_channels=DEFAULT_IN_CHANNELS, H=4, W=9) -> int:
    out_H = H - 2
    out_W = W - 4
    return 32 * out_H * out_W  # 320, independent d'in_channels

class CosMultiInput(nn.Module):
    """
    Cos compartit Multi-Input per al Truc.

    Entrades:
      · cartes  : Tensor (batch, in_channels, 4, 9) — Mapa de cartes 2D
      · context : Tensor (batch, context_size)      — Informació contextual

    Sortida:
      · Tensor (batch, 256) — Representació latent del joc

    `in_channels`/`context_size` depenen de `num_jugadors`/`senyes` de
    l'entorn (`joc.entorn.obs_builder.obs_shapes`); per defecte, 2 jugadors
    sense senyes (3, 17).
    """

    def __init__(self, in_channels: int = DEFAULT_IN_CHANNELS, context_size: int = DEFAULT_CONTEXT_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.context_size = context_size

        # Branca A: CNN sobre el mapa de cartes
        self.branca_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )
        dim_cnn = _calcular_mida_flatten(in_channels)

        # Branca B: Capa densa sobre el context
        self.branca_densa = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
        )
        dim_context = 32

        # Fusió
        dim_fusio = dim_cnn + dim_context
        self.fusio = nn.Sequential(
            nn.Linear(dim_fusio, 256),
            nn.ReLU(),
        )

    def forward(self, cartes: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x_cnn     = self.branca_cnn(cartes)
        x_context = self.branca_densa(context)
        x_fusio   = torch.cat([x_cnn, x_context], dim=1)
        return self.fusio(x_fusio)


# ---------------------------------------------------------------------------
# ModelPreEntrenament
# ---------------------------------------------------------------------------

class ModelPreEntrenament(nn.Module):
    """
    Model complet per al pre-entrenament supervisat.

    Combina el CosMultiInput amb tres caps de regressió/classificació per predir:
      - Els punts d'Envit (normalitzats, MSE).
      - Les accions legals permeses en l'estat actual (19 logits, BCE).
      - La força de cada carta per posició de mà (3 valors, MSE).
    """

    def __init__(self, in_channels: int = DEFAULT_IN_CHANNELS, context_size: int = DEFAULT_CONTEXT_SIZE):
        super().__init__()
        self.cos = CosMultiInput(in_channels=in_channels, context_size=context_size)
        self.cap_envido = nn.Linear(256, 1)
        self.cap_accions_legals = nn.Linear(256, 19)
        self.cap_forces = nn.Linear(256, 3)

    def forward(self, cartes: torch.Tensor, context: torch.Tensor):
        """
        Returns:
            val_envido     : (batch, 1)
            logits_accions : (batch, 19)
            val_forces     : (batch, 3)
        """
        latent = self.cos(cartes, context)
        val_envido = self.cap_envido(latent)
        logits_accions = self.cap_accions_legals(latent)
        val_forces = self.cap_forces(latent)
        return val_envido, logits_accions, val_forces
