import torch
import torch.nn as nn


def _calcular_mida_flatten(in_channels=6, H=4, W=9) -> int:
    out_H = H - 2
    out_W = W - 4
    return 32 * out_H * out_W  # 320

class CosMultiInput(nn.Module):
    """
    Cos compartit Multi-Input per al Truc.

    Entrades:
      · cartes  : Tensor (batch, 6, 4, 9)  — Mapa de cartes 2D
      · context : Tensor (batch, 23)       — Informació contextual

    Sortida:
      · Tensor (batch, 128) — Representació latent del joc
    """

    def __init__(self):
        super().__init__()

        # Branca A: CNN sobre el mapa de cartes
        self.branca_cnn = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )
        dim_cnn = _calcular_mida_flatten()

        # Branca B: Capa densa sobre el context
        self.branca_densa = nn.Sequential(
            nn.Linear(23, 32),
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

    def __init__(self):
        super().__init__()
        self.cos = CosMultiInput()
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
