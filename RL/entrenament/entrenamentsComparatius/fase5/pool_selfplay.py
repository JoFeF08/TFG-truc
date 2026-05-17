"""Pool mixt per a Fase 5 (Self-Play): combina AgentRegles i snapshots del PPO.

Cada SNAPSHOT_EVERY steps, el model actual es desa i s'afegeix al pool.
El sample() retorna uniformement entre totes les variants d'AgentRegles (sempre
presents) i els snapshots acumulats (finestra rodant de MAX_SNAPSHOTS).
"""
import random
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO

from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
from RL.entrenament.entrenamentsComparatius.fase4.pool_oponents import (
    POOL_OPONENTS, NOMS_VARIANTS, crear_oponent,
)


def _carregar_ppo_zip(ruta: str) -> PPO:
    """Carrega un snapshot PPO aplicant el monkey-patch d'optimizer (igual que loader.py)."""
    _orig = PPO.set_parameters

    def _sense_optimizer(self, load_path_or_dict, exact_match=True, device="auto"):  # exact_match ignorat intencionalment
        if isinstance(load_path_or_dict, dict):
            load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                 if "optimizer" not in k}
        return _orig(self, load_path_or_dict, exact_match=False, device=device)

    PPO.set_parameters = _sense_optimizer
    try:
        model = PPO.load(ruta,
                         custom_objects={"features_extractor_class": CosMultiInputSB3},
                         device="cpu")
    finally:
        PPO.set_parameters = _orig
    return model


class SelfPlayPool:
    """Pool d'oponents mixt: AgentRegles (fixos) + snapshots PPO (rodants)."""

    def __init__(self, snapshot_dir: Path, max_snapshots: int = 6):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        self._snapshots: list[tuple[str, SB3PPOEvalAgent]] = []

    def add_snapshot(self, model: PPO, step: int) -> None:
        path = self.snapshot_dir / f"snapshot_{step}.zip"
        model.save(str(path))
        sb3_model = _carregar_ppo_zip(str(path))
        agent = SB3PPOEvalAgent(sb3_model)
        nom = f"self_{step // 1_000_000}M"
        self._snapshots.append((nom, agent))
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)
        print(f"[SelfPlayPool] Snapshot guardat: {nom} ({len(self._snapshots)}/{self.max_snapshots})")

    def sample(self, rng: random.Random) -> tuple[str, Any]:
        """Retorna (nom, agent) mostrejat uniformement entre AgentRegles i snapshots."""
        candidats_regles = [(n, crear_oponent(n, seed=rng.randint(0, 2**31 - 1)))
                            for n, _ in POOL_OPONENTS]
        candidats = candidats_regles + self._snapshots
        return rng.choice(candidats)

    def get_recent(self, n: int = 3) -> list[tuple[str, Any]]:
        """Retorna els N snapshots més recents (per avaluació self-play)."""
        return self._snapshots[-n:]

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)
