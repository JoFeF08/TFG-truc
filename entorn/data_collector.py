"""
DataCollector: guarda cada transició (s, a) de les partides en fitxers JSON
per poder entrenar el model més endavant.
"""
import json
from datetime import datetime
from pathlib import Path


class DataCollector:
    """Recull transicions (state, action, player_id, ...) i desa la partida en JSON."""

    def __init__(self, directori: str = "dades_partides"):
        self.directori = Path(directori)
        self.directori.mkdir(parents=True, exist_ok=True)
        self.partida_id = None
        self.transicions = []
        self.payoffs = None

    def inici_partida(self):
        """Crida al començar una partida nova."""
        self.partida_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transicions = []
        self.payoffs = None

    def registrar(self, state: dict, action: int, player_id: int,
                  legal_actions: list, reward: float = 0, done: bool = False):
        """Registra una transició."""
        self.transicions.append({
            "state": state,
            "action": action,
            "player_id": player_id,
            "legal_actions": legal_actions,
            "reward": reward,
            "done": done,
        })

    def finalitzar_partida(self, payoffs: list):
        """Crida quan acaba la partida; desa el fitxer JSON."""
        self.payoffs = payoffs
        if not self.partida_id:
            return
        path = self.directori / f"partida_{self.partida_id}.json"
        d = {
            "partida_id": self.partida_id,
            "payoffs": self.payoffs,
            "transicions": self.transicions,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)


def carregar_partides(directori: str = "dades_partides"):
    """Carrega totes les partides guardades (lista de dicts)."""
    d = Path(directori)
    if not d.is_dir():
        return []
    partides = []
    for path in sorted(d.glob("partida_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                partides.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return partides
