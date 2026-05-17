from pathlib import Path
from RL.models.nfsp.average_policy import SLAgent
from RL.entrenament.entrenamentsComparatius.fase5.pool_selfplay import SelfPlayPool

class NFSPPool:
    """Combina SelfPlayPool (AgentRegles + snapshots PPO) amb SLAgent (average policy).
    Mostreja l'agent SL amb prob. eta; altrament delega a SelfPlayPool.sample."""

    def __init__(self, snapshot_dir: Path, sl_agent: SLAgent,
                 eta: float = 0.5, max_snapshots: int = 9):
        self.inner = SelfPlayPool(snapshot_dir, max_snapshots=max_snapshots)
        self.sl_agent = sl_agent
        self.eta = float(eta)

    def add_snapshot(self, model, step): 
        self.inner.add_snapshot(model, step)

    def sample(self, rng):
        if rng.random() < self.eta:
            return ("sl_avg", self.sl_agent)
        return self.inner.sample(rng)

    def get_recent(self, n=3): 
        return self.inner.get_recent(n)
        
    @property
    def n_snapshots(self): 
        return self.inner.n_snapshots
        
    def set_eta(self, eta): 
        self.eta = float(eta)
