import random
import numpy as np
import torch

class ReservoirBuffer:
    def __init__(self, capacity: int = 200_000, seed: int = 0):
        self.capacity = capacity
        self._buf: list[tuple[np.ndarray, int]] = []   # (obs_flat 240, action_id)
        self._n_seen = 0
        self._rng = random.Random(seed)

    def add(self, obs: np.ndarray, action: int) -> None:
        self._n_seen += 1
        if len(self._buf) < self.capacity:
            self._buf.append((obs.astype(np.float32), int(action)))
        else:
            j = self._rng.randint(0, self._n_seen - 1)
            if j < self.capacity:
                self._buf[j] = (obs.astype(np.float32), int(action))

    def add_batch(self, obs_batch, action_batch) -> None:
        # Assumint que obs_batch és una array 2D de numpy amb shape (N, 240)
        # i action_batch és iterable d'enters.
        for o, a in zip(obs_batch, action_batch):
            self.add(o, a)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._rng.sample(range(len(self._buf)), min(batch_size, len(self._buf)))
        obs = torch.from_numpy(np.stack([self._buf[i][0] for i in idx]))
        act = torch.tensor([self._buf[i][1] for i in idx], dtype=torch.long)
        return obs, act

    def __len__(self): 
        return len(self._buf)
