import random


class RandomAgent:
    """Agent que juga una acció legal a l'atzar. Oponent per defecte per als
    wrappers Gymnasium de self-play quan no se n'especifica cap altre."""
    use_raw = False

    def __init__(self, num_actions: int, seed=None):
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def step(self, state) -> int:
        return self.rng.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        return self.step(state), {}
