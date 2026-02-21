import time

from entorn.rols.player.player import TrucPlayer


class RandomPlayer(TrucPlayer):
    """
    Un agent bàsic que tria accions aleatòriament.
    Serveix com a baseline per comparar agents més intel·ligents.
    """

    def __init__(self, player_id, np_random, time_ms=600):
        super().__init__(player_id, np_random)
        self._time_ms = time_ms

    def triar_accio(self, estat):
        time.sleep(self._time_ms / 500)
        accions_legals = estat['accions_legals']
        if len(accions_legals) == 1:
            return accions_legals[0]
        return self.np_random.choice(accions_legals)
