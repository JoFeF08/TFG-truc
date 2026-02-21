from entorn.rols.player.player import TrucPlayer


class RandomPlayer(TrucPlayer):
    """
    Un agent bàsic que tria accions aleatòriament.
    Serveix com a baseline per comparar agents més intel·ligents.
    """

    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)

    def triar_accio(self, estat):
        accions_legals = estat['accions_legals']
        
        if len(accions_legals) == 1:
            return accions_legals[0]

        return self.np_random.choice(accions_legals)
