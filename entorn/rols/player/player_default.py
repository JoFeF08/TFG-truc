from entorn.rols.player.player import TrucPlayer


class DefaultPlayer(TrucPlayer):
    """
    Jugador que delega la tria d'acció en un model injectat.
    Si model és None, tria una acció aleatòria entre les legals.
    """

    def __init__(self, player_id, np_random, model=None):
        super().__init__(player_id, np_random)
        self.model = model

    def triar_accio(self, estat):
        if self.model is not None:
            return self.model.triar_accio(estat)
        # Fallback: acció aleatòria
        accions_legals = estat['accions_legals']
        if len(accions_legals) == 1:
            return int(accions_legals[0])
        return int(self.np_random.choice(accions_legals))
