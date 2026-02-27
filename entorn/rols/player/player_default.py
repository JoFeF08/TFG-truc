from entorn.rols.player.player import TrucPlayer
from entorn.rols.player.models import RandomModel


class DefaultPlayer(TrucPlayer):
    """
    Jugador que delega la tria d'acció en un model injectat.
    Si model és None, s'usa RandomModel per defecte.
    """

    def __init__(self, player_id, np_random, model=None):
        super().__init__(player_id, np_random)
        self.model = model if model is not None else RandomModel(np_random=np_random)

    def triar_accio(self, estat):
        return self.model.triar_accio(estat)
