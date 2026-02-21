from entorn.rols.player.player import TrucPlayer
from entorn.interficies import EscollidorAccio


class HumanPlayer(TrucPlayer):
    """Jugador humà que delega l'escollida d'acció en un EscollidorAccio"""

    def __init__(self, player_id, np_random, escollidor: EscollidorAccio):
        super().__init__(player_id, np_random)
        self.escollidor = escollidor

    def triar_accio(self, state):
        """Delega en l'escollidor"""
        return self.escollidor.escollir_accio(state["accions_legals"], state)
