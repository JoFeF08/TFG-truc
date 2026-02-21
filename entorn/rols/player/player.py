from abc import ABC, abstractmethod


class TrucPlayer(ABC):
    def __init__(self, player_id, np_random):
        self.player_id = player_id
        self.np_random = np_random
        self.hand = []
        self.ma_envits = []

    def get_player_id(self):
        return self.player_id

    @abstractmethod
    def triar_accio(self, estat):
        pass
