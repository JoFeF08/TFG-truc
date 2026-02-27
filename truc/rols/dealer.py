from truc.cartes_accions import init_joc_cartes

class TrucDealer:
    """
    S'utilitza aquesta classe purament per a abstracció.
    """
    def __init__(self, np_random, n_cartes=3):
        self.np_random = np_random
        self.n_cartes = n_cartes
        self.cartes = init_joc_cartes()

    def shuffle(self):
        self.cartes = init_joc_cartes()  # Recrear baralla completa sempre
        self.np_random.shuffle(self.cartes)

    def deal_cards(self, player, num=None):
        if num is None: num = self.n_cartes
        for _ in range(num):
            if self.cartes:
                player.hand.append(self.cartes.pop())
            else:
                # TODO: Llançar excepció no hi ha cartes
                pass
