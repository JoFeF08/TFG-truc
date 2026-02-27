from __future__ import annotations

from entorn.cartes_accions import ACTION_LIST
from controlador.interficie_model import Model
from vista.interficie_vista import Vista


class Controlador:
    """Controlador únic que funciona amb qualsevol Vista i Model"""

    def __init__(self, vista: Vista, model: Model) -> None:
        self.vista = vista
        self.model = model

    def executar_partida(self) -> None:
        config = self.vista.demanar_config()
        self.model.iniciar(config)

        while not self.model.es_final():
            pid = self.model.get_jugador_actual()
            if self.model.es_huma(pid):
                estat = self.model.get_estat(pid)
                self.vista.mostrar_estat(estat)
                accio = self.vista.escollir_accio(estat["accions_legals"], estat)
                self.model.aplicar_accio(accio)
                self.vista.mostrar_accio(pid, ACTION_LIST[accio], es_bot=False)
            else:
                accio, nom = self.model.get_accio_bot(pid)
                self.model.aplicar_accio(accio)
                self.vista.mostrar_accio(pid, nom, es_bot=True)

        resultat = self.model.get_resultat()
        self.vista.mostrar_fi_partida(resultat["score"], resultat["payoffs"])

    def bucle_principal(self) -> None:
        while True:
            self.executar_partida()
            if not self.vista.demanar_repetir():
                self.vista.mostrar_sortint()
                break
